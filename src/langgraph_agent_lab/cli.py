"""CLI for the lab."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated

import typer
import yaml

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer_kind = cfg.get("checkpointer", "memory")
    checkpointer = build_checkpointer(checkpointer_kind, cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)

    # A persistent checkpointer (sqlite) keeps every thread's history on disk across
    # process runs. If we reused the same thread_id on every invocation of this
    # command, a second run would resume on top of the first run's accumulated state
    # (correct LangGraph semantics, but it inflates nodes_visited/retry_count in the
    # metrics below). Suffix each thread_id with a run id so every invocation starts
    # from a clean state per scenario, while checkpoints from all past runs still
    # remain on disk as separate threads — real persistence, still reproducible metrics.
    run_id = uuid.uuid4().hex[:8]
    metrics = []
    thread_ids: dict[str, str] = {}
    for scenario in scenarios:
        state = initial_state(scenario)
        thread_id = f"{state['thread_id']}-{run_id}"
        thread_ids[scenario.id] = thread_id
        run_config = {"configurable": {"thread_id": thread_id}}
        final_state = graph.invoke(state, config=run_config)
        metrics.append(
            metric_from_state(
                final_state, scenario.expected_route.value, scenario.requires_approval
            )
        )
    report = summarize_metrics(metrics)

    if checkpointer_kind == "sqlite":
        # Crash-resume / state-history evidence: open a *fresh* SqliteSaver connection
        # (as a restarted process would) and replay this run's checkpoint history for
        # one thread straight from disk, independent of the in-memory graph above.
        demo_scenario = next((s for s in scenarios if s.id == "S05_error"), scenarios[-1])
        demo_thread_id = thread_ids[demo_scenario.id]
        fresh_checkpointer = build_checkpointer("sqlite", cfg.get("database_url"))
        fresh_graph = build_graph(checkpointer=fresh_checkpointer)
        fresh_config = {"configurable": {"thread_id": demo_thread_id}}
        history = list(fresh_graph.get_state_history(fresh_config))
        report.resume_success = len(history) > 1
        typer.echo(
            f"Resume check on '{demo_scenario.id}' (thread={demo_thread_id}): replayed "
            f"{len(history)} checkpoints from a fresh SqliteSaver connection "
            "(state survives a process restart)."
        )

    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    typer.echo(f"Wrote metrics to {output}")


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


if __name__ == "__main__":
    app()
