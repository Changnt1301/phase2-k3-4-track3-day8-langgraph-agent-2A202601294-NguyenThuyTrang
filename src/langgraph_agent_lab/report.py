"""Report generation helper.

Renders a markdown report from MetricsReport data, following the structure of
reports/lab_report_template.md.
"""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data."""
    lines: list[str] = []

    lines.append("# Day 08 Lab Report")
    lines.append("")
    lines.append("## 1. Team / student")
    lines.append("")
    lines.append("- Name: Nguyễn Thuỳ Trang")
    lines.append(
        "- Repo/commit: Changnt1301/phase2-k3-4-track3-day8-langgraph-agent-2A202601294-NguyenThuyTrang "
        "@ 2ae2bf5"
    )
    lines.append("- Date: 25/08/2026")
    lines.append("")

    lines.append("## 2. Architecture")
    lines.append("")
    lines.append(
        "The graph is a LangGraph `StateGraph` over `AgentState` with 11 nodes: "
        "`intake`, `classify`, `tool`, `evaluate`, `answer`, `clarify` "
        "(`ask_clarification_node`), `risky_action`, `approval`, `retry` "
        "(`retry_or_fallback_node`), `dead_letter`, `finalize`.\n\n"
        "`START -> intake -> classify`, then the conditional edge `route_after_classify` "
        "dispatches on the LLM-classified `route`: `simple -> answer`, "
        "`tool -> tool -> evaluate`, `missing_info -> clarify`, "
        "`risky -> risky_action -> approval`, `error -> retry`.\n\n"
        "`evaluate` gates a bounded retry loop back to `tool` via `route_after_evaluate` "
        "(`needs_retry -> retry`, else `-> answer`). `retry` is bounded by "
        "`route_after_retry` (`attempt < max_attempts -> tool`, else `-> dead_letter`), so "
        "the loop always terminates. `approval` branches on `route_after_approval` "
        "(`approved -> tool`, `rejected -> clarify`) — a risky action never reaches `tool` "
        "without an approval record. Every branch converges on `finalize -> END`.\n\n"
        "`classify_node` and `answer_node` call a real LLM through `get_llm()` "
        "(`llm.py`, provider selected from `.env`): `classify_node` uses "
        "`.with_structured_output(ClassificationResult)` for reliable route selection, "
        "and `answer_node` generates a response grounded in `tool_results` and `approval`."
    )
    lines.append("")

    lines.append("## 3. State schema")
    lines.append("")
    lines.append("| Field | Reducer | Why |")
    lines.append("|---|---|---|")
    lines.append("| `messages` | append | audit trail of node activity |")
    lines.append(
        "| `tool_results` | append | a retry loop calls `tool_node` more than once; "
        "history is preserved |"
    )
    lines.append("| `errors` | append | accumulate every transient failure across retries |")
    lines.append("| `events` | append | full audit log consumed by `metrics.py` and grading |")
    lines.append("| `route` | overwrite | only the current classification matters |")
    lines.append(
        "| `evaluation_result` | overwrite | only the latest tool-result verdict "
        "drives the retry gate |"
    )
    lines.append("| `pending_question` | overwrite | one clarification question per run |")
    lines.append("| `proposed_action` | overwrite | one risky action under review at a time |")
    lines.append("| `approval` | overwrite | the latest approval decision |")
    lines.append("| `attempt` | overwrite | a monotonically incremented counter, not a list |")
    lines.append("")

    lines.append("## 4. Scenario results")
    lines.append("")
    lines.append(f"- Total scenarios: {metrics.total_scenarios}")
    lines.append(f"- Success rate: {metrics.success_rate:.2%}")
    lines.append(f"- Avg nodes visited: {metrics.avg_nodes_visited:.2f}")
    lines.append(f"- Total retries: {metrics.total_retries}")
    lines.append(f"- Total interrupts/approvals: {metrics.total_interrupts}")
    lines.append(f"- Resume success demonstrated: {metrics.resume_success}")
    lines.append("")
    lines.append(
        "| Scenario | Expected route | Actual route | Success | Retries | "
        "Interrupts | Latency (ms) |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for item in metrics.scenario_metrics:
        status = "✅" if item.success else "❌"
        lines.append(
            f"| {item.scenario_id} | {item.expected_route} | {item.actual_route} | "
            f"{status} | {item.retry_count} | {item.interrupt_count} | {item.latency_ms} |"
        )
    lines.append("")

    lines.append("## 5. Failure analysis")
    lines.append("")
    lines.append("At least two failure modes considered by this design:")
    lines.append("")
    lines.append(
        "1. **Tool / transient failure.** `tool_node` simulates a transient failure for "
        "`error`-route queries on the first two attempts (returns a result containing the "
        "`ERROR` marker). `evaluate_node` detects the marker and sets "
        "`evaluation_result=needs_retry`, sending the graph back through "
        "`retry_or_fallback_node`, which increments `attempt`. `route_after_retry` bounds "
        "this loop (`attempt < max_attempts`); once exhausted (see `S07_dead_letter`, which "
        "sets `max_attempts=1`) it routes to `dead_letter_node` instead of looping forever, "
        "and `dead_letter_node` sets a `final_answer` explaining the escalation."
    )
    lines.append(
        "2. **Risky action without approval.** `risky_action_node` never calls the tool "
        "directly — it only records a `proposed_action` description and hands control to "
        "`approval_node`. `route_after_approval` proceeds to `tool` only when "
        "`approval['approved']` is true; a rejection routes to `clarify` instead. This means "
        "a destructive action (refund, deletion, outbound email) can never execute without an "
        "explicit approval record present in state."
    )
    lines.append("")

    lines.append("## 6. Persistence / recovery evidence")
    lines.append("")
    lines.append(
        "The graph is compiled with a checkpointer selected via `configs/lab.yaml`'s "
        "`checkpointer` key (`memory` or `sqlite`; see `persistence.py`). Each scenario run "
        "uses a distinct `thread_id` (`thread-<scenario_id>`), passed as "
        "`config={'configurable': {'thread_id': ...}}` on every `graph.invoke()` call, so "
        "LangGraph checkpoints state per thread rather than sharing one history across "
        "scenarios. With `checkpointer: sqlite`, checkpoints are written to "
        "`outputs/checkpoints.db` (WAL mode) on disk.\n\n"
        f"**Resume evidence**: `resume_success={metrics.resume_success}`. `cli.py`'s "
        "`run-scenarios` command opens a *second, independent* `SqliteSaver` connection "
        "after the run completes (exactly what a restarted process would do) and calls "
        "`graph.get_state_history()` on the `S05_error` thread — a scenario with a retry "
        "loop, so it has several checkpoints. Every checkpoint is read back successfully "
        "from disk through the fresh connection, proving the run's state survives a process "
        "restart rather than only living in memory."
    )
    lines.append("")

    lines.append("## 7. Extension work")
    lines.append("")
    lines.append(
        "- **SQLite persistence** (`persistence.py`): `SqliteSaver(conn=sqlite3.connect(...))` "
        "with WAL mode, selected via `configs/lab.yaml: checkpointer: sqlite`."
    )
    lines.append("")

    lines.append("## 8. Improvement plan")
    lines.append("")
    lines.append(
        "With one more day: upgrade `evaluate_node` from a heuristic ERROR-substring check to "
        "a real LLM-as-judge for higher-quality retry decisions; wire real `interrupt()`-based "
        "HITL behind `LANGGRAPH_INTERRUPT=true` with a small Streamlit approve/reject UI; add "
        "parallel tool fan-out with `Send()` for multi-lookup queries; and export a Mermaid "
        "diagram of the compiled graph (`graph.get_graph().draw_mermaid()`) into this report."
    )
    lines.append("")

    return "\n".join(lines)


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
