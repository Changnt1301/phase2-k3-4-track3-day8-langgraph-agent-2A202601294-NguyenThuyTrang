# Day 08 Lab Report

## 1. Team / student

- Name: Nguyễn Thuỳ Trang
- Repo/commit: Changnt1301/phase2-k3-4-track3-day8-langgraph-agent-2A202601294-NguyenThuyTrang @ 2ae2bf5
- Date: 25/08/2026

## 2. Architecture

The graph is a LangGraph `StateGraph` over `AgentState` with 11 nodes: `intake`, `classify`, `tool`, `evaluate`, `answer`, `clarify` (`ask_clarification_node`), `risky_action`, `approval`, `retry` (`retry_or_fallback_node`), `dead_letter`, `finalize`.

`START -> intake -> classify`, then the conditional edge `route_after_classify` dispatches on the LLM-classified `route`: `simple -> answer`, `tool -> tool -> evaluate`, `missing_info -> clarify`, `risky -> risky_action -> approval`, `error -> retry`.

`evaluate` gates a bounded retry loop back to `tool` via `route_after_evaluate` (`needs_retry -> retry`, else `-> answer`). `retry` is bounded by `route_after_retry` (`attempt < max_attempts -> tool`, else `-> dead_letter`), so the loop always terminates. `approval` branches on `route_after_approval` (`approved -> tool`, `rejected -> clarify`) — a risky action never reaches `tool` without an approval record. Every branch converges on `finalize -> END`.

`classify_node` and `answer_node` call a real LLM through `get_llm()` (`llm.py`, provider selected from `.env`): `classify_node` uses `.with_structured_output(ClassificationResult)` for reliable route selection, and `answer_node` generates a response grounded in `tool_results` and `approval`.

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| `messages` | append | audit trail of node activity |
| `tool_results` | append | a retry loop calls `tool_node` more than once; history is preserved |
| `errors` | append | accumulate every transient failure across retries |
| `events` | append | full audit log consumed by `metrics.py` and grading |
| `route` | overwrite | only the current classification matters |
| `evaluation_result` | overwrite | only the latest tool-result verdict drives the retry gate |
| `pending_question` | overwrite | one clarification question per run |
| `proposed_action` | overwrite | one risky action under review at a time |
| `approval` | overwrite | the latest approval decision |
| `attempt` | overwrite | a monotonically incremented counter, not a list |

## 4. Scenario results

- Total scenarios: 7
- Success rate: 100.00%
- Avg nodes visited: 6.43
- Total retries: 3
- Total interrupts/approvals: 2
- Resume success demonstrated: True

| Scenario | Expected route | Actual route | Success | Retries | Interrupts | Latency (ms) |
|---|---|---|---:|---:|---:|---:|
| S01_simple | simple | simple | ✅ | 0 | 0 | 0 |
| S02_tool | tool | tool | ✅ | 0 | 0 | 0 |
| S03_missing | missing_info | missing_info | ✅ | 0 | 0 | 0 |
| S04_risky | risky | risky | ✅ | 0 | 1 | 0 |
| S05_error | error | error | ✅ | 2 | 0 | 0 |
| S06_delete | risky | risky | ✅ | 0 | 1 | 0 |
| S07_dead_letter | error | error | ✅ | 1 | 0 | 0 |

## 5. Failure analysis

At least two failure modes considered by this design:

1. **Tool / transient failure.** `tool_node` simulates a transient failure for `error`-route queries on the first two attempts (returns a result containing the `ERROR` marker). `evaluate_node` detects the marker and sets `evaluation_result=needs_retry`, sending the graph back through `retry_or_fallback_node`, which increments `attempt`. `route_after_retry` bounds this loop (`attempt < max_attempts`); once exhausted (see `S07_dead_letter`, which sets `max_attempts=1`) it routes to `dead_letter_node` instead of looping forever, and `dead_letter_node` sets a `final_answer` explaining the escalation.
2. **Risky action without approval.** `risky_action_node` never calls the tool directly — it only records a `proposed_action` description and hands control to `approval_node`. `route_after_approval` proceeds to `tool` only when `approval['approved']` is true; a rejection routes to `clarify` instead. This means a destructive action (refund, deletion, outbound email) can never execute without an explicit approval record present in state.

## 6. Persistence / recovery evidence

The graph is compiled with a checkpointer selected via `configs/lab.yaml`'s `checkpointer` key (`memory` or `sqlite`; see `persistence.py`). Each scenario run uses a distinct `thread_id` (`thread-<scenario_id>`), passed as `config={'configurable': {'thread_id': ...}}` on every `graph.invoke()` call, so LangGraph checkpoints state per thread rather than sharing one history across scenarios. With `checkpointer: sqlite`, checkpoints are written to `outputs/checkpoints.db` (WAL mode) on disk.

**Resume evidence**: `resume_success=True`. `cli.py`'s `run-scenarios` command opens a *second, independent* `SqliteSaver` connection after the run completes (exactly what a restarted process would do) and calls `graph.get_state_history()` on the `S05_error` thread — a scenario with a retry loop, so it has several checkpoints. Every checkpoint is read back successfully from disk through the fresh connection, proving the run's state survives a process restart rather than only living in memory.

## 7. Extension work

- **SQLite persistence** (`persistence.py`): `SqliteSaver(conn=sqlite3.connect(...))` with WAL mode, selected via `configs/lab.yaml: checkpointer: sqlite`.

## 8. Improvement plan

With one more day: upgrade `evaluate_node` from a heuristic ERROR-substring check to a real LLM-as-judge for higher-quality retry decisions; wire real `interrupt()`-based HITL behind `LANGGRAPH_INTERRUPT=true` with a small Streamlit approve/reject UI; add parallel tool fan-out with `Send()` for multi-lookup queries; and export a Mermaid diagram of the compiled graph (`graph.get_graph().draw_mermaid()`) into this report.
