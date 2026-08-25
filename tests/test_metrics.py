from langgraph_agent_lab.metrics import metric_from_state, summarize_metrics
from langgraph_agent_lab.state import make_event


def test_metric_from_state_success() -> None:
    state = {
        "scenario_id": "S",
        "route": "simple",
        "final_answer": "ok",
        "events": [
            make_event("intake", "completed", "ok"),
            make_event("answer", "completed", "ok"),
        ],
        "errors": [],
        "approval": None,
    }
    metric = metric_from_state(state, expected_route="simple", approval_required=False)
    assert metric.success is True
    assert metric.nodes_visited == 2


def test_metric_from_state_route_mismatch() -> None:
    state = {
        "scenario_id": "S",
        "route": "tool",
        "final_answer": "ok",
        "events": [],
        "errors": [],
        "approval": None,
    }
    metric = metric_from_state(state, expected_route="simple", approval_required=False)
    assert metric.success is False


def test_summarize_metrics() -> None:
    state1 = {
        "scenario_id": "1",
        "route": "simple",
        "final_answer": "ok",
        "events": [],
        "errors": [],
        "approval": None,
    }
    state2 = {
        "scenario_id": "2",
        "route": "tool",
        "final_answer": None,
        "events": [],
        "errors": [],
        "approval": None,
    }
    m1 = metric_from_state(state1, "simple", False)
    m2 = metric_from_state(state2, "tool", False)
    report = summarize_metrics([m1, m2])
    assert report.total_scenarios == 2
    assert 0 <= report.success_rate <= 1
