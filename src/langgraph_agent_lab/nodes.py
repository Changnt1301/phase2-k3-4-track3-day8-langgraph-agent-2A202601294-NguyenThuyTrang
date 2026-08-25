"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, make_event


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── classify_node ─────────────────────────────────────────────────────

class ClassificationResult(BaseModel):
    """Structured output schema for intent classification."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description="The single best-matching route for this support query."
    )
    reasoning: str = Field(description="One short sentence explaining the classification.")


_CLASSIFY_PROMPT = """You are an intent classifier for a customer-support ticketing system.
Classify the customer query below into exactly ONE route:

- risky: The query asks for an action with real side effects — refunds, deletions,
  cancellations, sending emails/notifications, account changes. If a risky action is
  requested at all (even alongside other things), this wins.
- tool: The query asks to look up information that needs an external system — order
  status, tracking, account/search lookups. No side effects, just a read.
- missing_info: The query is too vague or incomplete to act on — it lacks the specific
  detail needed (e.g. "can you fix it?" without saying what "it" is).
- error: The query reports or describes a system failure, timeout, crash, or outage,
  and is not itself a request for a risky action.
- simple: A general question answerable directly, with no tool call or side-effecting
  action required.

Priority when more than one could apply: risky > tool > missing_info > error > simple.

Customer query:
\"\"\"{query}\"\"\"
"""


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM with structured output."""
    query = state.get("query", "")
    llm = get_llm()
    structured_llm = llm.with_structured_output(ClassificationResult)
    result = structured_llm.invoke(_CLASSIFY_PROMPT.format(query=query))
    route = result.route
    risk_level = "high" if route == "risky" else "low"
    return {
        "route": route,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "completed",
                f"classified as '{route}'",
                reasoning=result.reasoning,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulates a transient failure for the first two attempts of an `error`-route
    scenario, so the retry loop has something real to recover from.
    """
    attempt = state.get("attempt", 0)
    route = state.get("route", "")
    query = state.get("query", "")

    if route == "error" and attempt < 2:
        result = f"ERROR: transient tool failure on attempt {attempt} (query: {query[:60]!r})"
        return {
            "tool_results": [result],
            "events": [
                make_event(
                    "tool", "error", "simulated transient tool failure", attempt=attempt
                )
            ],
        }

    result = f"SUCCESS: mock tool result for {query[:60]!r} (attempt {attempt})"
    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", "tool call succeeded", attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Heuristic check (looks for the "ERROR" marker in the latest tool result).
    Acceptable for base score; LLM-as-judge is a bonus left as a future extension
    (see report.py's improvement plan).
    """
    tool_results = state.get("tool_results", []) or []
    latest = tool_results[-1] if tool_results else ""
    needs_retry = "ERROR" in latest
    evaluation_result = "needs_retry" if needs_retry else "success"
    return {
        "evaluation_result": evaluation_result,
        "events": [
            make_event(
                "evaluate",
                "completed",
                f"evaluation={evaluation_result}",
                latest_result=latest[:80],
            )
        ],
    }


_ANSWER_PROMPT = """You are a helpful customer-support agent. Write a concise, friendly final
response to the customer, grounded ONLY in the context below — do not invent order numbers,
amounts, or facts that are not present in the context.

Customer query:
\"\"\"{query}\"\"\"

Tool results (if any):
{tool_results}

Approval decision (only relevant if this was a risky action):
{approval}

Write the final response to the customer now. Do not include any preamble like "Here is the
response" — just the message itself.
"""


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM, grounded in tool_results/approval/query."""
    llm = get_llm()
    query = state.get("query", "")
    tool_results = state.get("tool_results", []) or []
    approval = state.get("approval")
    prompt = _ANSWER_PROMPT.format(
        query=query,
        tool_results="\n".join(tool_results) if tool_results else "(no tool was called)",
        approval=approval if approval else "(not applicable)",
    )
    response = llm.invoke(prompt)
    final_answer = response.content if hasattr(response, "content") else str(response)
    return {
        "final_answer": final_answer,
        "events": [make_event("answer", "completed", "final answer generated")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "").strip()
    question = (
        f'To help with "{query}", could you share a bit more detail — for example, '
        "which account, order, or item this is about, and exactly what you'd like us to do?"
    )
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "asked for clarification")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval."""
    query = state.get("query", "").strip()
    action = (
        f'Proposed action based on the request "{query}". This action has a real side effect '
        "(e.g. refund, deletion, or outbound email) and requires human approval before it runs."
    )
    return {
        "proposed_action": action,
        "events": [make_event("risky_action", "completed", "prepared risky action for approval")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default: mock approval (approved=True) so tests and CI run offline.
    Extension: LANGGRAPH_INTERRUPT=true switches to a real interrupt() pause.
    """
    proposed_action = state.get("proposed_action", "")

    if os.getenv("LANGGRAPH_INTERRUPT", "false").lower() == "true":
        from langgraph.types import interrupt

        decision = interrupt(
            {"proposed_action": proposed_action, "question": "Approve this action?"}
        )
        if isinstance(decision, dict):
            approved = bool(decision.get("approved", False))
            reviewer = decision.get("reviewer", "human-reviewer")
            comment = decision.get("comment", "")
        else:
            approved = bool(decision)
            reviewer = "human-reviewer"
            comment = ""
    else:
        approved = True
        reviewer = "mock-reviewer"
        comment = "auto-approved (mock mode)"

    approval = {"approved": approved, "reviewer": reviewer, "comment": comment}
    return {
        "approval": approval,
        "events": [make_event("approval", "completed", f"approval={approved}", reviewer=reviewer)],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt: increment the counter and log the transient failure."""
    attempt = state.get("attempt", 0) + 1
    message = f"retry attempt {attempt} scheduled after transient failure"
    return {
        "attempt": attempt,
        "errors": [message],
        "events": [make_event("retry", "completed", message, attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded."""
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    message = (
        f"We could not complete this request after {attempt} attempt(s) (limit {max_attempts}). "
        "It has been escalated to a human agent for follow-up."
    )
    return {
        "final_answer": message,
        "events": [
            make_event(
                "dead_letter", "completed", "max retries exceeded, escalated", attempt=attempt
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END."""
    return {
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
