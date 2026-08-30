"""The human-in-the-loop checkpoints from docs/architecture.md.

Each node here pauses the graph with `interrupt()` and waits for a human
response before the pipeline can proceed — nothing downstream of these
nodes runs without an explicit human decision.
"""
from __future__ import annotations

from langgraph.types import interrupt

from graph.audit import log_event
from graph.state import ApplicantState


def screening_decision_node(state: ApplicantState) -> ApplicantState:
    screening = state["screening"]
    response = interrupt(
        {
            "type": "screening_decision",
            "applicant_id": state["applicant_id"],
            "name": state["name"],
            "unit": state["unit"],
            "recommendation": "approve" if screening["passed"] else "reject",
            "reasons": screening["reasons"],
            "financials": state.get("financials", {}),
        }
    )
    decision = response["decision"]
    notes = response.get("notes", "")
    log_event(
        state["applicant_id"],
        "screening_decision",
        actor="human",
        details={"decision": decision, "notes": notes},
    )
    return {**state, "decision": decision, "decision_notes": notes}


def lease_send_approval_node(state: ApplicantState) -> ApplicantState:
    response = interrupt(
        {
            "type": "lease_send_approval",
            "applicant_id": state["applicant_id"],
            "name": state["name"],
            "unit": state["unit"],
            "message": "Approve sending the lease agreement to the applicant?",
        }
    )
    approved = bool(response["approved"])
    log_event(
        state["applicant_id"],
        "lease_send_approval",
        actor="human",
        details={"approved": approved},
    )
    return {**state, "lease_sent": approved}
