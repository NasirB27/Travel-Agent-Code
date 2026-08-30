from __future__ import annotations

from graph.audit import log_event
from graph.state import ApplicantState


def lease_draft_node(state: ApplicantState) -> ApplicantState:
    # Placeholder draft. A later phase wires this to a real lease template
    # and e-sign integration (see docs/architecture.md, step 6).
    draft = (
        "LEASE AGREEMENT (DRAFT)\n"
        f"Tenant: {state['name']}\n"
        f"Unit: {state['unit']}\n"
        f"Monthly Rent: ${state['monthly_rent']:,.2f}\n"
        "-- placeholder text: replace with the real lease template --"
    )
    return {**state, "lease_draft": draft}


def rejection_log_node(state: ApplicantState) -> ApplicantState:
    log_event(
        state["applicant_id"],
        "applicant_rejected",
        actor="system",
        details={
            "reasons": state["screening"]["reasons"],
            "notes": state.get("decision_notes", ""),
        },
    )
    return state
