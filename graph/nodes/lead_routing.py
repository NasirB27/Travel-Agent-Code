from __future__ import annotations

from graph.audit import log_event
from graph.lead_state import LeadState


def spam_log_node(state: LeadState) -> LeadState:
    # SPAM is the one category that skips human review entirely -- discarding
    # junk isn't a consequential decision, so it doesn't need a HITL gate.
    log_event(
        state["lead_id"],
        "lead_marked_spam",
        actor="system",
        details={"reasoning": state["reasoning"]},
    )
    return state
