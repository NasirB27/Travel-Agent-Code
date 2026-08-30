"""Human review of drafted Facebook post content before anything publishes
-- the "no post without approval" requirement from docs/architecture.md.
Same interrupt pattern as graph/nodes/approval.py and lead_reply.py.
"""
from __future__ import annotations

from langgraph.types import interrupt

from graph.ad_state import AdState
from graph.audit import log_event


def ad_approval_node(state: AdState) -> AdState:
    response = interrupt(
        {
            "type": "ad_content_approval",
            "campaign_id": state["campaign_id"],
            "unit": state["unit"],
            "content": state["content"],
        }
    )
    approved = bool(response["approved"])
    edited_content = response.get("edited_content")
    final_content = edited_content or state["content"]

    log_event(
        state["campaign_id"],
        "ad_content_decision",
        actor="human",
        details={"approved": approved, "edited": bool(edited_content)},
    )
    return {**state, "content": final_content, "approved": approved}
