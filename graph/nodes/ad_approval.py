"""Human review of drafted ad content before anything publishes -- the
"no post without approval" requirement from docs/architecture.md. Same
interrupt pattern as graph/nodes/approval.py and lead_reply.py.
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
            "platforms": state["platforms"],
            "content": state["content"],
        }
    )
    approved_platforms = list(response.get("approved_platforms", []))
    edited_content = response.get("edited_content", {})
    content = {**state["content"], **edited_content}

    log_event(
        state["campaign_id"],
        "ad_content_decision",
        actor="human",
        details={
            "approved_platforms": approved_platforms,
            "edited_platforms": list(edited_content.keys()),
        },
    )
    return {**state, "content": content, "approved_platforms": approved_platforms}
