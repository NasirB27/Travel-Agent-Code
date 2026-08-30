"""Drafts a suggested reply for a triaged lead, then pauses for human
approval before it's considered ready to send — the same interrupt pattern
as the applicant pipeline's approval nodes (graph/nodes/approval.py).

A drafted reply never counts as sent on its own; nothing here actually
dispatches an email/DM yet (see docs/architecture.md's "Open questions" on
the eventual delivery channel). This node's job stops at producing a
human-approved (or human-edited) final reply.
"""
from __future__ import annotations

import os

from langgraph.types import interrupt

from graph.audit import log_event
from graph.lead_state import LeadState

try:
    import anthropic
except ImportError:  # pragma: no cover - exercised when dependency absent
    anthropic = None

REPLY_SYSTEM_PROMPT = """\
Draft a short, warm, professional reply from a residential landlord to a
prospective tenant or existing tenant's inbound message. Do not commit to
specific rent, move-in dates, or lease terms that weren't in the original
message -- ask clarifying questions instead where needed. Return only the
reply text, no preamble.
"""

_TEMPLATES = {
    "QUALIFIED": (
        "Thanks for reaching out! This unit is still available -- could "
        "you share your target move-in date and I'll follow up with next "
        "steps to apply?"
    ),
    "FOLLOW_UP": (
        "Thanks for your interest! Could you share a bit more -- your "
        "desired move-in date and household size -- so I can point you to "
        "the right unit?"
    ),
    "SUPPORT": (
        "Thanks for flagging this -- I'll look into it and follow up with "
        "next steps shortly."
    ),
}


def _draft_with_claude(message: str, category: str) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=300,
        system=REPLY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Category: {category}\nMessage: {message}"}],
    )
    return response.content[0].text.strip()


def draft_reply_node(state: LeadState) -> LeadState:
    if anthropic is not None and os.environ.get("ANTHROPIC_API_KEY"):
        reply = _draft_with_claude(state["message"], state["category"])
    else:
        reply = _TEMPLATES.get(state["category"], _TEMPLATES["FOLLOW_UP"])
    return {**state, "suggested_reply": reply}


def reply_approval_node(state: LeadState) -> LeadState:
    response = interrupt(
        {
            "type": "lead_reply_approval",
            "lead_id": state["lead_id"],
            "category": state["category"],
            "reasoning": state["reasoning"],
            "message": state["message"],
            "suggested_reply": state["suggested_reply"],
        }
    )
    approved = bool(response["approved"])
    final_reply = response.get("edited_reply") or state["suggested_reply"]
    log_event(
        state["lead_id"],
        "lead_reply_decision",
        actor="human",
        details={"approved": approved, "category": state["category"], "final_reply": final_reply},
    )
    return {**state, "reply_approved": approved, "final_reply": final_reply}
