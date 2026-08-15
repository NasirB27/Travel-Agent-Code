"""Runs one demo inbound lead through the triage graph, prompting on the
command line to approve (or edit) the suggested reply.

Companion to main.py (the applicant-screening demo) -- see
docs/architecture.md for how a QUALIFIED lead here becomes an applicant
there.

Usage:
    python triage_lead.py
"""
from __future__ import annotations

import uuid

from langgraph.types import Command

from graph.lead_build import build_lead_graph

DEMO_MESSAGE = (
    "Hi! I saw your listing for the 2BR unit. I'm looking to move in "
    "around October 1st, my income is about $95k/year, and I have great "
    "credit. Is it still available?"
)


def prompt_reply_approval(payload: dict) -> dict:
    print("\n--- Human approval needed: lead reply ---")
    print(f"Category: {payload['category']} ({payload['reasoning']})")
    print(f"Inbound message: {payload['message']}")
    print(f"Suggested reply: {payload['suggested_reply']}")
    answer = input("Send as-is, edit, or discard? [send/edit/discard]: ").strip().lower()
    if answer.startswith("e"):
        edited = input("New reply text: ").strip()
        return {"approved": True, "edited_reply": edited}
    if answer.startswith("d"):
        return {"approved": False}
    return {"approved": True}


def run_demo() -> None:
    graph = build_lead_graph()
    lead_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": lead_id}}

    result = graph.invoke(
        {
            "lead_id": lead_id,
            "source": "dm",
            "contact": "prospect@example.com",
            "message": DEMO_MESSAGE,
        },
        config=config,
    )

    while "__interrupt__" in result:
        pending = result["__interrupt__"][0]
        response = prompt_reply_approval(pending.value)
        result = graph.invoke(Command(resume=response), config=config)

    print("\n--- Final state ---")
    print(f"Category: {result.get('category')}")
    if "reply_approved" in result:
        print(f"Reply approved: {result['reply_approved']}")
        print(f"Final reply: {result.get('final_reply')}")
    else:
        print("Marked as spam -- no reply sent.")
    print("(see audit_log.jsonl for the full decision trail)")


if __name__ == "__main__":
    run_demo()
