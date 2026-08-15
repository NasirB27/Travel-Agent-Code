"""Runs one demo applicant through the tenant-screening graph, prompting on
the command line for every human-in-the-loop decision.

This is the "prove the HITL loop works end to end" milestone from
docs/architecture.md's suggested implementation order (step 1). The CLI
prompts here stand in for the Slack/SMS approval channel that phase 1
calls out as a later swap-in — the graph itself doesn't know or care how
the human's response arrives, only that it does.

Usage:
    python main.py
"""
from __future__ import annotations

import uuid

from langgraph.types import Command

from graph.build import build_graph
from graph.state import ApplicantState

DEMO_DOCUMENTS: list[dict] = [
    {
        "doc_type": "paystub",
        "content": "monthly_gross_income: 6200\nemployer: Acme Consulting",
    },
    {
        "doc_type": "credit_report",
        "content": "credit_score: 705\nevictions_last_7_years: 0",
    },
]


def prompt_screening_decision(payload: dict) -> dict:
    print("\n--- Human approval needed: screening decision ---")
    print(f"Applicant: {payload['name']} (unit {payload['unit']})")
    print(f"System recommendation: {payload['recommendation'].upper()}")
    if payload["reasons"]:
        print("Reasons:")
        for reason in payload["reasons"]:
            print(f"  - {reason}")
    else:
        print("Reasons: meets all screening criteria")
    decision = input("Approve or reject? [approve/reject]: ").strip().lower()
    notes = input("Notes (optional): ").strip()
    return {"decision": "approved" if decision.startswith("a") else "rejected", "notes": notes}


def prompt_lease_send(payload: dict) -> dict:
    print("\n--- Human approval needed: send lease ---")
    print(payload["message"])
    answer = input("Send lease? [y/n]: ").strip().lower()
    return {"approved": answer.startswith("y")}


PROMPTERS = {
    "screening_decision": prompt_screening_decision,
    "lease_send_approval": prompt_lease_send,
}


def run_demo() -> None:
    graph = build_graph()
    applicant_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": applicant_id}}

    initial_state: ApplicantState = {
        "applicant_id": applicant_id,
        "name": "Jordan Rivera",
        "contact": "jordan@example.com",
        "unit": "Unit 2",
        "monthly_rent": 1800.0,
        "documents": DEMO_DOCUMENTS,
    }

    result = graph.invoke(initial_state, config=config)
    while "__interrupt__" in result:
        pending = result["__interrupt__"][0]
        prompter = PROMPTERS[pending.value["type"]]
        response = prompter(pending.value)
        result = graph.invoke(Command(resume=response), config=config)

    print("\n--- Final state ---")
    print(f"Decision: {result.get('decision')}")
    print(f"Lease sent: {result.get('lease_sent')}")
    print("(see audit_log.jsonl for the full decision trail)")


if __name__ == "__main__":
    run_demo()
