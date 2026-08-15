"""Runs one demo vacancy-ad campaign through the pipeline, prompting on the
command line to approve (per-platform, with optional edits) before the
dry-run "publish" step.

Companion to main.py and triage_lead.py.

Usage:
    python post_ad.py
"""
from __future__ import annotations

import uuid

from langgraph.types import Command

from graph.ad_build import build_ad_graph

DEMO_CAMPAIGN = {
    "unit": "Unit 3",
    "monthly_rent": 2100.0,
    "available_date": "October 1",
    "listing_facts": {
        "bedrooms": 2,
        "bathrooms": 1,
        "features": ["in-unit laundry", "steps from the Metro", "pet friendly"],
    },
    "platforms": ["twitter", "instagram", "linkedin"],
    "dry_run": True,
}


def prompt_ad_approval(payload: dict) -> dict:
    print("\n--- Human approval needed: ad content ---")
    print(f"Campaign: {payload['unit']} ({payload['campaign_id']})")
    approved_platforms = []
    edited_content = {}
    for platform in payload["platforms"]:
        print(f"\n[{platform}]\n{payload['content'].get(platform, '(no draft)')}")
        answer = input(f"Approve for {platform}? [y/n/edit]: ").strip().lower()
        if answer.startswith("e"):
            edited_content[platform] = input("New copy: ").strip()
            approved_platforms.append(platform)
        elif answer.startswith("y"):
            approved_platforms.append(platform)
    return {"approved_platforms": approved_platforms, "edited_content": edited_content}


def run_demo() -> None:
    graph = build_ad_graph()
    campaign_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": campaign_id}}

    result = graph.invoke({**DEMO_CAMPAIGN, "campaign_id": campaign_id}, config=config)

    while "__interrupt__" in result:
        pending = result["__interrupt__"][0]
        response = prompt_ad_approval(pending.value)
        result = graph.invoke(Command(resume=response), config=config)

    print("\n--- Final state ---")
    print(f"Approved platforms: {result.get('approved_platforms')}")
    print(f"Publish results: {result.get('publish_results')}")
    print("(see audit_log.jsonl for the full decision trail)")


if __name__ == "__main__":
    run_demo()
