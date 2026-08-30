"""Drafts a Facebook post for a vacant unit and prompts on the command line
for human approval before it "publishes" -- dry-run by default, or a real
post to your Page with --live (requires FACEBOOK_PAGE_ID and
FACEBOOK_PAGE_ACCESS_TOKEN -- see README.md).

Usage:
    python post_ad.py --unit "Unit 3" --rent 2100 --available "October 1" \\
        --bedrooms 2 --bathrooms 1 --features "in-unit laundry,pet friendly"

    python post_ad.py --unit "Unit 3" --rent 2100 --live   # posts for real
"""
from __future__ import annotations

import argparse
import sys
import uuid

from langgraph.types import Command

from graph.ad_build import build_ad_graph
from graph.nodes.scheduler import FacebookPublishError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", default="Unit 3")
    parser.add_argument("--rent", type=float, default=2100.0)
    parser.add_argument("--available", default="October 1")
    parser.add_argument("--bedrooms", type=int)
    parser.add_argument("--bathrooms", type=float)
    parser.add_argument("--features", default="", help="Comma-separated list, e.g. 'in-unit laundry,pet friendly'")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Actually post to Facebook once approved (default is dry-run).",
    )
    return parser.parse_args()


def prompt_ad_approval(payload: dict) -> dict:
    print("\n--- Human approval needed: Facebook post ---")
    print(f"Campaign: {payload['unit']} ({payload['campaign_id']})\n")
    print(payload["content"])
    answer = input("\nPost as-is, edit, or discard? [post/edit/discard]: ").strip().lower()
    if answer.startswith("e"):
        edited = input("New copy: ").strip()
        return {"approved": True, "edited_content": edited}
    if answer.startswith("d"):
        return {"approved": False}
    return {"approved": True}


def run_demo() -> None:
    args = parse_args()

    listing_facts = {}
    if args.bedrooms is not None:
        listing_facts["bedrooms"] = args.bedrooms
    if args.bathrooms is not None:
        listing_facts["bathrooms"] = args.bathrooms
    if args.features:
        listing_facts["features"] = [f.strip() for f in args.features.split(",") if f.strip()]

    graph = build_ad_graph()
    campaign_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": campaign_id}}

    campaign = {
        "campaign_id": campaign_id,
        "unit": args.unit,
        "monthly_rent": args.rent,
        "available_date": args.available,
        "listing_facts": listing_facts,
        "dry_run": not args.live,
    }

    result = graph.invoke(campaign, config=config)
    while "__interrupt__" in result:
        pending = result["__interrupt__"][0]
        response = prompt_ad_approval(pending.value)
        try:
            result = graph.invoke(Command(resume=response), config=config)
        except FacebookPublishError as exc:
            print(f"\nCouldn't publish to Facebook: {exc}")
            sys.exit(1)

    print("\n--- Final state ---")
    print(f"Approved: {result.get('approved')}")
    print(f"Publish result: {result.get('publish_result')}")
    print("(see audit_log.jsonl for the full decision trail)")


if __name__ == "__main__":
    run_demo()
