"""Publishes approved ad content -- dry-run only for now.

Real platform posting (Twitter/X, Instagram, LinkedIn) needs OAuth
credentials and API integration this project doesn't have yet, so
``dry_run`` defaults to True and live posting raises rather than silently
no-opping or guessing at an API call. This mirrors how lease sending and
lead-reply delivery are handled elsewhere in this repo: the human-approval
gate is real, the external delivery integration is a deliberately separate,
not-yet-built step (see docs/architecture.md).
"""
from __future__ import annotations

from graph.ad_state import AdState
from graph.audit import log_event


def publish_node(state: AdState) -> AdState:
    dry_run = state.get("dry_run", True)
    approved_platforms = state.get("approved_platforms", [])
    results: dict[str, str] = {}

    for platform in approved_platforms:
        if dry_run:
            results[platform] = "simulated (dry-run, not actually posted)"
        else:
            raise NotImplementedError(
                f"Live publishing to {platform!r} isn't wired up yet -- it needs "
                "that platform's API credentials/OAuth flow. Keep dry_run=True "
                "until that integration is built."
            )

    log_event(
        state["campaign_id"],
        "ad_published",
        actor="system",
        details={"dry_run": dry_run, "results": results},
    )
    return {**state, "publish_results": results}
