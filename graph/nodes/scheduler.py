"""Publishes an approved ad to the landlord's Facebook Page.

Dry-run by default (state["dry_run"] defaults to True): drafts and human
review happen the same way either way, but the actual Graph API call is
only made when dry_run is explicitly False. See README.md for how to
obtain FACEBOOK_PAGE_ID and FACEBOOK_PAGE_ACCESS_TOKEN.
"""
from __future__ import annotations

import os

import requests

from graph.ad_state import AdState
from graph.audit import log_event

DEFAULT_GRAPH_API_VERSION = "v21.0"


class FacebookPublishError(RuntimeError):
    """Raised when a Page post can't be published (missing credentials or
    the Graph API rejected the request)."""


def _post_to_facebook_page(message: str) -> str:
    page_id = os.environ.get("FACEBOOK_PAGE_ID")
    access_token = os.environ.get("FACEBOOK_PAGE_ACCESS_TOKEN")
    if not page_id or not access_token:
        raise FacebookPublishError(
            "FACEBOOK_PAGE_ID and FACEBOOK_PAGE_ACCESS_TOKEN must both be set "
            "to publish live -- see README.md for how to obtain a Page "
            "access token. Leave dry_run=True (the default) to draft/approve "
            "content without posting."
        )
    api_version = os.environ.get("FACEBOOK_GRAPH_API_VERSION", DEFAULT_GRAPH_API_VERSION)

    response = requests.post(
        f"https://graph.facebook.com/{api_version}/{page_id}/feed",
        data={"message": message, "access_token": access_token},
        timeout=15,
    )
    payload = response.json()
    if response.status_code >= 400 or "error" in payload:
        error_message = payload.get("error", {}).get("message", response.text)
        raise FacebookPublishError(f"Facebook Graph API rejected the post: {error_message}")
    return payload["id"]


def publish_node(state: AdState) -> AdState:
    if not state.get("approved"):
        result = "not published (rejected by human review)"
    elif state.get("dry_run", True):
        result = "simulated (dry-run, not actually posted)"
    else:
        post_id = _post_to_facebook_page(state["content"])
        result = f"posted (facebook post id: {post_id})"

    log_event(
        state["campaign_id"],
        "ad_published",
        actor="system",
        details={"dry_run": state.get("dry_run", True), "result": result},
    )
    return {**state, "publish_result": result}
