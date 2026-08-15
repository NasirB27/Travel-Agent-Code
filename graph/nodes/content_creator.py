"""Drafts Facebook post copy for a vacant unit.

Falls back to a deterministic template when no ANTHROPIC_API_KEY is
configured, matching the pattern used throughout this repo (see
application_parser.py, lead_triage.py). Image generation is out of scope
-- text copy only.
"""
from __future__ import annotations

import os

from graph.ad_state import AdState

try:
    import anthropic
except ImportError:  # pragma: no cover - exercised when dependency absent
    anthropic = None

CONTENT_SYSTEM_PROMPT = """\
You write a Facebook post advertising a vacant rental unit for a
residential landlord. Write a few sentences to a short paragraph, friendly
and inviting, ending with a clear call to action (e.g. "Comment or message
the page to schedule a tour!"). You may add a few relevant hashtags at the
end (e.g. #DCRentals). Never invent details (price, availability,
features) that weren't given.

Return only the post text -- no preamble, no JSON, no quotation marks
around it.
"""


def _describe_listing(state: AdState) -> str:
    facts = state.get("listing_facts", {})
    parts = [
        f"unit: {state['unit']}",
        f"monthly_rent: {state['monthly_rent']}",
        f"available_date: {state.get('available_date', 'now')}",
    ]
    if "bedrooms" in facts:
        parts.append(f"bedrooms: {facts['bedrooms']}")
    if "bathrooms" in facts:
        parts.append(f"bathrooms: {facts['bathrooms']}")
    if facts.get("features"):
        parts.append(f"features: {', '.join(facts['features'])}")
    return "\n".join(parts)


def _draft_with_claude(state: AdState) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=400,
        system=CONTENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _describe_listing(state)}],
    )
    return response.content[0].text.strip()


def _draft_with_fallback(state: AdState) -> str:
    facts = state.get("listing_facts", {})
    unit = state["unit"]
    rent = state["monthly_rent"]
    available = state.get("available_date", "now")
    features = facts.get("features", [])

    detail_bits = []
    if "bedrooms" in facts:
        detail_bits.append(f"{facts['bedrooms']}BR")
    if "bathrooms" in facts:
        detail_bits.append(f"{facts['bathrooms']}BA")
    details = "/".join(detail_bits)

    lines = [f"Now available: {unit} -- ${rent:,.0f}/mo, available {available}."]
    if details:
        lines[0] = f"Now available: {unit} ({details}) -- ${rent:,.0f}/mo, available {available}."
    if features:
        lines.append(f"Highlights: {', '.join(features)}.")
    lines.append("Comment or message the page to schedule a tour!")
    lines.append("#DCRentals #ApartmentHunting #WashingtonDC")
    return "\n\n".join(lines)


def content_creator_node(state: AdState) -> AdState:
    if anthropic is not None and os.environ.get("ANTHROPIC_API_KEY"):
        content = _draft_with_claude(state)
    else:
        content = _draft_with_fallback(state)
    return {**state, "content": content}
