"""Drafts platform-specific ad copy for a vacant unit.

Falls back to deterministic templates when no ANTHROPIC_API_KEY is
configured, matching the pattern used throughout this repo (see
application_parser.py, lead_triage.py). Image generation (the original
social-media-agents repo used Stability AI for this) is out of scope --
text copy only.
"""
from __future__ import annotations

import json
import os

from graph.ad_state import AdState

try:
    import anthropic
except ImportError:  # pragma: no cover - exercised when dependency absent
    anthropic = None

CONTENT_SYSTEM_PROMPT = """\
You write short listing ads for a residential landlord advertising a
vacant rental unit. Given the unit details, write ad copy for each
requested platform:
- twitter: under 280 characters, punchy, include a call to action.
- instagram: caption style with line breaks and 3-5 relevant hashtags.
- linkedin: a few sentences, professional tone, suitable for sharing with
  a personal/professional network.

Never invent details (price, availability, features) that weren't given.

Return ONLY a JSON object whose keys are the requested platform names and
whose values are the ad copy strings.
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
    parts.append(f"platforms: {', '.join(state['platforms'])}")
    return "\n".join(parts)


def _draft_with_claude(state: AdState) -> dict[str, str]:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=800,
        system=CONTENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _describe_listing(state)}],
    )
    return json.loads(response.content[0].text)


def _draft_with_fallback(state: AdState) -> dict[str, str]:
    facts = state.get("listing_facts", {})
    unit = state["unit"]
    rent = state["monthly_rent"]
    available = state.get("available_date", "now")
    features = facts.get("features", [])
    feature_line = ", ".join(features)

    detail_bits = []
    if "bedrooms" in facts:
        detail_bits.append(f"{facts['bedrooms']}BR")
    if "bathrooms" in facts:
        detail_bits.append(f"{facts['bathrooms']}BA")
    details = "/".join(detail_bits)

    headline = f"{unit} available {available} -- ${rent:,.0f}/mo"

    content: dict[str, str] = {}
    if "twitter" in state["platforms"]:
        body = f"{headline}. {details} in DC."
        if feature_line:
            body += f" {feature_line}."
        content["twitter"] = (body + " DM to tour!")[:280]
    if "instagram" in state["platforms"]:
        lines = [headline, details]
        if feature_line:
            lines.append(feature_line)
        lines.append("#DCRentals #ApartmentHunting #WashingtonDC")
        content["instagram"] = "\n".join(line for line in lines if line)
    if "linkedin" in state["platforms"]:
        sentence = f"Now available: {unit}, a {details} rental in DC for ${rent:,.0f}/mo, available {available}."
        if feature_line:
            sentence += f" Highlights: {feature_line}."
        sentence += " Reach out if you or someone in your network is looking."
        content["linkedin"] = sentence
    return content


def content_creator_node(state: AdState) -> AdState:
    if anthropic is not None and os.environ.get("ANTHROPIC_API_KEY"):
        content = _draft_with_claude(state)
    else:
        content = _draft_with_fallback(state)
    return {**state, "content": content}
