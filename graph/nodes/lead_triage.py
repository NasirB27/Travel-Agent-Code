"""Categorizes an inbound DM/email into QUALIFIED / FOLLOW_UP / SUPPORT / SPAM.

Per docs/architecture.md, this reimplements vercel-labs/lead-agent's
categorization pattern rather than depending on it directly — that repo is
archived/read-only, so it's a reference architecture to borrow from, not a
package to install. Falls back to a deterministic keyword classifier when
no ANTHROPIC_API_KEY is configured, matching the pattern used in
application_parser.py so the graph stays runnable and testable offline.
"""
from __future__ import annotations

import json
import os

from graph.lead_state import LeadCategory, LeadState

try:
    import anthropic
except ImportError:  # pragma: no cover - exercised when dependency absent
    anthropic = None

CLASSIFICATION_SYSTEM_PROMPT = """\
You triage inbound messages for a small residential landlord (a 4-unit
rental property). Classify the message into exactly one category:
- QUALIFIED: a genuine rental inquiry with enough detail to act on (e.g.
  desired move-in date, budget/income, household size, or an explicit
  request to apply).
- FOLLOW_UP: a genuine rental inquiry lacking enough detail to act on yet
  (e.g. "is this still available?").
- SUPPORT: an existing tenant reporting a maintenance/support issue.
- SPAM: unrelated, promotional, or clearly automated content.

Return ONLY a JSON object with fields:
- category (one of QUALIFIED, FOLLOW_UP, SUPPORT, SPAM)
- reasoning (one sentence)
"""

_SPAM_KEYWORDS = ("seo", "backlink", "crypto", "unsubscribe", "% off", "click here", "loan offer")
_SUPPORT_KEYWORDS = (
    "leak", "broken", "repair", "maintenance", "no heat", "no hot water",
    "not working", "clogged", "pest", "mice", "roach",
)
_QUALIFIED_KEYWORDS = (
    "move-in", "move in", "budget", "income", "credit score", "apply",
    "application", "when can i see", "lease",
)


def _classify_with_claude(message: str) -> dict:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=512,
        system=CLASSIFICATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": message}],
    )
    return json.loads(response.content[0].text)


def _classify_with_fallback(message: str) -> dict:
    text = message.lower()

    if any(keyword in text for keyword in _SPAM_KEYWORDS):
        return {"category": "SPAM", "reasoning": "Matched spam/promotional keywords"}
    if any(keyword in text for keyword in _SUPPORT_KEYWORDS):
        return {"category": "SUPPORT", "reasoning": "Matched maintenance/support keywords"}
    if any(keyword in text for keyword in _QUALIFIED_KEYWORDS):
        return {"category": "QUALIFIED", "reasoning": "Includes actionable rental-inquiry details"}
    return {"category": "FOLLOW_UP", "reasoning": "Rental inquiry without enough detail to act on yet"}


def classify_lead_node(state: LeadState) -> LeadState:
    message = state["message"]
    if anthropic is not None and os.environ.get("ANTHROPIC_API_KEY"):
        result = _classify_with_claude(message)
    else:
        result = _classify_with_fallback(message)

    category: LeadCategory = result["category"]
    return {**state, "category": category, "reasoning": result.get("reasoning", "")}
