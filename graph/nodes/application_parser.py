"""Extracts screening-relevant financial data from an applicant's documents.

Per docs/architecture.md, no off-the-shelf tool in the original brief
actually does this (realtor-agent's "Document Agent" only handles real
estate transaction paperwork, not W-2s/paystubs/credit reports) — this is a
custom structured-extraction node, using the same "LLM + schema" pattern as
the travel-planning prototype this repo previously contained.

Falls back to a deterministic line parser when no ANTHROPIC_API_KEY is
configured, so the graph is runnable and testable offline.
"""
from __future__ import annotations

import json
import os
import re

from graph.state import ApplicantState, Financials

try:
    import anthropic
except ImportError:  # pragma: no cover - exercised when dependency absent
    anthropic = None

EXTRACTION_SYSTEM_PROMPT = """\
You are extracting financial screening data from a prospective tenant's
submitted documents (paystubs, W-2s, credit reports, background checks).
Return ONLY a JSON object with these fields:
- monthly_gross_income (number)
- employer (string)
- credit_score (integer)
- evictions_last_7_years (integer)
Use the most recent/authoritative document for each field. If a field
cannot be determined, omit it. Do not include any text outside the JSON
object.
"""

_LINE_PATTERN = re.compile(
    r"^\s*(monthly_gross_income|employer|credit_score|evictions_last_7_years)"
    r"\s*:\s*(.+)$",
    re.IGNORECASE,
)


def _extract_with_claude(documents_text: str) -> Financials:
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": documents_text}],
    )
    return json.loads(message.content[0].text)


def _extract_with_fallback(documents_text: str) -> Financials:
    """Deterministic parser used when Claude extraction isn't configured.

    Expects simple ``field: value`` lines, which is what the demo documents
    in this repo use (see main.py). A production build would still send
    real paystub/credit-report PDFs through Claude; this fallback only
    exists so the graph is runnable without an API key.
    """
    financials: Financials = {}
    for line in documents_text.splitlines():
        match = _LINE_PATTERN.match(line)
        if not match:
            continue
        field, raw_value = match.group(1).lower(), match.group(2).strip()
        if field == "monthly_gross_income":
            financials[field] = float(raw_value)
        elif field in ("credit_score", "evictions_last_7_years"):
            financials[field] = int(raw_value)
        else:
            financials[field] = raw_value
    return financials


def application_parser_node(state: ApplicantState) -> ApplicantState:
    documents_text = "\n".join(doc["content"] for doc in state.get("documents", []))
    if anthropic is not None and os.environ.get("ANTHROPIC_API_KEY"):
        financials = _extract_with_claude(documents_text)
    else:
        financials = _extract_with_fallback(documents_text)
    return {**state, "financials": financials}
