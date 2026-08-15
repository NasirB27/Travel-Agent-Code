"""State schema for the vacancy-ad automation graph.

Facebook-only by design: this landlord advertises exclusively on their
Facebook Page, so the state is a single content draft rather than a
per-platform dict/list (see docs/architecture.md for the earlier
multi-platform version this replaced).
"""
from __future__ import annotations

from typing import TypedDict


class ListingFacts(TypedDict, total=False):
    bedrooms: int
    bathrooms: float
    features: list[str]  # e.g. ["in-unit laundry", "steps from the Metro"]


class AdState(TypedDict, total=False):
    campaign_id: str
    unit: str
    monthly_rent: float
    available_date: str
    listing_facts: ListingFacts
    content: str  # drafted Facebook post copy
    approved: bool
    dry_run: bool
    publish_result: str
