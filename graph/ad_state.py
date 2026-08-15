"""State schema for the vacancy-ad automation graph.

Per docs/architecture.md, this reimplements Klaudiusz321/social-media-agents'
content-creator/scheduler pattern natively rather than depending on that
repo directly: its trend-scanner is built for astronomy content and its
scheduler needs real platform credentials this project doesn't have yet.
What's preserved is the actual safety property that made it worth citing --
a dry-run mode and a human review step before anything publishes.
"""
from __future__ import annotations

from typing import Literal, TypedDict

Platform = Literal["twitter", "instagram", "linkedin"]


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
    platforms: list[Platform]
    content: dict[str, str]  # platform -> drafted copy
    approved_platforms: list[Platform]
    dry_run: bool
    publish_results: dict[str, str]  # platform -> outcome
