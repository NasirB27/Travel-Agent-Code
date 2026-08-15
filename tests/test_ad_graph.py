"""Exercises the vacancy-ad graph end to end: content drafting for all three
platforms, partial approval, edited copy, and the dry-run publish gate that
refuses to actually post."""
from __future__ import annotations

import uuid

import pytest
from langgraph.types import Command

from graph.ad_build import build_ad_graph


def _invoke_with_thread():
    graph = build_ad_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return graph, config


def _base_campaign(**overrides) -> dict:
    campaign = {
        "campaign_id": str(uuid.uuid4()),
        "unit": "Unit 3",
        "monthly_rent": 2100.0,
        "available_date": "October 1",
        "listing_facts": {"bedrooms": 2, "bathrooms": 1, "features": ["pet friendly"]},
        "platforms": ["twitter", "instagram", "linkedin"],
        "dry_run": True,
    }
    campaign.update(overrides)
    return campaign


def test_content_drafted_for_every_requested_platform_reaches_approval():
    graph, config = _invoke_with_thread()
    result = graph.invoke(_base_campaign(), config=config)

    assert "__interrupt__" in result
    pending = result["__interrupt__"][0]
    assert pending.value["type"] == "ad_content_approval"
    for platform in ("twitter", "instagram", "linkedin"):
        assert platform in pending.value["content"]
    assert len(pending.value["content"]["twitter"]) <= 280


def test_partial_approval_only_publishes_approved_platforms():
    graph, config = _invoke_with_thread()
    graph.invoke(_base_campaign(), config=config)

    final = graph.invoke(
        Command(resume={"approved_platforms": ["twitter"], "edited_content": {}}),
        config=config,
    )
    assert "__interrupt__" not in final
    assert final["approved_platforms"] == ["twitter"]
    assert set(final["publish_results"].keys()) == {"twitter"}
    assert "dry-run" in final["publish_results"]["twitter"]


def test_human_edited_copy_is_used_for_the_published_platform():
    graph, config = _invoke_with_thread()
    graph.invoke(_base_campaign(), config=config)

    final = graph.invoke(
        Command(
            resume={
                "approved_platforms": ["linkedin"],
                "edited_content": {"linkedin": "Custom landlord-written copy."},
            }
        ),
        config=config,
    )
    assert final["content"]["linkedin"] == "Custom landlord-written copy."


def test_rejecting_all_platforms_publishes_nothing():
    graph, config = _invoke_with_thread()
    graph.invoke(_base_campaign(), config=config)

    final = graph.invoke(Command(resume={"approved_platforms": [], "edited_content": {}}), config=config)
    assert final["publish_results"] == {}


def test_live_publishing_is_refused_without_a_platform_integration():
    graph, config = _invoke_with_thread()
    graph.invoke(_base_campaign(dry_run=False), config=config)

    with pytest.raises(NotImplementedError, match="twitter"):
        graph.invoke(Command(resume={"approved_platforms": ["twitter"], "edited_content": {}}), config=config)


def test_campaign_intake_rejects_missing_platforms():
    graph, config = _invoke_with_thread()
    incomplete = _base_campaign()
    incomplete["platforms"] = []
    with pytest.raises(ValueError, match="platform"):
        graph.invoke(incomplete, config=config)
