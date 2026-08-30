"""Exercises the Facebook ad graph end to end: content drafting, edited
copy, rejection, and the live Graph API publish path (with requests.post
monkeypatched -- no real network calls)."""
from __future__ import annotations

import json
import uuid

import pytest
from langgraph.types import Command

from graph.ad_build import build_ad_graph
from graph.nodes.scheduler import FacebookPublishError


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


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
        "dry_run": True,
    }
    campaign.update(overrides)
    return campaign


def test_content_drafted_reaches_approval():
    graph, config = _invoke_with_thread()
    result = graph.invoke(_base_campaign(), config=config)

    assert "__interrupt__" in result
    pending = result["__interrupt__"][0]
    assert pending.value["type"] == "ad_content_approval"
    assert "Unit 3" in pending.value["content"]


def test_approved_content_is_published_dry_run():
    graph, config = _invoke_with_thread()
    graph.invoke(_base_campaign(), config=config)

    final = graph.invoke(Command(resume={"approved": True}), config=config)
    assert "__interrupt__" not in final
    assert final["approved"] is True
    assert "dry-run" in final["publish_result"]


def test_human_edited_copy_is_used_for_the_published_content():
    graph, config = _invoke_with_thread()
    graph.invoke(_base_campaign(), config=config)

    final = graph.invoke(
        Command(resume={"approved": True, "edited_content": "Custom landlord-written copy."}),
        config=config,
    )
    assert final["content"] == "Custom landlord-written copy."


def test_rejecting_publishes_nothing():
    graph, config = _invoke_with_thread()
    graph.invoke(_base_campaign(), config=config)

    final = graph.invoke(Command(resume={"approved": False}), config=config)
    assert final["approved"] is False
    assert "not published" in final["publish_result"]


def test_live_publish_without_credentials_raises(monkeypatch):
    monkeypatch.delenv("FACEBOOK_PAGE_ID", raising=False)
    monkeypatch.delenv("FACEBOOK_PAGE_ACCESS_TOKEN", raising=False)

    graph, config = _invoke_with_thread()
    graph.invoke(_base_campaign(dry_run=False), config=config)

    with pytest.raises(FacebookPublishError, match="FACEBOOK_PAGE_ID"):
        graph.invoke(Command(resume={"approved": True}), config=config)


def test_live_publish_calls_the_graph_api(monkeypatch):
    monkeypatch.setenv("FACEBOOK_PAGE_ID", "123456")
    monkeypatch.setenv("FACEBOOK_PAGE_ACCESS_TOKEN", "test-token")

    captured = {}

    def fake_post(url, data=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        return _FakeResponse(200, {"id": "123456_987"})

    monkeypatch.setattr("graph.nodes.scheduler.requests.post", fake_post)

    graph, config = _invoke_with_thread()
    graph.invoke(_base_campaign(dry_run=False), config=config)
    final = graph.invoke(Command(resume={"approved": True}), config=config)

    assert final["publish_result"] == "posted (facebook post id: 123456_987)"
    assert captured["data"]["access_token"] == "test-token"
    assert "123456/feed" in captured["url"]


def test_live_publish_surfaces_graph_api_errors(monkeypatch):
    monkeypatch.setenv("FACEBOOK_PAGE_ID", "123456")
    monkeypatch.setenv("FACEBOOK_PAGE_ACCESS_TOKEN", "bad-token")

    def fake_post(url, data=None, timeout=None):
        return _FakeResponse(400, {"error": {"message": "Invalid OAuth access token."}})

    monkeypatch.setattr("graph.nodes.scheduler.requests.post", fake_post)

    graph, config = _invoke_with_thread()
    graph.invoke(_base_campaign(dry_run=False), config=config)

    with pytest.raises(FacebookPublishError, match="Invalid OAuth access token"):
        graph.invoke(Command(resume={"approved": True}), config=config)


def test_rejecting_a_live_campaign_never_calls_the_graph_api(monkeypatch):
    def fake_post(*args, **kwargs):
        raise AssertionError("Graph API should not be called for a rejected post")

    monkeypatch.setattr("graph.nodes.scheduler.requests.post", fake_post)

    graph, config = _invoke_with_thread()
    graph.invoke(_base_campaign(dry_run=False), config=config)
    final = graph.invoke(Command(resume={"approved": False}), config=config)
    assert "not published" in final["publish_result"]


def test_campaign_intake_rejects_missing_fields():
    graph, config = _invoke_with_thread()
    incomplete = _base_campaign()
    del incomplete["monthly_rent"]
    with pytest.raises(ValueError, match="monthly_rent"):
        graph.invoke(incomplete, config=config)
