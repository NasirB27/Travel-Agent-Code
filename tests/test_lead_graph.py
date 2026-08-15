"""Exercises the lead-triage graph end to end: classification into all four
categories, the SPAM short-circuit (no human review), and the reply-approval
interrupt (including a human edit)."""
from __future__ import annotations

import uuid

import pytest
from langgraph.types import Command

from graph.lead_build import build_lead_graph


def _invoke_with_thread():
    graph = build_lead_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return graph, config


def _base_lead(message: str) -> dict:
    return {
        "lead_id": str(uuid.uuid4()),
        "source": "dm",
        "contact": "prospect@example.com",
        "message": message,
    }


def test_qualified_inquiry_reaches_reply_approval():
    graph, config = _invoke_with_thread()
    result = graph.invoke(
        _base_lead("Hi, interested in Unit 2, looking to move in Sept 1, budget around $1800/mo."),
        config=config,
    )
    assert result["category"] == "QUALIFIED"
    assert "__interrupt__" in result
    pending = result["__interrupt__"][0]
    assert pending.value["type"] == "lead_reply_approval"
    assert pending.value["suggested_reply"]


def test_vague_inquiry_is_follow_up():
    graph, config = _invoke_with_thread()
    result = graph.invoke(_base_lead("Hey is this still available?"), config=config)
    assert result["category"] == "FOLLOW_UP"


def test_maintenance_message_is_support():
    graph, config = _invoke_with_thread()
    result = graph.invoke(
        _base_lead("Hi, the kitchen sink is leaking again, can someone fix it?"),
        config=config,
    )
    assert result["category"] == "SUPPORT"


def test_spam_short_circuits_without_reply_approval():
    graph, config = _invoke_with_thread()
    result = graph.invoke(
        _base_lead("Boost your SEO rankings today! Click here for 50% off backlinks."),
        config=config,
    )
    assert result["category"] == "SPAM"
    assert "__interrupt__" not in result
    assert "suggested_reply" not in result


def test_human_can_edit_the_reply_before_approval():
    graph, config = _invoke_with_thread()
    graph.invoke(
        _base_lead("Interested in applying, my income is $90k, want to move in October."),
        config=config,
    )
    final = graph.invoke(
        Command(resume={"approved": True, "edited_reply": "Thanks! Let's set up a showing this week."}),
        config=config,
    )
    assert final["reply_approved"] is True
    assert final["final_reply"] == "Thanks! Let's set up a showing this week."


def test_human_can_discard_the_reply():
    graph, config = _invoke_with_thread()
    graph.invoke(_base_lead("Is the unit still available?"), config=config)
    final = graph.invoke(Command(resume={"approved": False}), config=config)
    assert final["reply_approved"] is False


def test_lead_intake_rejects_incomplete_lead():
    graph, config = _invoke_with_thread()
    incomplete = _base_lead("hello")
    del incomplete["contact"]
    with pytest.raises(ValueError, match="contact"):
        graph.invoke(incomplete, config=config)
