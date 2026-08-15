"""Exercises the tenant-screening graph end to end, including both
interrupt points, without any real network calls or terminal input."""
from __future__ import annotations

import uuid

import pytest
from langgraph.types import Command

from graph.build import build_graph

QUALIFIED_DOCUMENTS = [
    {"doc_type": "paystub", "content": "monthly_gross_income: 6200\nemployer: Acme Consulting"},
    {"doc_type": "credit_report", "content": "credit_score: 705\nevictions_last_7_years: 0"},
]

UNQUALIFIED_DOCUMENTS = [
    {"doc_type": "paystub", "content": "monthly_gross_income: 2000\nemployer: Acme Consulting"},
    {"doc_type": "credit_report", "content": "credit_score: 540\nevictions_last_7_years: 1"},
]


def _invoke_with_thread():
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return graph, config


def _base_applicant(documents: list[dict]) -> dict:
    return {
        "applicant_id": str(uuid.uuid4()),
        "name": "Test Applicant",
        "contact": "test@example.com",
        "unit": "Unit 1",
        "monthly_rent": 1800.0,
        "documents": documents,
    }


def test_screening_score_flags_a_qualified_applicant():
    graph, config = _invoke_with_thread()
    result = graph.invoke(_base_applicant(QUALIFIED_DOCUMENTS), config=config)

    assert "__interrupt__" in result
    pending = result["__interrupt__"][0]
    assert pending.value["type"] == "screening_decision"
    assert pending.value["recommendation"] == "approve"
    assert pending.value["reasons"] == []


def test_screening_score_flags_an_unqualified_applicant():
    graph, config = _invoke_with_thread()
    result = graph.invoke(_base_applicant(UNQUALIFIED_DOCUMENTS), config=config)

    pending = result["__interrupt__"][0]
    assert pending.value["recommendation"] == "reject"
    assert len(pending.value["reasons"]) == 3  # income, credit score, evictions


def test_human_rejection_short_circuits_before_lease_draft():
    graph, config = _invoke_with_thread()
    graph.invoke(_base_applicant(QUALIFIED_DOCUMENTS), config=config)

    result = graph.invoke(
        Command(resume={"decision": "rejected", "notes": "landlord override"}),
        config=config,
    )

    assert "__interrupt__" not in result
    assert result["decision"] == "rejected"
    assert "lease_draft" not in result


def test_human_approval_reaches_lease_send_interrupt_and_can_be_approved():
    graph, config = _invoke_with_thread()
    graph.invoke(_base_applicant(QUALIFIED_DOCUMENTS), config=config)

    mid = graph.invoke(
        Command(resume={"decision": "approved", "notes": "looks good"}),
        config=config,
    )
    assert "__interrupt__" in mid
    lease_prompt = mid["__interrupt__"][0]
    assert lease_prompt.value["type"] == "lease_send_approval"
    assert "lease_draft" in mid

    final = graph.invoke(Command(resume={"approved": True}), config=config)
    assert "__interrupt__" not in final
    assert final["lease_sent"] is True


def test_intake_rejects_incomplete_applicant():
    graph, config = _invoke_with_thread()
    incomplete = _base_applicant(QUALIFIED_DOCUMENTS)
    del incomplete["monthly_rent"]

    with pytest.raises(ValueError, match="monthly_rent"):
        graph.invoke(incomplete, config=config)
