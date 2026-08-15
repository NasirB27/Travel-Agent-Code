"""Wires the tenant-screening pipeline together as a LangGraph StateGraph.

Graph shape (see docs/architecture.md for the full picture):

    intake -> application_parser -> screening_score -> screening_decision*
        screening_decision -> (approved) -> lease_draft -> lease_send_approval*
        screening_decision -> (rejected) -> rejection_log

    * = human-in-the-loop interrupt; nothing downstream runs without a
        human response.
"""
from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.nodes.application_parser import application_parser_node
from graph.nodes.approval import lease_send_approval_node, screening_decision_node
from graph.nodes.intake import intake_node
from graph.nodes.lease import lease_draft_node, rejection_log_node
from graph.nodes.screening import screening_score_node
from graph.state import ApplicantState


def _route_after_screening_decision(state: ApplicantState) -> str:
    return "lease_draft" if state["decision"] == "approved" else "rejection_log"


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    graph = StateGraph(ApplicantState)

    graph.add_node("intake", intake_node)
    graph.add_node("application_parser", application_parser_node)
    graph.add_node("screening_score", screening_score_node)
    graph.add_node("screening_decision", screening_decision_node)
    graph.add_node("lease_draft", lease_draft_node)
    graph.add_node("lease_send_approval", lease_send_approval_node)
    graph.add_node("rejection_log", rejection_log_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "application_parser")
    graph.add_edge("application_parser", "screening_score")
    graph.add_edge("screening_score", "screening_decision")
    graph.add_conditional_edges(
        "screening_decision",
        _route_after_screening_decision,
        {"lease_draft": "lease_draft", "rejection_log": "rejection_log"},
    )
    graph.add_edge("lease_draft", "lease_send_approval")
    graph.add_edge("lease_send_approval", END)
    graph.add_edge("rejection_log", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
