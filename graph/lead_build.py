"""Wires the lead-triage pipeline together as a LangGraph StateGraph.

Graph shape (see docs/architecture.md, roadmap step 3):

    lead_intake -> classify_lead --(SPAM)--> spam_log -> END
                                --(else)--> draft_reply -> reply_approval* -> END

    * = human-in-the-loop interrupt.
"""
from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.lead_state import LeadState
from graph.nodes.lead_intake import lead_intake_node
from graph.nodes.lead_reply import draft_reply_node, reply_approval_node
from graph.nodes.lead_routing import spam_log_node
from graph.nodes.lead_triage import classify_lead_node


def _route_after_classification(state: LeadState) -> str:
    return "spam_log" if state["category"] == "SPAM" else "draft_reply"


def build_lead_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    graph = StateGraph(LeadState)

    graph.add_node("lead_intake", lead_intake_node)
    graph.add_node("classify_lead", classify_lead_node)
    graph.add_node("draft_reply", draft_reply_node)
    graph.add_node("reply_approval", reply_approval_node)
    graph.add_node("spam_log", spam_log_node)

    graph.add_edge(START, "lead_intake")
    graph.add_edge("lead_intake", "classify_lead")
    graph.add_conditional_edges(
        "classify_lead",
        _route_after_classification,
        {"spam_log": "spam_log", "draft_reply": "draft_reply"},
    )
    graph.add_edge("draft_reply", "reply_approval")
    graph.add_edge("reply_approval", END)
    graph.add_edge("spam_log", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
