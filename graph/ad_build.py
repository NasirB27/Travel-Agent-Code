"""Wires the vacancy-ad automation pipeline together as a LangGraph
StateGraph.

Graph shape (see docs/architecture.md, roadmap step 4):

    campaign_intake -> content_creator -> ad_approval* -> publish

    * = human-in-the-loop interrupt; nothing posts to the Facebook Page
        without an explicit human decision.

This graph is meant to be invoked when a unit goes vacant -- the "vacancy
watcher" trigger itself (a cron job, a status flag flip, etc.) is
deployment plumbing outside this graph's scope, same as how the lead-triage
graph doesn't own how an inbound DM/email actually reaches it.
"""
from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.ad_state import AdState
from graph.nodes.ad_approval import ad_approval_node
from graph.nodes.campaign_intake import campaign_intake_node
from graph.nodes.content_creator import content_creator_node
from graph.nodes.scheduler import publish_node


def build_ad_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    graph = StateGraph(AdState)

    graph.add_node("campaign_intake", campaign_intake_node)
    graph.add_node("content_creator", content_creator_node)
    graph.add_node("ad_approval", ad_approval_node)
    graph.add_node("publish", publish_node)

    graph.add_edge(START, "campaign_intake")
    graph.add_edge("campaign_intake", "content_creator")
    graph.add_edge("content_creator", "ad_approval")
    graph.add_edge("ad_approval", "publish")
    graph.add_edge("publish", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
