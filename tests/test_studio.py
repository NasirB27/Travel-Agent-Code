"""Checks that langgraph.json points at real graphs, and that those graphs
are compiled without a checkpointer (the LangGraph server supplies its own
persistence for Studio)."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

from langgraph.graph.state import CompiledStateGraph

ROOT = Path(__file__).resolve().parent.parent


def test_langgraph_json_graphs_resolve_without_checkpointer():
    config = json.loads((ROOT / "langgraph.json").read_text())
    assert set(config["graphs"]) == {"applicant_screening", "lead_triage", "vacancy_ad"}

    for spec in config["graphs"].values():
        path, attr = spec.split(":")
        module_name = Path(path).with_suffix("").as_posix().lstrip("./").replace("/", ".")
        graph = getattr(importlib.import_module(module_name), attr)
        assert isinstance(graph, CompiledStateGraph)
        assert graph.checkpointer is None
