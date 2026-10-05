"""Entry points for LangGraph Studio / `langgraph dev` (see langgraph.json).

These compile the same workflows as build.py, lead_build.py, and
ad_build.py, but without a checkpointer: the LangGraph server supplies its
own persistence, and threads, interrupts, and resumes are driven from
Studio instead of the CLI prompts in main.py / triage_lead.py / post_ad.py.
"""
from __future__ import annotations

from graph.ad_build import ad_workflow
from graph.build import applicant_workflow
from graph.lead_build import lead_workflow

applicant_screening = applicant_workflow().compile()
lead_triage = lead_workflow().compile()
vacancy_ad = ad_workflow().compile()
