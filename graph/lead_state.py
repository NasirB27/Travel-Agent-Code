"""State schema for the lead qualification/triage graph.

Kept separate from ApplicantState (graph/state.py): a lead is an inbound
DM/email that hasn't necessarily applied for anything yet, whereas an
applicant has submitted documents against a specific unit. A QUALIFIED
lead is the hand-off point into the applicant pipeline, done manually for
now (see docs/architecture.md).
"""
from __future__ import annotations

from typing import Literal, Optional, TypedDict

LeadCategory = Literal["QUALIFIED", "FOLLOW_UP", "SUPPORT", "SPAM"]


class LeadState(TypedDict, total=False):
    lead_id: str
    source: Literal["dm", "email"]
    contact: str
    message: str
    category: LeadCategory
    reasoning: str
    suggested_reply: str
    reply_approved: bool
    final_reply: Optional[str]
