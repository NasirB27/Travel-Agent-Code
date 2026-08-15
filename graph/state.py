"""Shared state schema for the tenant-screening graph."""
from __future__ import annotations

from typing import Literal, Optional, TypedDict


class ApplicantDocument(TypedDict):
    doc_type: Literal["paystub", "w2", "credit_report", "background_check"]
    # Raw text content for this demo. A production pipeline would OCR/parse
    # an uploaded PDF or image before it reaches this stage.
    content: str


class Financials(TypedDict, total=False):
    monthly_gross_income: float
    employer: str
    credit_score: int
    evictions_last_7_years: int


class ScreeningResult(TypedDict, total=False):
    passed: bool
    reasons: list[str]


class ApplicantState(TypedDict, total=False):
    applicant_id: str
    name: str
    contact: str
    unit: str
    monthly_rent: float
    documents: list[ApplicantDocument]
    financials: Financials
    screening: ScreeningResult
    decision: Optional[Literal["approved", "rejected"]]
    decision_notes: str
    lease_draft: str
    lease_sent: bool
