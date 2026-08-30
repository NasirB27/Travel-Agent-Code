"""Deterministic screening score against fixed, written-down criteria.

Per docs/architecture.md, the LLM is used for document extraction only —
the pass/fail decision itself is a plain rule check against criteria
applied identically to every applicant, which is what makes the process
defensible under fair housing rules.
"""
from __future__ import annotations

from graph.state import ApplicantState, ScreeningResult

MIN_INCOME_MULTIPLIER = 3.0
MIN_CREDIT_SCORE = 620
MAX_EVICTIONS_LAST_7_YEARS = 0


def screening_score_node(state: ApplicantState) -> ApplicantState:
    financials = state.get("financials", {})
    rent = state["monthly_rent"]
    reasons: list[str] = []

    income = financials.get("monthly_gross_income")
    if income is None:
        reasons.append("No income data extracted from submitted documents")
    elif income < MIN_INCOME_MULTIPLIER * rent:
        required = MIN_INCOME_MULTIPLIER * rent
        reasons.append(
            f"Income ${income:,.0f}/mo is below {MIN_INCOME_MULTIPLIER}x rent "
            f"(${required:,.0f}/mo required)"
        )

    credit_score = financials.get("credit_score")
    if credit_score is None:
        reasons.append("No credit score extracted from submitted documents")
    elif credit_score < MIN_CREDIT_SCORE:
        reasons.append(
            f"Credit score {credit_score} is below minimum {MIN_CREDIT_SCORE}"
        )

    evictions = financials.get("evictions_last_7_years", 0)
    if evictions > MAX_EVICTIONS_LAST_7_YEARS:
        reasons.append(
            f"{evictions} eviction(s) in the last 7 years exceeds policy of "
            f"{MAX_EVICTIONS_LAST_7_YEARS}"
        )

    screening: ScreeningResult = {"passed": len(reasons) == 0, "reasons": reasons}
    return {**state, "screening": screening}
