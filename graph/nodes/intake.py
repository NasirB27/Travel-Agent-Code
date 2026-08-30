from graph.state import ApplicantState

REQUIRED_FIELDS = ["applicant_id", "name", "unit", "monthly_rent"]


def intake_node(state: ApplicantState) -> ApplicantState:
    missing = [field for field in REQUIRED_FIELDS if not state.get(field)]
    if missing:
        raise ValueError(f"Missing required applicant fields: {missing}")
    return state
