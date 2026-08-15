from graph.lead_state import LeadState

REQUIRED_FIELDS = ["lead_id", "source", "contact", "message"]


def lead_intake_node(state: LeadState) -> LeadState:
    missing = [field for field in REQUIRED_FIELDS if not state.get(field)]
    if missing:
        raise ValueError(f"Missing required lead fields: {missing}")
    return state
