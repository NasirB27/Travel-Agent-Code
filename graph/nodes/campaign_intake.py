from graph.ad_state import AdState

REQUIRED_FIELDS = ["campaign_id", "unit", "monthly_rent", "platforms"]


def campaign_intake_node(state: AdState) -> AdState:
    missing = [field for field in REQUIRED_FIELDS if not state.get(field)]
    if missing:
        raise ValueError(f"Missing required campaign fields: {missing}")
    if not state["platforms"]:
        raise ValueError("At least one platform must be specified")
    return state
