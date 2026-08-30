"""Append-only audit log for every automated recommendation and human decision.

Every screening pipeline needs a defensible record of what was decided and
by whom — this is what the architecture plan (docs/architecture.md) calls
out as a Fair Housing / FCRA requirement, not just good practice.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_AUDIT_LOG = Path("audit_log.jsonl")


def log_event(
    entity_id: str,
    event: str,
    actor: str,
    details: dict | None = None,
    path: Path = DEFAULT_AUDIT_LOG,
) -> None:
    """Append one audit entry. ``entity_id`` is whatever the event is about
    (an applicant_id from the screening graph, a lead_id from the triage
    graph, etc.) -- kept generic since this log is shared across pipelines.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entity_id": entity_id,
        "event": event,
        "actor": actor,
        "details": details or {},
    }
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
