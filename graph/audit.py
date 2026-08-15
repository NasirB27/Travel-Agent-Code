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
    applicant_id: str,
    event: str,
    actor: str,
    details: dict | None = None,
    path: Path = DEFAULT_AUDIT_LOG,
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "applicant_id": applicant_id,
        "event": event,
        "actor": actor,
        "details": details or {},
    }
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
