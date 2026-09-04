"""Small structured progress events for long MinerU NPU runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any


def log_phase(stage: str, event: str, **fields: Any) -> None:
    """Write one flushed JSON line that survives tee into run.log."""
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": str(stage),
        "event": str(event),
        **fields,
    }
    print(
        "MINERU_PHASE " + json.dumps(payload, sort_keys=True, default=str),
        flush=True,
    )
