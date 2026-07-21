"""Low-perturbation host/device execution timeline recording.

The recorder uses the process-wide monotonic clock for host work. Device spans
are added only after the pipeline's existing event-resolution boundaries; this
module never synchronizes an accelerator.

Every event carries a ``track`` that states what kind of resource it describes,
so the viewer can lay events out honestly instead of stacking unrelated
timelines in one row:

- ``host``: work or blocking on a host thread. The viewer groups these by
  ``thread`` and nests them by containment into per-thread flame charts.
  ``event_type="scope"`` marks frames that exist to contain nested work (for
  example a generator ``next()`` that drives a whole pipeline stage) and are
  not themselves a measurement of exclusive work or idle waiting.
- ``device``: accelerator execution reconstructed from device events. ``lane``
  names the device lane ("prefill", "decode").
- ``queue``: a request sitting in a queue between stages. ``lane`` names the
  queue. These spans overlap heavily by design; the viewer aggregates them
  into queue-depth charts rather than painting them on top of each other.
- ``slot``: occupancy of a fixed decode slot; ``lane`` is the slot index.
"""

from __future__ import annotations

import json
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


ROW_ORDER = (
    "Pipeline",
    "Page input",
    "Layout detection",
    "Layout postprocess",
    "Crop / page preparation",
    "CPU preprocessing",
    "CPU MRoPE",
    "CPU / queue wait",
    "H2D / D2H transfer",
    "Vision prefill",
    "Text prefill",
    "Decode ready wait",
    "Decode admission",
    "Text decode",
    "Decode request residency",
    "Decode control / wait",
    "Result assembly",
    "Artifacts / tracing",
)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return str(value)


class TimelineRecorder:
    """Thread-safe in-memory trace with one shared host monotonic clock."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self.reset()

    def reset(self, metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.origin_ns = time.perf_counter_ns()
            self.metadata = _json_value(metadata or {})
            self._events: list[dict[str, Any]] = []
            self._next_id = 1

    @staticmethod
    def now_ns() -> int:
        return time.perf_counter_ns()

    def _relative_ns(self, absolute_ns: int) -> int:
        return max(0, int(absolute_ns) - int(self.origin_ns))

    def record_span(
        self,
        row: str,
        name: str,
        start_ns: int,
        end_ns: int,
        *,
        flow_id: str | None = None,
        flow_ids: list[str] | tuple[str, ...] | None = None,
        event_type: str = "work",
        clock: str = "host_monotonic",
        track: str = "host",
        lane: str | int | None = None,
        args: dict[str, Any] | None = None,
        thread_name: str | None = None,
    ) -> int | None:
        if not self.enabled:
            return None
        relative_start = self._relative_ns(start_ns)
        relative_end = max(relative_start, self._relative_ns(end_ns))
        related_flows = [str(item) for item in (flow_ids or ())]
        if flow_id is not None:
            match = re.search(r"page_(\d+)", str(flow_id))
            if match is not None:
                page_flow = f"page:{int(match.group(1))}"
                if page_flow not in related_flows:
                    related_flows.append(page_flow)
        with self._lock:
            event_id = self._next_id
            self._next_id += 1
            self._events.append(
                {
                    "id": event_id,
                    "kind": "span",
                    "row": str(row),
                    "name": str(name),
                    "start_ns": relative_start,
                    "end_ns": relative_end,
                    "duration_ns": relative_end - relative_start,
                    "flow_id": None if flow_id is None else str(flow_id),
                    "flow_ids": related_flows,
                    "event_type": str(event_type),
                    "clock": str(clock),
                    "track": str(track),
                    "lane": None if lane is None else lane,
                    "thread": thread_name or threading.current_thread().name,
                    "args": _json_value(args or {}),
                }
            )
        return event_id

    def record_span_seconds(
        self,
        row: str,
        name: str,
        start_seconds: float,
        end_seconds: float,
        **kwargs: Any,
    ) -> int | None:
        return self.record_span(
            row,
            name,
            int(float(start_seconds) * 1_000_000_000),
            int(float(end_seconds) * 1_000_000_000),
            **kwargs,
        )

    @contextmanager
    def span(
        self,
        row: str,
        name: str,
        *,
        flow_id: str | None = None,
        event_type: str = "work",
        track: str = "host",
        args: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            self.record_span(
                row,
                name,
                started,
                time.perf_counter_ns(),
                flow_id=flow_id,
                event_type=event_type,
                track=track,
                args=args,
            )

    def instant(
        self,
        row: str,
        name: str,
        *,
        timestamp_ns: int | None = None,
        flow_id: str | None = None,
        event_type: str = "instant",
        track: str = "host",
        lane: str | int | None = None,
        args: dict[str, Any] | None = None,
    ) -> int | None:
        if not self.enabled:
            return None
        stamp = time.perf_counter_ns() if timestamp_ns is None else int(timestamp_ns)
        return self.record_span(
            row,
            name,
            stamp,
            stamp,
            flow_id=flow_id,
            event_type=event_type,
            track=track,
            lane=lane,
            args=args,
        )

    def counter(
        self,
        row: str,
        name: str,
        value: int | float,
        *,
        timestamp_ns: int | None = None,
        track: str = "queue",
        lane: str | int | None = None,
        args: dict[str, Any] | None = None,
    ) -> int | None:
        payload = dict(args or {})
        payload["value"] = value
        return self.instant(
            row,
            name,
            timestamp_ns=timestamp_ns,
            event_type="counter",
            track=track,
            lane=lane,
            args=payload,
        )

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events]

    def snapshot(self) -> dict[str, Any]:
        events = self.events()
        end_ns = max((int(event["end_ns"]) for event in events), default=0)
        rows = [row for row in ROW_ORDER if any(event["row"] == row for event in events)]
        rows.extend(
            sorted(
                {str(event["row"]) for event in events}.difference(rows)
            )
        )
        return {
            "schema_version": 2,
            "clock": "nanoseconds relative to one process-wide perf_counter_ns origin",
            "device_clock_note": (
                "device-event spans use event-relative offsets anchored at host enqueue; "
                "they are resolved only at synchronization points already present in the pipeline"
            ),
            "metadata": self.metadata,
            "duration_ns": end_ns,
            "rows": rows,
            "events": sorted(events, key=lambda event: (event["start_ns"], event["id"])),
        }

    def write_json(self, path: Path) -> None:
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def write_html(self, path: Path) -> None:
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.snapshot(),
            ensure_ascii=False,
            separators=(",", ":"),
        ).replace("</", "<\\/")
        path.write_text(render_trace_html(payload), encoding="utf-8")


VIEWER_TEMPLATE_PATH = Path(__file__).with_name("timeline_viewer.html")
_TRACE_PLACEHOLDER = "/*__TRACE_JSON__*/null"


def render_trace_html(payload_json: str) -> str:
    """Embed a trace snapshot into the self-contained viewer template."""

    template = VIEWER_TEMPLATE_PATH.read_text(encoding="utf-8")
    if _TRACE_PLACEHOLDER not in template:
        raise RuntimeError(
            f"viewer template {VIEWER_TEMPLATE_PATH} lost its trace placeholder"
        )
    return template.replace(_TRACE_PLACEHOLDER, payload_json, 1)
