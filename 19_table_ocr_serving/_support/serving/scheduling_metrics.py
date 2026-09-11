"""Host-side scheduling attribution without device synchronization.

Ready-source spans describe time during which no new decode is submitted.
An already submitted decode may overlap a span. These are not exclusive NPU
stall measurements and must not be added to device or residency timings.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
import time
from typing import Any, Iterable


@dataclass
class _Request:
    started_at: float
    ready_at: float | None = None
    first_decode_at: float | None = None
    launched: Counter[int] = field(default_factory=Counter)
    consumed: Counter[int] = field(default_factory=Counter)
    cpu_eligible_at: float | None = None
    cpu_idle_eligible_at: float | None = None
    cpu_readiness: dict[str, Any] = field(default_factory=dict)


class RequestSchedulingMetrics:
    """One recorder per serving run; called only on the scheduler thread."""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.requests: dict[str, _Request] = {}
        # One span per prepared request, not per iteration. Late arrivals can
        # reach the worker after a prefill that delayed them, requiring history.
        self.prefills: list[tuple[str, float, float, str]] = []
        self.prefill_ends: list[float] = []

    def register(self, request_id: str, started_at: float) -> None:
        if request_id in self.requests:
            raise ValueError(f"duplicate scheduling metrics request: {request_id}")
        self.requests[request_id] = _Request(started_at=started_at)

    def record_prefill(
        self, request_id: str, started_at: float, finished_at: float,
        *, status: str = "ok",
    ) -> None:
        # Exclude idle blocking before this request arrived.
        started_at = max(started_at, self.requests[request_id].started_at)
        if finished_at < started_at:
            raise ValueError("prefill span ends before it starts")
        if self.prefill_ends and started_at < self.prefill_ends[-1]:
            raise ValueError("ready-source spans must not overlap")
        self.prefills.append((request_id, started_at, finished_at, status))
        self.prefill_ends.append(finished_at)
        if status == "ok":
            self.requests[request_id].ready_at = finished_at
        else:
            self.requests.pop(request_id)

    def cpu_prefill_eligible(self, request_id: str, *, block: bool) -> None:
        """FIFO head has an unreserved slot; CPU may still be unfinished.

        Once eligible, this head cannot lose its slot to a later request:
        prefill is FIFO on the scheduler thread. Read the clock only at the
        first eligibility observation or transition into an idle blocking pull.
        """
        request = self.requests[request_id]
        if request.cpu_eligible_at is None:
            request.cpu_eligible_at = time.perf_counter()
        if block and request.cpu_idle_eligible_at is None:
            request.cpu_idle_eligible_at = time.perf_counter()

    def cpu_prepared(
        self, request_id: str, *, submitted_at: float, queue_wait_s: float,
        finished_at: float, consumed_at: float,
    ) -> None:
        request = self.requests[request_id]
        eligible = request.cpu_eligible_at
        assert eligible is not None
        started = submitted_at + queue_wait_s
        blocked = max(0.0, finished_at - eligible)
        queue = max(0.0, min(started, finished_at) - eligible)
        service = max(0.0, finished_at - max(started, eligible))
        idle = (0.0 if request.cpu_idle_eligible_at is None else
                max(0.0, finished_at - request.cpu_idle_eligible_at))
        request.cpu_readiness = {
            "prefill_blocked_s": blocked,
            "blocked_cpu_queue_s": queue,
            "blocked_cpu_service_s": service,
            "scheduler_idle_blocked_s": idle,
            "ready_to_consumer_poll_s": max(0.0, consumed_at - max(eligible, finished_at)),
            "eligible_offset_s": eligible - request.started_at,
            "cpu_finished_offset_s": finished_at - request.started_at,
            "semantics": (
                "Observed FIFO-head/free-slot eligibility until CPU preparation "
                "finishes; zero if already ready. Includes CPU queue and service, "
                "excludes full-slot and earlier-request prefill waits. Idle subset "
                "means no active/ready decode request at a blocking pull, not "
                "device-traced NPU idle. Poll-boundary measurement, not a "
                "counterfactual E2E saving; do not subtract from latency."
            ),
        }

    def step(self, request_ids: Iterable[str], started_at: float) -> None:
        ids = tuple(request_ids)
        for request_id in ids:
            request = self.requests[request_id]
            if request.first_decode_at is None:
                request.first_decode_at = started_at
            request.launched[len(ids)] += 1

    def consume(self, request_ids: Iterable[str]) -> None:
        """Count output-bearing slots after stale epoch/look-ahead filtering."""
        ids = tuple(request_ids)
        for request_id in ids:
            self.requests[request_id].consumed[len(ids)] += 1

    def finish(self, request_id: str, completed_at: float) -> dict[str, Any]:
        request = self.requests.pop(request_id)
        split = request.first_decode_at
        if split is None:
            split = completed_at
        spans: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        seconds: Counter[str] = Counter()
        first = bisect_right(self.prefill_ends, request.started_at)
        for owner, started, ended, status in self.prefills[first:]:
            if started >= completed_at:
                break
            if owner == request_id:
                continue
            for phase, left, right in (
                ("before_first_decode", request.started_at, split),
                ("during_decode", split, completed_at),
            ):
                begin, end = max(started, left), min(ended, right)
                if end <= begin:
                    continue
                counts[phase] += 1
                seconds[phase] += end - begin
                spans.append({
                    "other_request_id": owner,
                    "other_request_status": status,
                    "phase": phase,
                    "start_offset_s": begin - request.started_at,
                    "end_offset_s": end - request.started_at,
                    "host_pause_s": end - begin,
                })
        return {
            "format": "request_scheduling_metrics_v1",
            "clock": "host_monotonic",
            "scope": "request_submission_to_completion_detection",
            "pause_semantics": (
                "Other-request CPU preparation waits and prefill in the ready "
                "source. No new decode is submitted within these spans; an "
                "in-flight decode can overlap them. Not exclusive device "
                "stall time; do not add to device-stage or residency timings."
            ),
            "batch_size": self.batch_size,
            "cpu_readiness": dict(request.cpu_readiness),
            "first_decode_offset_s": (
                None if request.first_decode_at is None
                else request.first_decode_at - request.started_at
            ),
            "own_prefill_ready_offset_s": (
                None if request.ready_at is None
                else request.ready_at - request.started_at
            ),
            "own_prefill_ready_to_first_decode_s": (
                None if request.ready_at is None or request.first_decode_at is None
                else max(0.0, request.first_decode_at - request.ready_at)
            ),
            "before_first_decode_other_prefill_count": counts["before_first_decode"],
            "before_first_decode_other_prefill_host_s": seconds["before_first_decode"],
            "decode_other_prefill_count": counts["during_decode"],
            "decode_other_prefill_host_s": seconds["during_decode"],
            "other_prefill_spans": spans,
            "launched_decode_iterations_by_active_slots": dict(request.launched),
            "consumed_decode_iterations_by_useful_slots": dict(request.consumed),
            "occupancy_semantics": (
                "Launched counts include completion look-ahead. Consumed counts "
                "exclude stale slot epochs and count tokens before any final "
                "repetition trimming. Prefill's first token is not a decode iteration."
            ),
        }
