"""Impatient, step-level reference scheduling for independently arriving tables.

The policy and accounting are CPU-only. Actual generation belongs to the
backend: no saved output, latency, or acceptance trace drives this scheduler.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


def accept_native_proposal(proposal: Iterable[int], targets: Iterable[int]) -> tuple[list[int], int]:
    """Commit only the matching prefix plus the authoritative next token."""
    proposed = [int(value) for value in proposal]
    verified = [int(value) for value in targets]
    if len(verified) < len(proposed) + 1:
        raise ValueError("verifier must return proposal length plus one predictions")
    accepted = 0
    for draft, target in zip(proposed, verified):
        if draft != target:
            break
        accepted += 1
    return proposed[:accepted] + [verified[accepted]], accepted


@dataclass(frozen=True)
class PhaseWork:
    request_id: str
    phase: str
    query_length: int = 1

    @property
    def key(self) -> tuple[str, int]:
        return self.phase, self.query_length


class TablePhasePolicy:
    """Prefill immediately; otherwise serve the least-recently served request.

    Matching ready work joins that call without waiting. Different adaptive
    verifier shapes remain independent, rather than changing anyone's K.
    """

    def __init__(self) -> None:
        self.last_served: dict[str, int] = {}
        self.turn = 0

    def choose(self, work: Iterable[PhaseWork]) -> list[PhaseWork]:
        ready = list(work)
        if not ready:
            return []
        if len({item.request_id for item in ready}) != len(ready):
            raise ValueError("a request must expose exactly one next phase")
        prefills = [item for item in ready if item.phase.endswith("prefill")]
        first = min(
            prefills or ready,
            key=lambda item: self.last_served.get(item.request_id, -1),
        )
        selected = (
            [first]
            if prefills
            else [item for item in ready if item.key == first.key]
        )
        self.turn += 1
        for item in selected:
            self.last_served[item.request_id] = self.turn
        return selected

    def retire(self, request_id: str) -> None:
        self.last_served.pop(request_id, None)


class PhaseLedger:
    """Disjoint host action intervals, attributed to each live request.

    Shared work is charged once globally and once to each participant's latency.
    Per-request totals must therefore NEVER be summed into device utilization.
    Device time is a separate overlapping diagnostic, not additional latency.
    """

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.global_wall: Counter[str] = Counter()
        self.global_device: Counter[str] = Counter()
        self.calls: Counter[str] = Counter()
        self.decode_combinations: Counter[str] = Counter()
        self.same_phase_unbatched: Counter[str] = Counter()

    def admit(self, request_id: str) -> None:
        if request_id in self.rows:
            raise ValueError(f"duplicate live request {request_id}")
        self.rows[request_id] = {
            "own_action_wall_s": Counter(),
            "other_action_wait_s": Counter(),
            "decode_phase_combination_wall_s": Counter(),
            "own_calls": Counter(),
        }

    def record(
        self,
        action: str,
        *,
        owners: Iterable[str],
        phases: dict[str, str],
        wall_s: float,
        device_s: float = 0.0,
        decode: bool = False,
    ) -> None:
        if wall_s < 0 or device_s < 0:
            raise ValueError("negative phase duration")
        selected = set(owners)
        if not selected <= phases.keys() or not phases.keys() <= self.rows.keys():
            raise ValueError("phase accounting references an unknown request")
        self.global_wall[action] += wall_s
        self.global_device[action] += device_s
        self.calls[action] += 1
        combination = "+".join(sorted(phases.values()))
        if decode:
            self.decode_combinations[combination] += wall_s
            if len(phases) > 1 and len(set(phases.values())) == 1 and len(selected) == 1:
                self.same_phase_unbatched[combination] += wall_s
        for request_id in phases:
            row = self.rows[request_id]
            bucket = "own_action_wall_s" if request_id in selected else "other_action_wait_s"
            row[bucket][action] += wall_s
            if request_id in selected:
                row["own_calls"][action] += 1
            if decode:
                row["decode_phase_combination_wall_s"][combination] += wall_s

    def retire(self, request_id: str) -> dict[str, Any]:
        return {key: dict(value) for key, value in self.rows.pop(request_id).items()}

    def summary(self) -> dict[str, Any]:
        return {
            "action_host_wall_s": dict(self.global_wall),
            "action_device_s": dict(self.global_device),
            "action_calls": dict(self.calls),
            "decode_phase_combination_wall_s": dict(self.decode_combinations),
            "same_phase_unbatched_wall_s": dict(self.same_phase_unbatched),
            "timing_contract": "disjoint host actions; device times overlap host actions",
        }
