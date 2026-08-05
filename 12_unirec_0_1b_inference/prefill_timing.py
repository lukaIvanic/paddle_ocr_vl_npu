"""Low-overhead device-event timing for UniRec prefill stages."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


class PrefillDeviceTimeline:
    """Measure queued device work and synchronize only at prefill completion."""

    def __init__(self, device: torch.device):
        self.device = device
        self._events: list[tuple[str, Any, Any]] = []

    def _event(self) -> Any | None:
        if self.device.type == "cuda":
            return torch.cuda.Event(enable_timing=True)
        if self.device.type == "npu":
            import torch_npu

            return torch_npu.npu.Event(enable_timing=True)
        return None

    def measure(self, name: str, fn: Callable[[], Any]) -> Any:
        start = self._event()
        end = self._event()
        if start is None or end is None:
            return fn()
        start.record()
        result = fn()
        end.record()
        self._events.append((name, start, end))
        return result

    def resolve(self) -> dict[str, float]:
        if not self._events:
            return {}
        self._events[-1][2].synchronize()
        totals: dict[str, float] = {}
        for name, start, end in self._events:
            totals[name] = totals.get(name, 0.0) + float(start.elapsed_time(end)) / 1000.0
        return totals
