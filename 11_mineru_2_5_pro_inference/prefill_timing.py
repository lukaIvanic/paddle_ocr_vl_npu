"""Low-overhead device-event timing for MinerU multimodal prefill."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


class PrefillDeviceTimeline:
    """Measure queued device work and synchronize once at prefill completion."""

    def __init__(self, device: torch.device, samples: list[dict[str, Any]] | None = None):
        self.device = device
        self.samples = samples
        self._events: list[tuple[str, Any, Any, dict[str, Any] | None]] = []

    def _event(self) -> Any | None:
        if self.device.type == "cuda":
            return torch.cuda.Event(enable_timing=True)
        if self.device.type == "npu":
            import torch_npu

            return torch_npu.npu.Event(enable_timing=True)
        return None

    def measure(self, name: str, fn: Callable[[], Any], *, tags: dict[str, Any] | None = None) -> Any:
        start = self._event()
        end = self._event()
        if start is None or end is None:
            return fn()
        start.record()
        result = fn()
        end.record()
        self._events.append((name, start, end, tags))
        return result

    def resolve(self) -> dict[str, float]:
        if not self._events:
            return {}
        self._events[-1][2].synchronize()
        totals: dict[str, float] = {}
        for name, start, end, tags in self._events:
            elapsed_s = float(start.elapsed_time(end)) / 1000.0
            totals[name] = totals.get(name, 0.0) + elapsed_s
            if tags is not None and self.samples is not None:
                self.samples.append({**tags, "stage": name, "device_s": elapsed_s})
        self._events.clear()
        return totals
