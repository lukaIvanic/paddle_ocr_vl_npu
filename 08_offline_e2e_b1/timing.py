"""Wall-clock and device-event timing helpers."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        import torch_npu

        torch_npu.npu.synchronize(device)


def timed_wall(device: torch.device | None, fn: Callable[[], Any]) -> tuple[Any, float]:
    if device is not None:
        synchronize(device)
    started = time.perf_counter()
    result = fn()
    if device is not None:
        synchronize(device)
    return result, time.perf_counter() - started


class DeviceTimeline:
    """Record per-stage device time without synchronizing between stages.

    The caller resolves the timeline at a natural phase boundary. The reported
    values are accelerator execution time, while coarse phase and request
    timings remain ordinary synchronized wall time.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._events: dict[str, tuple[Any, Any]] = {}

    def _event(self):
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
            began = time.perf_counter()
            result = fn()
            elapsed = time.perf_counter() - began
            self._events[name] = (elapsed, None)
            return result
        start.record()
        result = fn()
        end.record()
        self._events[name] = (start, end)
        return result

    def resolve(self) -> dict[str, float]:
        synchronize(self.device)
        resolved: dict[str, float] = {}
        for name, (start, end) in self._events.items():
            if end is None:
                resolved[name] = float(start)
            else:
                resolved[name] = float(start.elapsed_time(end)) / 1000.0
        return resolved
