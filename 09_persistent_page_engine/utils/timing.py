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
        self._events: dict[str, tuple[Any, Any, int]] = {}
        self._anchor_event: Any | None = None
        self._anchor_host_ns: int | None = None

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
        enqueued_ns = time.perf_counter_ns()
        if start is None or end is None:
            began = time.perf_counter()
            result = fn()
            elapsed = time.perf_counter() - began
            self._events[name] = (elapsed, None, enqueued_ns)
            return result
        if self._anchor_event is None:
            self._anchor_event = start
            self._anchor_host_ns = enqueued_ns
        start.record()
        result = fn()
        end.record()
        self._events[name] = (start, end, enqueued_ns)
        return result

    def resolve(self) -> dict[str, float]:
        return {
            name: float(span["seconds"])
            for name, span in self.resolve_spans().items()
        }

    def resolve_spans(self) -> dict[str, dict[str, float | int | str]]:
        """Resolve duration and relative device position at the existing boundary.

        ``start_ns`` and ``end_ns`` share the host monotonic clock. Accelerator
        spans use event-to-event offsets from the first recorded event, anchored
        at the host timestamp immediately before that event was enqueued. No
        synchronization is introduced beyond the one resolve() already used.
        """

        synchronize(self.device)
        resolved: dict[str, dict[str, float | int | str]] = {}
        for name, (start, end, enqueued_ns) in self._events.items():
            if end is None:
                seconds = float(start)
                start_ns = int(enqueued_ns)
                clock = "host_monotonic"
            else:
                seconds = float(start.elapsed_time(end)) / 1000.0
                if self._anchor_event is None or self._anchor_host_ns is None:
                    raise RuntimeError("device timeline lost its anchor event")
                offset_seconds = (
                    float(self._anchor_event.elapsed_time(start)) / 1000.0
                )
                start_ns = self._anchor_host_ns + int(offset_seconds * 1_000_000_000)
                clock = "device_event_reconstructed"
            resolved[name] = {
                "seconds": seconds,
                "start_ns": start_ns,
                "end_ns": start_ns + int(seconds * 1_000_000_000),
                "clock": clock,
            }
        return resolved
