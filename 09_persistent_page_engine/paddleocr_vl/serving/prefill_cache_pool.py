"""Zero-once KV storage for prefetched requests waiting on decode admission."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from ..model.text_decode import LocalPaddleOCRVLStaticCache


def _cache_nbytes(cache: LocalPaddleOCRVLStaticCache) -> int:
    return sum(
        int(tensor.numel()) * int(tensor.element_size())
        for tensor in cache.flat_tensors()
    )


@dataclass
class _FreeSlot:
    slot_index: int
    ready_event: Any | None = None


class PrefillKVCacheLease:
    """Exclusive ownership of one B=1 row in the prefill KV arena."""

    def __init__(
        self,
        pool: PrefillKVCachePool,
        *,
        slot_index: int,
        generation: int,
        cache: LocalPaddleOCRVLStaticCache,
    ) -> None:
        self._pool = pool
        self.slot_index = int(slot_index)
        self.generation = int(generation)
        self.cache = cache
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if self._released:
            raise RuntimeError("prefill KV cache lease was released twice")
        self._released = True
        self._pool._release(self)


class PrefillKVCachePool:
    """Fixed-capacity, zero-once arena of private request KV caches."""

    def __init__(
        self,
        cache: LocalPaddleOCRVLStaticCache,
        *,
        device: torch.device,
    ) -> None:
        if not cache.key_caches or not cache.value_caches:
            raise ValueError("prefill KV cache arena must contain K and V tensors")
        capacity = int(cache.key_caches[0].shape[0])
        if capacity <= 0:
            raise ValueError("prefill KV cache arena capacity must be positive")
        if any(
            int(tensor.shape[0]) != capacity for tensor in cache.flat_tensors()
        ):
            raise ValueError("prefill KV cache arena tensors disagree on capacity")

        self.cache = cache
        self.device = torch.device(device)
        self.capacity = capacity
        self.nbytes = _cache_nbytes(cache)
        self._free = deque(_FreeSlot(index) for index in range(capacity))
        self._active: dict[int, PrefillKVCacheLease] = {}
        self._generations = [0] * capacity
        self.acquisitions = 0
        self.reuses = 0
        self.releases = 0
        self.high_water_active = 0

    def _cache_view(self, slot_index: int) -> LocalPaddleOCRVLStaticCache:
        start = int(slot_index)
        end = start + 1
        return LocalPaddleOCRVLStaticCache(
            key_caches=tuple(
                tensor[start:end] for tensor in self.cache.key_caches
            ),
            value_caches=tuple(
                tensor[start:end] for tensor in self.cache.value_caches
            ),
            cache_length=int(self.cache.cache_length),
        )

    def acquire(self) -> PrefillKVCacheLease:
        if not self._free:
            raise RuntimeError(
                "prefill KV cache arena exhausted: "
                f"capacity={self.capacity} active={len(self._active)}"
            )
        free = self._free.popleft()
        if free.ready_event is not None:
            import torch_npu

            torch_npu.npu.current_stream().wait_event(free.ready_event)
            self.reuses += 1
        self._generations[free.slot_index] += 1
        lease = PrefillKVCacheLease(
            self,
            slot_index=free.slot_index,
            generation=self._generations[free.slot_index],
            cache=self._cache_view(free.slot_index),
        )
        if free.slot_index in self._active:
            raise RuntimeError("prefill KV cache arena handed out an active slot")
        self._active[free.slot_index] = lease
        self.acquisitions += 1
        self.high_water_active = max(self.high_water_active, len(self._active))
        return lease

    def _release(self, lease: PrefillKVCacheLease) -> None:
        active = self._active.get(lease.slot_index)
        if active is not lease:
            raise RuntimeError("prefill KV cache arena received a stale lease")
        import torch_npu

        ready_event = torch_npu.npu.current_stream().record_event()
        del self._active[lease.slot_index]
        self._free.append(_FreeSlot(lease.slot_index, ready_event))
        self.releases += 1

    def stats(self) -> dict[str, int | str]:
        return {
            "storage": "zero_once_fixed_arena",
            "capacity": self.capacity,
            "allocated_bytes": self.nbytes,
            "acquisitions": self.acquisitions,
            "reuses": self.reuses,
            "releases": self.releases,
            "active_slots": len(self._active),
            "free_slots": len(self._free),
            "high_water_active_slots": self.high_water_active,
        }
