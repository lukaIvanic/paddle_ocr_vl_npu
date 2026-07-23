"""Reference-counted leases over packed text-prefill KV buffers.

The packed text graph writes several causally isolated request prefixes into
one compact B=1 cache.  A lease keeps that cache alive while its members wait
for decode admission; each member exposes only its own sequence slice.  Once
all member handles are released, the buffer returns to the pool.

Recycling is intentionally stream-ordered: callers must release a member only
after its final cache read has been enqueued on the same device stream as the
next packed graph.  The Experiment 09 prefill and decode-admission copies use
that single-stream ordering today.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

from .text_decode import LocalPaddleOCRVLStaticCache


def static_cache_nbytes(cache: LocalPaddleOCRVLStaticCache) -> int:
    return sum(
        int(tensor.numel()) * int(tensor.element_size())
        for tensor in cache.flat_tensors()
    )


@dataclass
class _PackedKVBuffer:
    buffer_id: int
    bucket: int
    cache: LocalPaddleOCRVLStaticCache
    nbytes: int
    generation: int = 0


class PackedKVCacheMember:
    """One request's prefix inside a pooled packed KV cache."""

    def __init__(
        self,
        lease: PackedKVCacheLease,
        *,
        member_index: int,
        offset: int,
        length: int,
    ) -> None:
        self._lease = lease
        self.member_index = int(member_index)
        self.offset = int(offset)
        self.length = int(length)
        self._released = False

    @property
    def buffer_id(self) -> int:
        return self._lease.buffer_id

    @property
    def generation(self) -> int:
        return self._lease.generation

    @property
    def released(self) -> bool:
        return self._released

    def cache_view(self) -> LocalPaddleOCRVLStaticCache:
        if self._released:
            raise RuntimeError("packed KV member has already been released")
        start = self.offset
        end = start + self.length
        cache = self._lease.cache
        return LocalPaddleOCRVLStaticCache(
            key_caches=tuple(
                tensor[:, :, start:end, :] for tensor in cache.key_caches
            ),
            value_caches=tuple(
                tensor[:, :, start:end, :] for tensor in cache.value_caches
            ),
            cache_length=self.length,
        )

    def release(self) -> None:
        if self._released:
            raise RuntimeError("packed KV member was released twice")
        self._released = True
        self._lease._release_member(self.member_index)


class PackedKVCacheLease:
    """One acquired packed cache and all member references into it."""

    def __init__(
        self,
        pool: PackedKVCachePool,
        buffer: _PackedKVBuffer,
        offsets: tuple[int, ...],
        lengths: tuple[int, ...],
    ) -> None:
        self._pool = pool
        self._buffer = buffer
        self._released_members: set[int] = set()
        self.members = tuple(
            PackedKVCacheMember(
                self,
                member_index=index,
                offset=offset,
                length=length,
            )
            for index, (offset, length) in enumerate(zip(offsets, lengths))
        )

    @property
    def cache(self) -> LocalPaddleOCRVLStaticCache:
        return self._buffer.cache

    @property
    def bucket(self) -> int:
        return self._buffer.bucket

    @property
    def buffer_id(self) -> int:
        return self._buffer.buffer_id

    @property
    def generation(self) -> int:
        return self._buffer.generation

    @property
    def remaining_members(self) -> int:
        return len(self.members) - len(self._released_members)

    @property
    def released(self) -> bool:
        return self.remaining_members == 0

    def _release_member(self, member_index: int) -> None:
        if member_index in self._released_members:
            raise RuntimeError("packed KV member was released twice")
        self._released_members.add(member_index)
        if self.released:
            self._pool._return(self)

    def release_all(self) -> None:
        for member in self.members:
            if not member.released:
                member.release()


class PackedKVCachePool:
    """Bucket-local pool of compact caches used by packed text prefill."""

    def __init__(
        self,
        allocator: Callable[[int], LocalPaddleOCRVLStaticCache],
    ) -> None:
        self._allocator = allocator
        self._free: dict[int, list[_PackedKVBuffer]] = defaultdict(list)
        self._active: dict[int, PackedKVCacheLease] = {}
        self._buffers: dict[int, _PackedKVBuffer] = {}
        self._next_buffer_id = 0
        self.acquisitions = 0
        self.allocations = 0
        self.reuses = 0
        self.high_water_active_buffers = 0
        self.high_water_active_bytes = 0

    @property
    def active_buffers(self) -> int:
        return len(self._active)

    @property
    def active_bytes(self) -> int:
        return sum(
            lease._buffer.nbytes for lease in self._active.values()
        )

    @property
    def allocated_bytes(self) -> int:
        return sum(buffer.nbytes for buffer in self._buffers.values())

    def acquire(
        self,
        bucket: int,
        *,
        segment_offsets: Iterable[int],
        segment_lengths: Iterable[int],
    ) -> PackedKVCacheLease:
        bucket = int(bucket)
        offsets = tuple(int(value) for value in segment_offsets)
        lengths = tuple(int(value) for value in segment_lengths)
        if not offsets or len(offsets) != len(lengths):
            raise ValueError("packed KV lease requires aligned non-empty segments")
        for offset, length in zip(offsets, lengths):
            if offset < 0 or length <= 0 or offset + length > bucket:
                raise ValueError(
                    "packed KV segment is outside its bucket: "
                    f"offset={offset} length={length} bucket={bucket}"
                )

        free = self._free[bucket]
        if free:
            buffer = free.pop()
            self.reuses += 1
        else:
            cache = self._allocator(bucket)
            if int(cache.cache_length) != bucket:
                raise ValueError(
                    "packed KV allocator returned the wrong cache length: "
                    f"expected={bucket} got={cache.cache_length}"
                )
            buffer = _PackedKVBuffer(
                buffer_id=self._next_buffer_id,
                bucket=bucket,
                cache=cache,
                nbytes=static_cache_nbytes(cache),
            )
            self._next_buffer_id += 1
            self._buffers[buffer.buffer_id] = buffer
            self.allocations += 1
        buffer.generation += 1
        lease = PackedKVCacheLease(self, buffer, offsets, lengths)
        if buffer.buffer_id in self._active:
            raise RuntimeError("packed KV pool handed out an active buffer")
        self._active[buffer.buffer_id] = lease
        self.acquisitions += 1
        self.high_water_active_buffers = max(
            self.high_water_active_buffers,
            self.active_buffers,
        )
        self.high_water_active_bytes = max(
            self.high_water_active_bytes,
            self.active_bytes,
        )
        return lease

    def _return(self, lease: PackedKVCacheLease) -> None:
        buffer_id = lease.buffer_id
        active = self._active.get(buffer_id)
        if active is not lease:
            raise RuntimeError("packed KV pool received a stale lease")
        del self._active[buffer_id]
        self._free[lease.bucket].append(lease._buffer)

    def stats(self) -> dict[str, int]:
        return {
            "acquisitions": self.acquisitions,
            "allocations": self.allocations,
            "reuses": self.reuses,
            "active_buffers": self.active_buffers,
            "active_bytes": self.active_bytes,
            "allocated_buffers": len(self._buffers),
            "allocated_bytes": self.allocated_bytes,
            "high_water_active_buffers": self.high_water_active_buffers,
            "high_water_active_bytes": self.high_water_active_bytes,
        }
