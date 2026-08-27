"""Bounded request queue with explicit upstream and closed states."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from queue import Full
from threading import Condition
import time
from typing import Generic, TypeVar


ItemT = TypeVar("ItemT")


@dataclass(frozen=True)
class ReadyQueuePull(Generic[ItemT]):
    """One queue observation.

    ``item`` is set when work is ready. ``closed`` is true only after the
    producer closed the queue and every earlier item was consumed. Both false
    means the service is still open but no item is ready now.
    """

    item: ItemT | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        if self.item is not None and self.closed:
            raise ValueError("a ready queue pull cannot be both item and closed")

    @property
    def idle(self) -> bool:
        return self.item is None and not self.closed


class PersistentReadyQueue(Generic[ItemT]):
    """Single-consumer queue for an always-on pipeline stage.

    ``upstream_pending`` counts submitted units that can still publish work.
    A consumer can therefore wait for a full physical batch without guessing
    how long a producer might take.  If upstream drains before the batch fills,
    the partial batch becomes dispatchable immediately.
    """

    def __init__(self, maxsize: int = 0) -> None:
        if maxsize < 0:
            raise ValueError("queue maxsize cannot be negative")
        self._maxsize = int(maxsize)
        self._items: deque[ItemT] = deque()
        self._condition = Condition()
        self._close_requested = False
        self._close_observed = False
        self._upstream_pending = 0

    def register_upstream(self, count: int = 1) -> None:
        if count < 1:
            raise ValueError("upstream registration count must be positive")
        with self._condition:
            if self._close_requested:
                raise RuntimeError("cannot register upstream work after close")
            self._upstream_pending += int(count)
            self._condition.notify_all()

    def complete_upstream(self, count: int = 1) -> None:
        if count < 1:
            raise ValueError("upstream completion count must be positive")
        with self._condition:
            if count > self._upstream_pending:
                raise RuntimeError(
                    "upstream completion exceeds registration: "
                    f"complete={count} pending={self._upstream_pending}"
                )
            self._upstream_pending -= int(count)
            self._condition.notify_all()

    def put(self, item: ItemT, *, timeout: float | None = None) -> None:
        if item is None:
            raise ValueError("None is not a valid ready item")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            if self._close_requested:
                raise RuntimeError("cannot put work after the queue was closed")
            while self._maxsize and len(self._items) >= self._maxsize:
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    raise Full
                self._condition.wait(timeout=remaining)
                if self._close_requested:
                    raise RuntimeError("cannot put work after the queue was closed")
            self._items.append(item)
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if self._close_requested:
                return
            self._close_requested = True
            self._condition.notify_all()

    def pull(
        self,
        *,
        wait: bool,
        timeout: float | None = None,
    ) -> ReadyQueuePull[ItemT]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            if self._close_observed:
                return ReadyQueuePull(closed=True)
            while not self._items:
                if self._close_requested:
                    self._close_observed = True
                    return ReadyQueuePull(closed=True)
                if not wait:
                    return ReadyQueuePull()
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    return ReadyQueuePull()
                self._condition.wait(timeout=remaining)
            value = self._items.popleft()
            self._condition.notify_all()
            return ReadyQueuePull(item=value)

    def wait_until_dispatchable(
        self,
        capacity: int,
        *,
        active_count: int = 0,
    ) -> bool:
        """Wait for a full batch, upstream drain, or explicit close.

        Returns true when at least one item can be consumed.  Returns false only
        after close when no queued item remains.  An open idle service waits for
        the next upstream registration instead of polling.
        """
        if capacity < 1:
            raise ValueError("dispatch capacity must be positive")
        if active_count < 0:
            raise ValueError("active count cannot be negative")
        with self._condition:
            while True:
                if len(self._items) >= capacity:
                    return True
                if self._items and self._upstream_pending == 0:
                    return True
                if active_count and self._upstream_pending == 0:
                    return True
                if self._close_requested:
                    return bool(self._items) or bool(active_count)
                self._condition.wait()

    @property
    def close_requested(self) -> bool:
        with self._condition:
            return self._close_requested

    @property
    def close_observed(self) -> bool:
        with self._condition:
            return self._close_observed

    @property
    def upstream_pending(self) -> int:
        with self._condition:
            return self._upstream_pending

    def qsize(self) -> int:
        with self._condition:
            return len(self._items)
