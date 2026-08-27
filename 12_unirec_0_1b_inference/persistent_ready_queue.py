"""Bounded request queue with distinct idle and closed states."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock
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


class _CloseMarker:
    pass


_CLOSE = _CloseMarker()


class PersistentReadyQueue(Generic[ItemT]):
    """Single-consumer queue for an always-on decode service."""

    def __init__(self, maxsize: int = 0) -> None:
        if maxsize < 0:
            raise ValueError("queue maxsize cannot be negative")
        self._queue: Queue[ItemT | _CloseMarker] = Queue(maxsize=maxsize)
        self._lock = Lock()
        self._close_requested = False
        self._close_observed = False

    def put(self, item: ItemT, *, timeout: float | None = None) -> None:
        if item is None:
            raise ValueError("None is not a valid ready item")
        with self._lock:
            if self._close_requested:
                raise RuntimeError("cannot put work after the queue was closed")
        if timeout is None:
            self._queue.put(item)
        else:
            self._queue.put(item, timeout=timeout)

    def close(self) -> None:
        with self._lock:
            if self._close_requested:
                return
            self._close_requested = True
        self._queue.put(_CLOSE)

    def pull(
        self,
        *,
        wait: bool,
        timeout: float | None = None,
    ) -> ReadyQueuePull[ItemT]:
        if self._close_observed:
            return ReadyQueuePull(closed=True)
        try:
            if wait:
                if timeout is None:
                    value = self._queue.get()
                else:
                    value = self._queue.get(timeout=max(0.0, timeout))
            else:
                value = self._queue.get_nowait()
        except Empty:
            return ReadyQueuePull()
        if value is _CLOSE:
            self._close_observed = True
            return ReadyQueuePull(closed=True)
        return ReadyQueuePull(item=value)

    @property
    def close_requested(self) -> bool:
        return self._close_requested

    @property
    def close_observed(self) -> bool:
        return self._close_observed

    def qsize(self) -> int:
        return self._queue.qsize()
