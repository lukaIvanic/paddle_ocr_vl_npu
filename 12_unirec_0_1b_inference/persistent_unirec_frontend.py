"""Persistent layout and crop-preparation service for UniRec pages."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Thread
import time
from typing import Any, Callable


@dataclass(frozen=True)
class FrontendRequest:
    request_id: str
    page_index: int
    path: Path
    submitted_at: float


class PersistentUniRecFrontend:
    """Route arbitrary page requests through hot layout and CPU crop workers."""

    def __init__(
        self,
        *,
        layout: Any,
        crop_pool: Any,
        on_page_ready: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.layout = layout
        self.crop_pool = crop_pool
        self.on_page_ready = on_page_ready
        self._condition = Condition()
        self._requests: dict[int, FrontendRequest] = {}
        self._futures: dict[int, Future[dict[str, Any]]] = {}
        self._next_page_index = 0
        self._submitted = 0
        self._layout_completed = 0
        self._crop_completed = 0
        self._layout_closed = False
        self._close_requested = False
        self._error: BaseException | None = None
        self._layout_thread = Thread(
            target=self._layout_results_loop,
            name="unirec-serving-layout-results",
            daemon=True,
        )
        self._crop_thread = Thread(
            target=self._crop_results_loop,
            name="unirec-serving-crop-results",
            daemon=True,
        )
        self._layout_thread.start()
        self._crop_thread.start()

    def _record_error(self, exception: BaseException) -> None:
        with self._condition:
            if self._error is None:
                self._error = exception
            for future in self._futures.values():
                if not future.done():
                    future.set_exception(exception)
            self._condition.notify_all()

    def _layout_results_loop(self) -> None:
        try:
            while True:
                item = self.layout.receive_event()
                status = item["status"]
                if status == "layout_closed":
                    with self._condition:
                        self._layout_closed = True
                        self._condition.notify_all()
                    return
                page_index = int(item["page_index"])
                with self._condition:
                    request = self._requests[page_index]
                self.crop_pool.submit(
                    request_id=request.request_id,
                    page_index=page_index,
                    path=request.path,
                    rgb=None,
                    rgb_descriptor=item.get("rgb_descriptor"),
                    layout_result=item["layout_result"],
                    started_at=request.submitted_at,
                )
                with self._condition:
                    self._layout_completed += 1
                    self._condition.notify_all()
        except BaseException as exception:
            self._record_error(exception)

    def _crop_results_loop(self) -> None:
        try:
            while True:
                with self._condition:
                    done = (
                        self._layout_closed
                        and self._crop_completed >= self._layout_completed
                    )
                    failed = self._error is not None
                if done or failed:
                    return
                try:
                    item = self.crop_pool.receive(timeout=0.1)
                except TimeoutError:
                    continue
                page_index = int(item["page_index"])
                payload = item["payload"]
                callback = self.on_page_ready
                if callback is not None:
                    callback(payload)
                with self._condition:
                    future = self._futures.pop(page_index)
                    self._requests.pop(page_index, None)
                    if not future.done():
                        future.set_result(payload)
                    self._crop_completed += 1
                    self._condition.notify_all()
        except BaseException as exception:
            self._record_error(exception)

    def submit(
        self,
        path: Path,
        *,
        request_id: str | None = None,
    ) -> Future[dict[str, Any]]:
        resolved = path.expanduser().resolve()
        with self._condition:
            if self._close_requested:
                raise RuntimeError("cannot submit after frontend close")
            if self._error is not None:
                raise RuntimeError("persistent frontend failed") from self._error
            page_index = self._next_page_index
            self._next_page_index += 1
            identifier = request_id or f"page-{page_index:012d}"
            request = FrontendRequest(
                request_id=identifier,
                page_index=page_index,
                path=resolved,
                submitted_at=time.perf_counter(),
            )
            future: Future[dict[str, Any]] = Future()
            self._requests[page_index] = request
            self._futures[page_index] = future
            self._submitted += 1
        try:
            self.layout.submit(
                request_id=identifier,
                page_index=page_index,
                path=resolved,
                started_at=request.submitted_at,
            )
        except BaseException as exception:
            self._record_error(exception)
            raise
        return future

    def wait_idle(self, *, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._crop_completed < self._submitted:
                if self._error is not None:
                    raise RuntimeError("persistent frontend failed") from self._error
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        "persistent frontend did not become idle: "
                        f"completed={self._crop_completed} "
                        f"submitted={self._submitted}"
                    )
                self._condition.wait(timeout=remaining)

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "submitted": self._submitted,
                "layout_completed": self._layout_completed,
                "crop_completed": self._crop_completed,
                "inflight": self._submitted - self._crop_completed,
                "close_requested": self._close_requested,
                "layout_closed": self._layout_closed,
                "failed": self._error is not None,
            }

    def close(self) -> None:
        with self._condition:
            if self._close_requested:
                return
            self._close_requested = True
        self.layout.request_close()
        self._layout_thread.join(timeout=1800.0)
        if self._layout_thread.is_alive():
            raise RuntimeError("persistent layout result thread did not stop")
        self._crop_thread.join(timeout=1800.0)
        if self._crop_thread.is_alive():
            raise RuntimeError("persistent crop result thread did not stop")
        self.crop_pool.close()
        self.layout.close()

    def __enter__(self) -> "PersistentUniRecFrontend":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
