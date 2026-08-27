"""Request-facing lifecycle for the persistent UniRec pipeline."""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from threading import Condition
import time
from typing import Any, Iterable


class PersistentUniRecService:
    """Accept independent page requests and complete them in memory."""

    def __init__(self, *, frontend: Any, npu_pipeline: Any) -> None:
        self.frontend = frontend
        self.npu_pipeline = npu_pipeline
        self._condition = Condition()
        self._futures: dict[str, Future[dict[str, Any]]] = {}
        self._submitted = 0
        self._completed = 0
        self._next_request = 0
        self._closed = False
        self._measurement_started_at: float | None = None
        self._measurement_submitted_base = 0
        self._measurement_completed_base = 0

    def complete(self, request_id: str, response: dict[str, Any]) -> None:
        with self._condition:
            try:
                future = self._futures.pop(request_id)
            except KeyError as exception:
                raise RuntimeError(
                    f"completed unknown service request {request_id}"
                ) from exception
            future.set_result(response)
            self._completed += 1
            self._condition.notify_all()

    def fail(self, exception: BaseException) -> None:
        """Fail every request still owned by the service."""
        with self._condition:
            pending = list(self._futures.values())
            self._futures.clear()
            for future in pending:
                if not future.done():
                    future.set_exception(exception)
                    self._completed += 1
            self._condition.notify_all()

    def submit(
        self,
        path: Path,
        *,
        request_id: str | None = None,
    ) -> Future[dict[str, Any]]:
        with self._condition:
            if self._closed:
                raise RuntimeError("cannot submit after service close")
            identifier = request_id or f"request-{self._next_request:012d}"
            self._next_request += 1
            if identifier in self._futures:
                raise RuntimeError(f"duplicate service request id {identifier}")
            future: Future[dict[str, Any]] = Future()
            self._futures[identifier] = future
            self._submitted += 1
        try:
            self.npu_pipeline.register_request(identifier)
            frontend_future = self.frontend.submit(
                path,
                request_id=identifier,
            )
        except BaseException as exception:
            self.npu_pipeline.cancel_request(identifier)
            with self._condition:
                self._futures.pop(identifier, None)
                future.set_exception(exception)
                self._completed += 1
                self._condition.notify_all()
            raise

        def propagate_frontend_error(completed: Future[Any]) -> None:
            exception = completed.exception()
            if exception is None:
                return
            self.npu_pipeline.cancel_request(identifier)
            with self._condition:
                pending = self._futures.pop(identifier, None)
                if pending is not None and not pending.done():
                    pending.set_exception(exception)
                    self._completed += 1
                self._condition.notify_all()

        frontend_future.add_done_callback(propagate_frontend_error)
        return future

    def submit_many(
        self,
        paths: Iterable[Path],
        *,
        request_prefix: str,
    ) -> list[Future[dict[str, Any]]]:
        return [
            self.submit(
                path,
                request_id=f"{request_prefix}-{index:06d}",
            )
            for index, path in enumerate(paths)
        ]

    @staticmethod
    def wait_futures(
        futures: Iterable[Future[dict[str, Any]]],
        *,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        started = time.monotonic()
        results = []
        for future in futures:
            remaining = (
                None
                if timeout is None
                else max(0.0, timeout - (time.monotonic() - started))
            )
            results.append(future.result(timeout=remaining))
        return results

    def wait_idle(self, *, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._completed < self._submitted:
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        "persistent service did not become idle: "
                        f"completed={self._completed} submitted={self._submitted}"
                    )
                self._condition.wait(timeout=remaining)
        self.frontend.wait_idle(timeout=timeout)
        self.npu_pipeline.wait_idle(timeout=timeout)

    def reset_measurement(self) -> dict[str, Any]:
        self.wait_idle()
        prior_frontend_metrics = self.frontend.reset_metrics()
        prior_npu_metrics = self.npu_pipeline.reset_metrics()
        with self._condition:
            if self._futures:
                raise RuntimeError("cannot reset with pending service futures")
            self._measurement_submitted_base = self._submitted
            self._measurement_completed_base = self._completed
            self._measurement_started_at = time.perf_counter()
        return {
            "frontend": prior_frontend_metrics,
            "npu": prior_npu_metrics,
        }

    def measurement(self) -> dict[str, Any]:
        with self._condition:
            if self._measurement_started_at is None:
                raise RuntimeError("measurement was not started")
            wall_s = time.perf_counter() - self._measurement_started_at
            submitted = self._submitted - self._measurement_submitted_base
            completed = self._completed - self._measurement_completed_base
        return {
            "wall_s": wall_s,
            "submitted_pages": submitted,
            "completed_pages": completed,
            "pages_per_s": completed / wall_s if wall_s > 0 else None,
            "frontend": self.frontend.snapshot(),
            "npu": self.npu_pipeline.metrics(),
        }

    def close(self) -> dict[str, Any]:
        with self._condition:
            if self._closed:
                raise RuntimeError("persistent service was already closed")
            self._closed = True
        self.frontend.close()
        self.npu_pipeline.wait_idle()
        return self.npu_pipeline.close()
