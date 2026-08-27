#!/usr/bin/env python3
"""CPU request-lifecycle tests for the persistent UniRec service."""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
import tempfile
import unittest

from persistent_unirec_service import PersistentUniRecService


class FakeNpuPipeline:
    def __init__(self) -> None:
        self.complete = None
        self.payloads = []
        self.resets = 0

    def submit(self, payload: dict[str, object]) -> None:
        self.payloads.append(payload)
        assert self.complete is not None
        self.complete(
            str(payload["request_id"]),
            {"markdown": str(payload["request_id"])},
        )

    def wait_idle(self, *, timeout: float | None = None) -> None:
        del timeout

    def reset_metrics(self) -> dict[str, int]:
        self.resets += 1
        return {"reset": self.resets}

    def metrics(self) -> dict[str, int]:
        return {"payloads": len(self.payloads)}

    def close(self) -> dict[str, bool]:
        return {"closed": True}


class FakeFrontend:
    def __init__(self, npu: FakeNpuPipeline) -> None:
        self.npu = npu
        self.submitted = 0
        self.closed = False

    def submit(self, path: Path, *, request_id: str) -> Future[dict[str, object]]:
        self.submitted += 1
        payload = {
            "request_id": request_id,
            "image_path": str(path),
            "crops": [],
        }
        self.npu.submit(payload)
        future: Future[dict[str, object]] = Future()
        future.set_result(payload)
        return future

    def wait_idle(self, *, timeout: float | None = None) -> None:
        del timeout

    def snapshot(self) -> dict[str, int]:
        return {"submitted": self.submitted}

    def close(self) -> None:
        self.closed = True


class PersistentUniRecServiceTest(unittest.TestCase):
    def test_warmup_reset_then_later_measured_requests(self) -> None:
        npu = FakeNpuPipeline()
        frontend = FakeFrontend(npu)
        service = PersistentUniRecService(frontend=frontend, npu_pipeline=npu)
        npu.complete = service.complete
        with tempfile.TemporaryDirectory() as temporary:
            page = Path(temporary) / "page.jpg"
            page.touch()
            warmup = service.submit(page, request_id="warmup")
            self.assertEqual(warmup.result(timeout=1.0)["markdown"], "warmup")
            service.reset_measurement()

            first = service.submit(page, request_id="measured-1")
            second = service.submit(page, request_id="measured-2")
            self.assertEqual(first.result(timeout=1.0)["markdown"], "measured-1")
            self.assertEqual(second.result(timeout=1.0)["markdown"], "measured-2")
            service.wait_idle(timeout=1.0)
            measurement = service.measurement()
            self.assertEqual(measurement["submitted_pages"], 2)
            self.assertEqual(measurement["completed_pages"], 2)
            self.assertGreater(measurement["pages_per_s"], 0)
            self.assertEqual(service.close(), {"closed": True})

        self.assertTrue(frontend.closed)


if __name__ == "__main__":
    unittest.main()
