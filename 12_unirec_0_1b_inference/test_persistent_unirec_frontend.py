#!/usr/bin/env python3
"""CPU lifecycle tests for the persistent UniRec frontend."""

from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue
import tempfile
import unittest

from persistent_unirec_frontend import PersistentUniRecFrontend


class FakeLayout:
    def __init__(self) -> None:
        self.tasks: Queue[dict[str, object] | None] = Queue()
        self.closed = False

    def submit(self, **task: object) -> None:
        self.tasks.put(dict(task))

    def receive_event(self) -> dict[str, object]:
        task = self.tasks.get()
        if task is None:
            return {"status": "layout_closed"}
        return {
            "status": "layout_page",
            **task,
            "rgb_descriptor": None,
            "layout_result": {"boxes": []},
        }

    def request_close(self) -> None:
        self.tasks.put(None)

    def close(self) -> None:
        self.closed = True


class FakeCropPool:
    def __init__(self) -> None:
        self.results: Queue[dict[str, object]] = Queue()
        self.closed = False

    def submit(self, **task: object) -> None:
        self.results.put(
            {
                "status": "ok",
                "page_index": task["page_index"],
                "request_id": task["request_id"],
                "payload": {
                    "page_index": task["page_index"],
                    "request_id": task["request_id"],
                    "image_path": str(task["path"]),
                    "crops": [],
                },
            }
        )

    def receive(self, *, timeout: float) -> dict[str, object]:
        try:
            return self.results.get(timeout=timeout)
        except Empty as exception:
            raise TimeoutError from exception

    def close(self) -> None:
        self.closed = True


class PersistentUniRecFrontendTest(unittest.TestCase):
    def test_accepts_later_requests_without_restart(self) -> None:
        layout = FakeLayout()
        crops = FakeCropPool()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_path = root / "first.jpg"
            second_path = root / "second.jpg"
            first_path.touch()
            second_path.touch()
            frontend = PersistentUniRecFrontend(
                layout=layout,
                crop_pool=crops,
            )
            first = frontend.submit(first_path, request_id="first")
            self.assertEqual(first.result(timeout=1.0)["request_id"], "first")
            frontend.wait_idle(timeout=1.0)
            self.assertEqual(frontend.snapshot()["inflight"], 0)

            second = frontend.submit(second_path, request_id="second")
            self.assertEqual(second.result(timeout=1.0)["request_id"], "second")
            frontend.wait_idle(timeout=1.0)
            frontend.close()

        self.assertTrue(layout.closed)
        self.assertTrue(crops.closed)
        self.assertEqual(frontend.snapshot()["submitted"], 2)
        self.assertEqual(frontend.snapshot()["crop_completed"], 2)


if __name__ == "__main__":
    unittest.main()
