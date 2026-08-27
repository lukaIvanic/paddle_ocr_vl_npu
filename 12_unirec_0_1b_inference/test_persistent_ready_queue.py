#!/usr/bin/env python3
"""CPU tests for the persistent UniRec decode input queue."""

from __future__ import annotations

import threading
import time
import unittest

from persistent_ready_queue import PersistentReadyQueue


class PersistentReadyQueueTest(unittest.TestCase):
    def test_idle_is_not_end_of_service(self) -> None:
        source: PersistentReadyQueue[int] = PersistentReadyQueue()
        first = source.pull(wait=False)
        self.assertTrue(first.idle)
        self.assertFalse(first.closed)

        source.put(7)
        second = source.pull(wait=False)
        self.assertEqual(second.item, 7)
        self.assertFalse(second.closed)

        third = source.pull(wait=False)
        self.assertTrue(third.idle)

    def test_close_drains_earlier_items_then_stays_closed(self) -> None:
        source: PersistentReadyQueue[str] = PersistentReadyQueue()
        source.put("first")
        source.put("second")
        source.close()

        self.assertEqual(source.pull(wait=False).item, "first")
        self.assertEqual(source.pull(wait=False).item, "second")
        self.assertTrue(source.pull(wait=False).closed)
        self.assertTrue(source.pull(wait=False).closed)
        with self.assertRaisesRegex(RuntimeError, "after the queue was closed"):
            source.put("late")

    def test_blocking_pull_wakes_for_later_request(self) -> None:
        source: PersistentReadyQueue[int] = PersistentReadyQueue()

        def publish() -> None:
            time.sleep(0.02)
            source.put(11)

        thread = threading.Thread(target=publish)
        thread.start()
        started = time.perf_counter()
        pulled = source.pull(wait=True, timeout=1.0)
        elapsed = time.perf_counter() - started
        thread.join()

        self.assertEqual(pulled.item, 11)
        self.assertGreaterEqual(elapsed, 0.01)

    def test_timed_pull_returns_idle_without_closing(self) -> None:
        source: PersistentReadyQueue[int] = PersistentReadyQueue()
        started = time.perf_counter()
        pulled = source.pull(wait=True, timeout=0.02)
        elapsed = time.perf_counter() - started
        self.assertTrue(pulled.idle)
        self.assertGreaterEqual(elapsed, 0.01)
        source.put(3)
        self.assertEqual(source.pull(wait=False).item, 3)


if __name__ == "__main__":
    unittest.main()
