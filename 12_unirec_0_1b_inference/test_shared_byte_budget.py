#!/usr/bin/env python3
"""CPU-only checks for variable-sized process-shared credits."""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
import unittest

from shared_byte_budget import SharedByteBudget


def _reserve_in_spawned_process(
    budget: SharedByteBudget,
    acquired: object,
) -> None:
    budget.reserve(30)
    acquired.set()
    budget.release(30)


class SharedByteBudgetTest(unittest.TestCase):
    def test_reservation_waits_instead_of_oversubscribing(self) -> None:
        budget = SharedByteBudget(mp.get_context("spawn"), 100)
        budget.reserve(80)
        acquired = threading.Event()

        def reserve_tail() -> None:
            budget.reserve(30)
            acquired.set()

        thread = threading.Thread(target=reserve_tail)
        thread.start()
        time.sleep(0.02)
        self.assertFalse(acquired.is_set())
        self.assertEqual(budget.snapshot()["peak_bytes"], 80)
        budget.release(80)
        thread.join(timeout=1.0)
        self.assertTrue(acquired.is_set())
        self.assertLessEqual(budget.snapshot()["peak_bytes"], 100)
        budget.release(30)
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["live_bytes"], 0)
        self.assertEqual(snapshot["reservation_count"], 2)
        self.assertEqual(snapshot["release_count"], 2)
        self.assertEqual(snapshot["wait_count"], 1)

    def test_oversized_single_payload_fails_immediately(self) -> None:
        budget = SharedByteBudget(mp.get_context("spawn"), 100)
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            budget.reserve(101)
        self.assertEqual(budget.snapshot()["live_bytes"], 0)

    def test_spawned_producer_shares_the_same_credits(self) -> None:
        context = mp.get_context("spawn")
        budget = SharedByteBudget(context, 100)
        budget.reserve(80)
        acquired = context.Event()
        process = context.Process(
            target=_reserve_in_spawned_process,
            args=(budget, acquired),
        )
        process.start()
        self.assertFalse(acquired.wait(timeout=0.05))
        budget.release(80)
        self.assertTrue(acquired.wait(timeout=2.0))
        process.join(timeout=2.0)
        self.assertEqual(process.exitcode, 0)
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["live_bytes"], 0)
        self.assertEqual(snapshot["reservation_count"], 2)
        self.assertEqual(snapshot["release_count"], 2)


if __name__ == "__main__":
    unittest.main()
