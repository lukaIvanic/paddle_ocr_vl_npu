from collections import deque
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.base import Adapter
from coordinator import Coordinator


class SimulatedAdapter(Adapter):
    def __init__(self, capacity):
        super().__init__(capacity, None)
        self.ready = deque()
        self.slots = []
        self.finished = 0
        self.closed = False
        self.high_water_ready = 0

    @property
    def ready_count(self):
        return len(self.ready)

    def set_upstream(self, pending, *, closed=False):
        self.closed = closed and not pending

    def prefill(self):
        self.ready.extend(self.take_prefill_requests())
        self.high_water_ready = max(self.high_water_ready, self.ready_count)

    def advance(self, count):
        for _ in range(count):
            while len(self.slots) < self.capacity and self.ready:
                self.slots.append(self.ready.popleft())
            self.slots = [remaining - 1 for remaining in self.slots]
            self.finished += self.slots.count(0)
            self.slots = [remaining for remaining in self.slots if remaining]
            self.active = len(self.slots)
            self.graph_calls += 1
            self.done = self.closed and not self.slots and not self.ready
            if self.done or self.occupied < self.capacity:
                break


class Pages:
    def __init__(self, pages):
        self.pages = deque(pages)

    @property
    def has_pending(self):
        return bool(self.pages)

    @property
    def exhausted(self):
        return not self.pages

    def advance(self, adapters):
        for name, requests in self.pages.popleft().items():
            adapters[name].enqueue_page(requests)


class PagePrefillTests(unittest.TestCase):
    def test_paddle_closed_input_is_not_eof_with_ready_surplus(self):
        from adapters.paddle import ReadySource
        source = ReadySource()
        source.items.extend(["first", "last"])
        source.closed = True
        self.assertIsNone(source.pull_for_decode_slots(block=False, available_slots=0))
        self.assertFalse(source.closed)
        self.assertEqual(source.pull_for_decode_slots(block=False, available_slots=1), "first")
        self.assertFalse(source.closed)
        self.assertEqual(source.pull_for_decode_slots(block=False, available_slots=1), "last")
        self.assertTrue(source.closed)

    def test_prefill_is_not_limited_by_two_decode_vacancies(self):
        for capacity in (64, 128):
            adapter = SimulatedAdapter(capacity)
            adapter.active = capacity - 2
            adapter.enqueue_page(list(range(20)))
            self.assertEqual(len(adapter.take_prefill_requests()), 20)

    def test_ready_storage_bounds_large_page_and_preserves_order(self):
        adapter = SimulatedAdapter(64)
        adapter.ready.extend([1] * 10)
        adapter.enqueue_page(list(range(150)))
        first = adapter.take_prefill_requests()
        self.assertEqual(first, list(range(54)))
        adapter.ready.clear()
        self.assertEqual(adapter.take_prefill_requests(), list(range(54, 118)))
        self.assertEqual(adapter.take_prefill_requests(), list(range(118, 150)))
        self.assertFalse(adapter.pending)

    def test_does_not_mix_pages_or_enqueue_empty_pages(self):
        adapter = SimulatedAdapter(128)
        adapter.enqueue_page([])
        adapter.enqueue_page([1, 2])
        adapter.enqueue_page([3, 4])
        self.assertEqual(adapter.take_prefill_requests(), [1, 2])
        self.assertEqual(adapter.take_prefill_requests(), [3, 4])

    def test_full_decode_priority_with_surplus_ready_requests(self):
        adapter = SimulatedAdapter(64)
        adapter.active = 62
        adapter.ready.extend([1] * 20)
        adapter.enqueue_page([1] * 10)
        self.assertEqual(adapter.free, 0)
        self.assertEqual(Coordinator({"paddle": adapter}, Pages([])).action(), ("paddle", "decode"))

    def test_large_and_sparse_pages_drain_in_all_routes(self):
        for routes in (("unirec",), ("paddle",), ("unirec", "paddle")):
            adapters = {name: SimulatedAdapter(128 if name == "unirec" else 64) for name in routes}
            pages = [{name: [1 + i % 17 for i in range(333)] for name in routes}]
            pages += [{routes[i % len(routes)]: [i % 9 + 1]} for i in range(31)]
            expected = {name: sum(len(page.get(name, [])) for page in pages) for name in routes}
            Coordinator(adapters, Pages(pages)).run()
            for name, adapter in adapters.items():
                self.assertEqual(adapter.finished, expected[name])
                self.assertTrue(adapter.done)
                self.assertLessEqual(adapter.high_water_ready, adapter.ready_capacity)


if __name__ == "__main__":
    unittest.main()
