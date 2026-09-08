from collections import deque
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_preparation import CpuPreparation
from coordinator import Coordinator
from test_page_prefill import SimulatedAdapter, Pages


def request(index):
    return SimpleNamespace(request_id=str(index), steps=1 + index % 3)


class PreparedAdapter(SimulatedAdapter):
    def __init__(self, prepare):
        super().__init__(4)
        self.start_cpu_preparation(prepare, 4, "test-persistent-cpu")

    def prefill(self):
        _, prepared = self.take_prepared_requests()
        self.ready.extend(prepared)


class CpuTests(unittest.TestCase):
    def test_storage_bound_and_persistent_thread(self):
        owner = get_ident()
        worker = CpuPreparation(lambda r: (get_ident(), r.request_id), 4, "test-cpu")
        pages = deque([deque(request(i) for i in range(10))])
        try:
            worker.pump(pages, lambda: None)
            self.assertEqual(len(worker.futures), 4)
            for f in worker.futures.values():
                f.result(timeout=2)
            worker.pump(pages, lambda: None)
            self.assertEqual(len(worker.futures), 4)  # Completed arrays still count.
            values = worker.take([pages[0].popleft() for _ in range(4)])
            worker.pump(pages, lambda: None)
            for f in worker.futures.values():
                f.result(timeout=2)
            values += worker.take([pages[0].popleft() for _ in range(4)])
            self.assertEqual(len({v[0] for v in values}), 1)
            self.assertNotEqual(values[0][0], owner)
            self.assertEqual(worker.summary()["high_water_requests"], 4)
        finally:
            worker.close()

    def test_closed_input_waits_for_cpu_then_drains(self):
        adapter = PreparedAdapter(lambda r: r.steps)
        try:
            Coordinator({"unirec": adapter}, Pages([
                {"unirec": [request(i) for i in range(31)]}
            ])).run()
            self.assertEqual(adapter.finished, 31)
            self.assertEqual(adapter.cpu.consumed, 31)
            self.assertEqual(adapter.cpu.summary()["worker_threads_observed"], 1)
        finally:
            adapter.cpu.close()

    def test_other_full_decoder_can_run_while_cpu_is_blocked(self):
        release = Event()
        entered = Event()
        def prepare(r):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("CPU worker was not released")
            return r.steps
        unirec = PreparedAdapter(prepare)
        paddle = SimulatedAdapter(2)
        paddle.slots = [1, 1]
        paddle.active = 2
        advance = paddle.advance
        def run_paddle(count):
            release.set()
            advance(count)
        paddle.advance = run_paddle
        unirec.enqueue_page([request(0)])
        try:
            unirec.pump_preparation(lambda: None)
            self.assertTrue(entered.wait(2))
            Coordinator({"unirec": unirec, "paddle": paddle}, Pages([])).run()
            self.assertEqual(unirec.finished, 1)
            self.assertEqual(paddle.finished, 2)
        finally:
            release.set()
            unirec.cpu.close()

    def test_worker_error_is_propagated(self):
        def fail(_request):
            raise ValueError("bad crop")
        adapter = PreparedAdapter(fail)
        try:
            with self.assertRaisesRegex(ValueError, "bad crop"):
                Coordinator({"unirec": adapter}, Pages([{"unirec": [request(0)]}])).run()
        finally:
            adapter.cpu.close()
