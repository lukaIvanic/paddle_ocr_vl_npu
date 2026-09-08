from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layout_preparation import LayoutPreparation
from page_source import PageInbox, PageSource
from coordinator import Coordinator
from test_coordinator import Engine
from routing import Routing


class LayoutTests(unittest.TestCase):
    def test_persistent_workers_owner_detection_and_bounded_fifo(self):
        owner = get_ident()
        wake = Event()
        def detect(value):
            self.assertEqual(get_ident(), owner)
            return value
        stages = LayoutPreparation(lambda p, i: i, detect, lambda x: x)
        stages.notify = wake.set
        next_input, results = 0, []
        try:
            while len(results) < 20:
                wake.clear()
                stages.check_errors()
                if stages.input_future is None and next_input < 20:
                    stages.submit(None, next_input)
                    next_input += 1
                if stages.available:
                    result = stages.advance()
                    if result is not None:
                        results.append(result)
                else:
                    self.assertTrue(wake.wait(2))
            self.assertEqual(results, list(range(20)))
            self.assertFalse(stages.pending)
            self.assertEqual(stages.summary()["high_water_pages"], 2)
            for stage in ("input", "crops"):
                self.assertEqual(len(stages.threads[stage]), 1)
                self.assertNotIn(owner, stages.threads[stage])
        finally:
            stages.close()

    def test_completed_input_still_reserves_its_slot(self):
        stages = LayoutPreparation(lambda p, i: i, lambda x: x, lambda x: x)
        try:
            stages.submit(None, 0)
            stages.input_future.result(timeout=2)
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                stages.submit(None, 1)
        finally:
            stages.close()

    def test_errors_from_both_cpu_stages_surface(self):
        def fail(*args):
            raise ValueError("bad page")
        for where in ("input", "crops"):
            stages = LayoutPreparation(fail if where == "input" else lambda p, i: i,
                                       lambda x: x, fail if where == "crops" else lambda x: x)
            try:
                stages.submit(None, 0)
                if where == "crops":
                    stages.input_future.result(timeout=2)
                    stages.advance()
                    future = stages.crop_future
                else:
                    future = stages.input_future
                with self.assertRaisesRegex(ValueError, "bad page"):
                    future.result(timeout=2)
                with self.assertRaisesRegex(ValueError, "bad page"):
                    stages.check_errors()
            finally:
                stages.close()

    def test_inflight_last_page_is_upstream_and_full_decode_can_run(self):
        release = Event()
        entered = Event()
        class Source(PageSource):
            def prepare_input(self, path, ordinal):
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("owner blocked instead of decoding")
                return ordinal
            def detect(self, ordinal):
                return ordinal
            def prepare_crops(self, ordinal):
                return SimpleNamespace(ordinal=ordinal, requests=[], request_block_indices=[], timing_s={}), 0.0
            def finish(self, ordinal):
                self.pages.pop(ordinal)
                self.completed += 1
        inbox = PageInbox()
        inbox.submit("page.png")
        inbox.close()
        source = Source(inbox, None, Routing("paddle", "paddle", "paddle"), None, None)
        engine = Engine()
        engine.enqueue_page = engine.pending.extend
        engine.occupied = engine.capacity
        advance = engine.advance
        def decode(count):
            release.set()
            advance(count)
        engine.advance = decode
        try:
            source.pump()
            self.assertTrue(entered.wait(2))
            self.assertFalse(inbox.items)
            self.assertTrue(source.has_pending)
            self.assertFalse(source.exhausted)
            self.assertFalse(source.can_advance)
            Coordinator({"paddle": engine}, source).run()
            self.assertEqual(source.completed, 1)
            self.assertTrue(source.exhausted)
            self.assertTrue(engine.done)
            self.assertEqual(source.summary()["counts"], {"submitted": 1, "input": 1, "detected": 1, "crops": 1})
        finally:
            release.set()
            source.close()
