import argparse
from collections import deque
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coordinator import Coordinator
from routing import Routing, add_arguments


class Engine:
    def __init__(self, capacity=2):
        self.capacity = capacity
        self.occupied = 0
        self.pending = deque()
        self.done = False
        self.upstream = True
        self.completed = 0
        self.closed = False

    @property
    def free(self):
        return self.capacity - self.occupied

    def set_upstream(self, pending, *, closed=False):
        self.upstream = pending
        self.closed = closed

    def prefill(self):
        while self.free and self.pending:
            self.pending.popleft()
            self.occupied += 1

    def advance(self, count):
        self.completed += self.occupied
        self.occupied = 0
        self.done = self.closed and not self.upstream


class Pages:
    def __init__(self, routes):
        self.routes = deque(routes)
        self.exhausted = not self.routes

    @property
    def has_pending(self):
        return bool(self.routes)

    def advance(self, engines):
        for model in self.routes.popleft():
            engines[model].pending.append(object())
        self.exhausted = not self.routes


class Tests(unittest.TestCase):
    def test_report_uses_exclusive_scheduled_time(self):
        from summarize_run import summarize
        run = {
            "pages": 2, "wall_s": 20, "pages_per_s": .1, "setup_s": 5,
            "action_wall_s": {"unirec.decode": 4},
            "peak_torch_allocated_bytes": 2**30,
            "peak_torch_reserved_bytes": 2**31,
            "engines": {"unirec": {
                "capacity": 128, "prefill_tokens": {}, "prefill_device_s": {},
                "summary": {"raw_decode_token_slots": 1280,
                            "effective_decode_tokens": 640,
                            "decode_iterations": 10, "decode_s": 2,
                            "timing_detail": {"run_wall_s": 100}},
            }},
        }
        report = summarize(run)["engines"]["unirec"]
        self.assertEqual(report["active_slot_utilization"], .5)
        self.assertEqual(report["mean_active_slots"], 64)
        self.assertEqual(report["scheduled_useful_tok_s"], 160)
        self.assertEqual(report["execution_useful_tok_s"], 320)
        run["action_wall_s"]["paddle.decode"] = 4
        run["engines"]["paddle"] = {
            "capacity": 64,
            "prefill_tokens": {"vision_real": 80, "vision_physical": 100, "text_input": 40},
            "prefill_device_s": {"vision_prefill": 2, "text_prefill": 2},
            "summary": {"raw_decode_token_slots": 640, "effective_decode_tokens": 320,
                        "active_decode_token_slots": 352, "graph_calls": 10,
                        "timing_s": {"decode_model_and_argmax_device": 2}},
        }
        report = summarize(run)["engines"]["paddle"]
        self.assertEqual(report["vision_useful_token_fraction"], .8)
        self.assertEqual(report["active_slot_utilization"], .55)
        self.assertEqual(report["useful_slot_utilization"], .5)

    def test_routing(self):
        p = argparse.ArgumentParser()
        add_arguments(p)
        args = p.parse_args([])
        r = Routing(args.text_model, args.table_model, args.formula_model)
        self.assertEqual(r.model_for("OCR:"), "unirec")
        self.assertEqual(r.model_for("Table Recognition:"), "paddle")
        self.assertEqual(Routing("paddle", "paddle", "paddle").models, ("paddle",))

    def test_full_priority_and_alternating_ties(self):
        engines = {n: Engine() for n in ("unirec", "paddle")}
        c = Coordinator(engines, Pages([["paddle"]]))
        engines["unirec"].occupied = 2
        engines["paddle"].pending.append(1)
        self.assertEqual(c.action(), ("unirec", "decode"))
        engines["paddle"].occupied = 2
        c.last_model = "unirec"
        self.assertEqual(c.action(), ("paddle", "decode"))

    def test_sparse_pages_and_partial_tail(self):
        engines = {n: Engine() for n in ("unirec", "paddle")}
        pages = Pages([["paddle"]] + [["unirec"]] * 101)
        c = Coordinator(engines, pages)
        self.assertEqual(c.action(), (None, "layout"))
        c.run()
        self.assertEqual(engines["paddle"].completed, 1)
        self.assertEqual(engines["unirec"].completed, 101)
        self.assertTrue(all(e.done for e in engines.values()))

    def test_empty_input(self):
        e = Engine()
        Coordinator({"unirec": e}, Pages([])).run()
        self.assertTrue(e.done)

    def test_summary_does_not_copy_completion_history(self):
        from dataclasses import dataclass
        from types import SimpleNamespace
        from run_pipeline import engine_report
        @dataclass
        class Summary:
            graph_calls: int
            completions: object
        adapter = SimpleNamespace(summary=Summary(12, object()), graph_calls=12, capacity=64)
        self.assertEqual(engine_report(adapter)["summary"], {"graph_calls": 12})

    def test_underfilled_requests_more_layout_before_decode(self):
        e = Engine(capacity=64)
        e.occupied = 1
        c = Coordinator({"paddle": e}, Pages([["paddle"]]))
        self.assertEqual(c.action(), (None, "layout"))

    def test_open_service_drains_then_accepts_later_input(self):
        import threading
        from page_source import PageInbox
        inbox = PageInbox()
        completed = threading.Event()
        class LiveEngine(Engine):
            def advance(self, count):
                super().advance(count)
                if self.completed:
                    completed.set()
        class LivePages:
            @property
            def has_pending(self):
                return bool(inbox.items)
            @property
            def exhausted(self):
                return inbox.closed and not inbox.items
            def advance(self, engines):
                with inbox.condition:
                    inbox.items.popleft()
                engines["paddle"].pending.append(1)
            def wait(self):
                inbox.wait()
        e = LiveEngine(capacity=64)
        c = Coordinator({"paddle": e}, LivePages())
        worker = threading.Thread(target=c.run, daemon=True)
        worker.start()
        inbox.submit("first.png")
        self.assertTrue(completed.wait(2))
        self.assertFalse(e.done)
        completed.clear()
        inbox.submit("second.png")
        self.assertTrue(completed.wait(2))
        inbox.close()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(e.completed, 2)


if __name__ == "__main__":
    unittest.main()
