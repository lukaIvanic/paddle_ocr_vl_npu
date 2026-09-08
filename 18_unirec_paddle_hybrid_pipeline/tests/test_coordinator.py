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

    @property
    def free(self):
        return self.capacity - self.occupied

    def set_upstream(self, pending, *, closed=False):
        self.upstream = pending

    def prefill(self):
        while self.free and self.pending:
            self.pending.popleft()
            self.occupied += 1

    def advance(self, count):
        self.completed += self.occupied
        self.occupied = 0
        self.done = not self.upstream


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


if __name__ == "__main__":
    unittest.main()
