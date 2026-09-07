from types import SimpleNamespace
import threading
import unittest

from paddle_layout_source import PaddleLayoutPageSource
from streaming_pipeline import PageInbox
from test_streaming_pipeline import client, Block


class FakeLayout:
    def __init__(self, count=2):
        self.count = count
        self.detect_threads = []
        self.prepare_threads = []

    def prepare(self, name, loader, ordinal):
        self.prepare_threads.append(threading.get_ident())
        return name, loader()

    def detect(self, prepared):
        self.detect_threads.append(threading.get_ident())
        return prepared

    def crops(self, detected, client):
        blocks = [Block(type="text", bbox=[0, 0, 1, 1]) for _ in range(self.count)]
        return blocks, ([0] * self.count, ["ocr"] * self.count,
                        [None] * self.count, list(range(self.count))), {
                            "layout_timing_s": {"model": 0.1},
                            "blocks": [{"source_label": "text"}] * self.count}


class LivePaddleTests(unittest.TestCase):
    def test_order_only_merging_preserves_paddle_geometry_without_joining_pixels(self):
        import importlib.util
        import numpy as np
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "09_persistent_page_engine/pipeline/layout_postprocess.py"
        spec = importlib.util.spec_from_file_location("layout_geometry_test", path)
        geometry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(geometry)
        blocks = [{"label": "text", "box": [offset, 0, offset + 10, 5],
                   "img": np.full((5, 10, 3), offset, dtype=np.uint8)} for offset in [0, 11]]
        ordinary = geometry.merge_blocks(blocks, [])
        separate = geometry.merge_blocks(blocks, [], merge_images=False)
        self.assertEqual([r["box"] for r in ordinary], [r["box"] for r in separate])
        self.assertEqual([r["group_id"] for r in ordinary], [r["group_id"] for r in separate])
        self.assertIsNone(ordinary[1]["img"])
        self.assertEqual(ordinary[0]["img"].shape, (10, 10, 3))
        for old, new in zip(blocks, separate):
            self.assertTrue(np.array_equal(old["img"], new["img"]))

    def test_cli_defaults_and_explicit_native_option(self):
        from run_page_pipeline import pipeline_args
        args = pipeline_args(["--input-images", "/tmp/page.png", "--output-dir", "/tmp/out"])
        self.assertEqual(args.layout_backend, "pp-doclayout-v3")
        self.assertEqual(args.processor_max_pixels, 1103872)
        self.assertEqual(args.local_decode_increfa_length_mode, "pse_sentinel_310p")
        self.assertEqual(args.warmup_pages, 0)
        self.assertFalse(args.layout_graph_capture)
        self.assertEqual(pipeline_args(["--dataset-json", "/tmp/pages.json", "--output-dir", "/tmp/out",
                                        "--layout-backend", "mineru"]).layout_backend, "mineru")
        with self.assertRaises(ValueError):
            pipeline_args(["--output-dir", "/tmp/out"])

    def test_live_inbox_layout_owner_recognition_routing_and_bounded_window(self):
        owner = threading.get_ident()
        layout = FakeLayout()
        inbox = PageInbox(2)
        completed, geometry = [], []
        source = PaddleLayoutPageSource(client(), inbox, layout_frontend=layout,
            on_page=lambda name, blocks: completed.append(name),
            on_layout=lambda name, meta: geometry.append(name), page_window=1)
        try:
            self.assertIsNone(source.pull(block=True))
            self.assertFalse(source.closed)
            inbox.submit("a", lambda: 1)
            inbox.submit("b", lambda: 2)
            inbox.close_input()
            while not source.closed:
                item = source.pull(block=True)
                if item is not None:
                    self.assertEqual(source.inflight[item[0]]["phase"], "recognition")
                    source.complete(item[0], [42])
            self.assertEqual(completed, ["a", "b"])
            self.assertEqual(geometry, completed)
            self.assertEqual(layout.detect_threads, [owner, owner])
            self.assertTrue(all(t != owner for t in layout.prepare_threads))
            self.assertEqual(source.metadata()["layout_calls"], 2)
            self.assertEqual(source.max_pages, 1)
            self.assertEqual(source.phase_admitted, {"recognition": 4})
        finally:
            source.close()

    def test_empty_pages_drain_and_duplicate_names_fail(self):
        source = PaddleLayoutPageSource(client(), [(str(i), lambda: 0) for i in range(5)],
            layout_frontend=FakeLayout(0), on_page=lambda *args: None, page_window=1)
        try:
            while not source.closed:
                self.assertIsNone(source.pull(block=True))
            self.assertEqual(source.completed_pages, 5)
        finally:
            source.close()
        with self.assertRaises(ValueError):
            PaddleLayoutPageSource(client(), [("a", lambda: 0), ("a", lambda: 0)],
                layout_frontend=FakeLayout(), on_page=lambda *args: None)

    def test_detection_failure_propagates(self):
        layout = FakeLayout()
        def fail(_):
            raise RuntimeError("detector failed")
        layout.detect = fail
        source = PaddleLayoutPageSource(client(), [("a", lambda: 0)],
            layout_frontend=layout, on_page=lambda *args: None)
        try:
            with self.assertRaisesRegex(RuntimeError, "detector failed"):
                source.pull(block=True)
        finally:
            source.close()


if __name__ == "__main__":
    unittest.main()
