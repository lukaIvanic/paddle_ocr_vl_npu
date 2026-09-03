from concurrent.futures import Future
from types import SimpleNamespace
import unittest

from streaming_pipeline import BoundedWriter, MinerUPageSource, PageInbox


class Block(dict):
    def __setattr__(self, key, value):
        self[key] = value


class Helper:
    enable_cross_page_table_merge = False

    def prepare_for_layout(self, image):
        return image

    def parse_layout_output(self, text):
        return [Block(type="text", bbox=[0, 0, 1, 1]), Block(type="text", bbox=[0, 0, 1, 1])]

    def prepare_for_extract(self, image, blocks):
        indices = list(range(image))
        return [image] * image, ["ocr"] * image, [None] * image, indices

    def post_process(self, blocks):
        return blocks


class Adapter:
    skip_token_ids = set()

    def __init__(self):
        self.processor = SimpleNamespace(
            apply_chat_template=lambda *a, **k: "prompt",
            batch_decode=lambda rows, **k: [str(row[0]) for row in rows])

    def build_messages(self, *args, **kwargs):
        return []

    def _prepare_cpu_inputs(self, *args):
        return None, None, None, 0, 0

    def _finish_generation(self, *args):
        return SimpleNamespace(max_new_tokens=5)


def client():
    return SimpleNamespace(helper=Helper(), client=Adapter(), prompts={"[layout]": "layout"}, sampling_params={})


class PipelineTests(unittest.TestCase):
    def test_live_input_survives_idle_and_drains_on_close(self):
        inbox = PageInbox(capacity=2)
        completed = []
        source = MinerUPageSource(client(), inbox, on_page=lambda name, blocks: completed.append(name), page_window=1)
        try:
            self.assertIsNone(source.pull(block=True))
            self.assertFalse(source.closed)
            inbox.submit("first", lambda: 0)
            index, _ = source.pull(block=True)
            source.complete(index, [9])
            self.assertIsNone(source.pull(block=True))
            self.assertFalse(source.closed)
            inbox.submit("second", lambda: 0)
            inbox.close_input()
            index, _ = source.pull(block=True)
            source.complete(index, [9])
            self.assertIsNone(source.pull(block=True))
            self.assertTrue(source.closed)
            self.assertEqual(completed, ["first", "second"])
        finally:
            source.close()

    def test_zero_crop_pages_can_roll_past_window(self):
        completed = []
        source = MinerUPageSource(client(), [(str(i), lambda: 0) for i in range(10)],
                                 on_page=lambda name, blocks: completed.append(name), page_window=1)
        try:
            while not source.closed:
                item = source.pull(block=True)
                if item is not None:
                    source.complete(item[0], [9])
            self.assertEqual(len(completed), 10)
        finally:
            source.close()

    def test_rolling_pages_and_out_of_order_completions(self):
        results = []
        source = MinerUPageSource(client(), [(str(i), lambda: 2) for i in range(10)],
                                 on_page=lambda name, blocks: results.append((name, blocks)),
                                 page_window=2, prepare_depth=3)
        try:
            waiting = []
            while not source.closed:
                item = source.pull(block=True)
                if item is not None:
                    waiting.append(item[0])
                if waiting and (item is None or len(waiting) == 3):
                    source.complete(waiting.pop(), [7, 9])
            self.assertEqual(len(results), 10)
            self.assertEqual(len({name for name, _ in results}), 10)
            self.assertLessEqual(source.max_pages, 2)
            self.assertLessEqual(source.max_prepare, 3)
            self.assertFalse(source.inflight)
            self.assertTrue(source.upstream_exhausted)
            self.assertTrue(all(block["content"] == "7" for _, blocks in results for block in blocks))
        finally:
            source.close()

    def test_temporary_empty_is_not_eof(self):
        source = MinerUPageSource(client(), [("p", lambda: 2)], on_page=lambda *x: None, page_window=1)
        try:
            item = source.pull(block=True)
            self.assertIsNotNone(item)
            self.assertIsNone(source.pull(block=True))
            self.assertFalse(source.closed)
            self.assertFalse(source.upstream_exhausted)
        finally:
            source.close()

    def test_empty_page_completes(self):
        results = []
        source = MinerUPageSource(client(), [("p", lambda: 0)], on_page=lambda *x: results.append(x))
        try:
            index, _ = source.pull(block=True)
            source.complete(index, [9])
            self.assertIsNone(source.pull(block=True))
            self.assertTrue(source.closed)
            self.assertEqual(len(results), 1)
        finally:
            source.close()

    def test_producer_error_propagates(self):
        def fail():
            raise ValueError("loader failed")
        source = MinerUPageSource(client(), [("p", fail)], on_page=lambda *x: None)
        try:
            with self.assertRaisesRegex(ValueError, "loader failed"):
                source.pull(block=True)
        finally:
            source.close()

    def test_writer_errors_propagate(self):
        def fail(*args):
            raise OSError("disk full")
        writer = BoundedWriter(fail)
        writer.submit("p")
        with self.assertRaisesRegex(OSError, "disk full"):
            writer.close()

    def test_writer_preserves_submission_order(self):
        output = []
        writer = BoundedWriter(output.append, capacity=2)
        for i in range(20):
            writer.submit(i)
        writer.close()
        self.assertEqual(output, list(range(20)))
        self.assertLessEqual(writer.max_pending, 2)


if __name__ == "__main__":
    unittest.main()
