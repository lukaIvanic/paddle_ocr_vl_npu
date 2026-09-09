from pathlib import Path
import sys
from threading import Event
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamed_execution import StreamedExecution


class StreamTests(unittest.TestCase):
    def test_early_publication_allows_stage_overlap(self):
        executor = StreamedExecution()
        text_started = Event()
        def vision():
            executor.publish("vision", "page-ready")
            if not text_started.wait(2):
                raise RuntimeError("text was serialized behind whole vision window")
            return "vision-finished"
        def text():
            text_started.set()
            return "text-finished"
        try:
            executor.submit("vision", vision)
            self.assertEqual(executor.receive(), ("publish", "vision", "page-ready"))
            self.assertIn("vision", executor.running)
            executor.submit("text", text)
            results = {executor.receive()[1], executor.receive()[1]}
            self.assertEqual(results, {"vision", "text"})
            self.assertFalse(executor.running)
            self.assertGreater(executor.summary()["multi_stage_host_overlap_s"], 0)
        finally:
            executor.close()

    def test_one_operation_per_stage_and_direct_failure(self):
        executor = StreamedExecution()
        release = Event()
        try:
            executor.submit("decode", lambda: release.wait(2))
            with self.assertRaisesRegex(RuntimeError, 'already running'):
                executor.submit("decode", lambda: None)
            release.set()
            executor.receive()
            def fail():
                raise ValueError("bad stage")
            executor.submit("text", fail)
            with self.assertRaisesRegex(ValueError, "bad stage"):
                executor.receive()
        finally:
            release.set()
            executor.close()

    def test_streamed_owner_selection_preserves_paddle_ties(self):
        from coordinator import Coordinator
        from test_coordinator import Engine, Pages
        u, p = Engine(), Engine()
        u.streamed = True
        u.occupied = p.occupied = 2
        owner = Coordinator(dict(unirec=u, paddle=p), Pages([]))
        owner.last_model = 'paddle'
        self.assertEqual(owner.action(), ('unirec', 'stream'))
        owner.last_model = 'unirec'
        self.assertEqual(owner.action(), ('paddle', 'decode'))


if __name__ == '__main__':
    unittest.main()
