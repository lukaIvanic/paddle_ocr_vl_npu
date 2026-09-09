from pathlib import Path
import sys
from threading import Event
import unittest
import importlib.util
from collections import deque
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamed_execution import StreamedExecution


class StreamTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch')
    def test_direct_packed_export_preserves_kv_segments_and_token_accounting(self):
        import torch
        from adapters.unirec import UniRecAdapter
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'12_unirec_0_1b_inference'))
        keys = tuple((torch.full((1, 2, n, 3), float(n)),) for n in (2, 4))
        vals = tuple((torch.full((1, 2, n, 3), float(n+1)),) for n in (2, 4))
        packed = SimpleNamespace(segment_lengths=(2,4), cross_key_cache=keys, cross_value_cache=vals,
                                 real_source_tokens=6, physical_source_tokens=8)
        runtime = SimpleNamespace(run=lambda **kwargs: packed, metadata={'execution':'compiled_packed_s8'})
        adapter = UniRecAdapter.__new__(UniRecAdapter)
        adapter.runner = SimpleNamespace(_get_compiled_packed_text_prefill_runtime=lambda:runtime)
        out = adapter.export_prefill_group([(None,{'id':0}),(None,{'id':1})])
        self.assertTrue(torch.equal(out[0].packed_cross_kv, torch.stack((*keys[0], *vals[0]))))
        self.assertTrue(torch.equal(out[1].packed_cross_kv, torch.stack((*keys[1], *vals[1]))))
        self.assertEqual([x.actual_cross_attention_length for x in out],[2,4])
        self.assertEqual([x.text_prefill_real_source_tokens for x in out],[2,4])
        self.assertEqual([x.text_prefill_physical_source_tokens for x in out],[2,6])
        self.assertEqual([x.prep for x in out],[{'id':0},{'id':1}])

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

    def test_cross_page_supply_and_tail_use_existing_capacity(self):
        from adapters.unirec_streamed import StreamedUniRecAdapter
        adapter = StreamedUniRecAdapter.__new__(StreamedUniRecAdapter)
        adapter.chunks, adapter.text_groups = deque(), deque()
        adapter.buffer_capacity = 8
        adapter.external_pending = True
        adapter.pending = deque([deque(range(3)), deque(range(3,7))])
        adapter.cpu = SimpleNamespace(ready=lambda rows: True)
        self.assertFalse(adapter.prefill_available)  # Request another page.
        adapter.pending.append(deque([7]))
        self.assertTrue(adapter.prefill_available)
        self.assertEqual(adapter.planned_prefill_requests(), list(range(8)))
        adapter.cpu.ready = lambda rows: False
        self.assertFalse(adapter.prefill_available)  # Do not consume CPU-in-flight.
        adapter.cpu.ready = lambda rows: True
        adapter.pending.pop()
        adapter.set_supply(False, True)
        self.assertTrue(adapter.prefill_available)  # No hypothetical future input.

    def test_staged_work_prevents_premature_source_close(self):
        from adapters.unirec_streamed import StreamedUniRecAdapter
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'12_unirec_0_1b_inference'))
        from persistent_ready_queue import PersistentReadyQueue
        adapter = StreamedUniRecAdapter.__new__(StreamedUniRecAdapter)
        adapter.source = PersistentReadyQueue(maxsize=4)
        adapter.source.register_upstream()
        adapter.pending = deque()
        adapter.buffer_rows = 1
        adapter.set_supply(False, True)
        adapter.set_upstream(False, closed=True)
        self.assertEqual(adapter.source.upstream_pending, 1)
        self.assertFalse(adapter.source.close_requested)
        adapter.buffer_rows = 0
        adapter.set_upstream(False, closed=True)
        self.assertTrue(adapter.source.close_requested)


if __name__ == '__main__':
    unittest.main()
