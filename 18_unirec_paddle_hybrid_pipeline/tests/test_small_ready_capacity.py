from collections import deque
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.base import Adapter
from coordinator import Coordinator
from test_page_prefill import Pages


class SmallReady(Adapter):
    def __init__(self):
        super().__init__(4, None, ready_capacity=2)
        self.ready = deque()
        self.slots = []
        self.finished = 0
        self.closed = False
        self.upstream = True
        self.peak_ready = self.peak_active = 0
        self.admission_only = 0

    @property
    def ready_count(self):
        return len(self.ready)

    def set_upstream(self, pending, *, closed=False):
        self.upstream = pending
        self.closed = closed and not pending

    def prefill(self):
        self.ready.extend(self.take_prefill_requests())
        self.peak_ready = max(self.peak_ready, len(self.ready))

    def advance(self, count):
        while self.ready and len(self.slots) < self.capacity:
            self.slots.append(self.ready.popleft())
        self.active = len(self.slots)
        self.peak_active = max(self.peak_active, self.active)
        if self.active < self.capacity and self.upstream:
            self.admission_only += 1
            return
        self.slots = [n - 1 for n in self.slots]
        self.finished += self.slots.count(0)
        self.slots = [n for n in self.slots if n]
        self.active = len(self.slots)
        self.graph_calls += 1
        self.done = self.closed and not self.slots and not self.ready


class SmallReadyTests(unittest.TestCase):
    def test_half_reservoir_fills_full_arena_and_drains(self):
        engines = {n: SmallReady() for n in ('unirec', 'paddle')}
        pages = Pages([{'unirec': [1, 3, 2, 4] * 9, 'paddle': [1, 4, 2] * 7},
                       {'unirec': [2], 'paddle': [3]}])
        Coordinator(engines, pages).run()
        for n, expected in (('unirec', 37), ('paddle', 22)):
            e = engines[n]
            self.assertEqual(e.finished, expected)
            self.assertEqual(e.peak_ready, 2)
            self.assertEqual(e.peak_active, 4)
            self.assertGreater(e.admission_only, 0)

    def test_full_decode_precedes_other_ready_admission(self):
        u, p = SmallReady(), SmallReady()
        u.active = 4
        p.ready.extend([1, 1])
        c = Coordinator({'unirec': u, 'paddle': p}, Pages([]))
        self.assertEqual(c.action(), ('unirec', 'decode'))
        u.active = 0
        self.assertEqual(c.action(), ('paddle', 'decode'))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'torch unavailable in authoring interpreter')
class PaddleAdmissionTests(unittest.TestCase):
    def test_actual_scheduler_admits_two_halves_before_first_decode(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'09_persistent_page_engine'))
        from paddleocr_vl.model.text_decode import LocalPaddleOCRVLStaticCache
        from paddleocr_vl.serving.continuous_decode import ContinuousDecodeScheduler, DecodeArena, ReadyDecodeRequest
        from adapters.paddle import ReadySource
        def cache(batch):
            return LocalPaddleOCRVLStaticCache(key_caches=(torch.zeros((batch,1,32,1)),),
                value_caches=(torch.zeros((batch,1,32,1)),), cache_length=32)
        arena = DecodeArena(cache=cache(4), device=torch.device('cpu'), batch_size=4,
                            eos_token_id=9, decode_token_id_map=torch.arange(10))
        source = ReadySource()
        def enqueue(indices):
            for i in indices:
                source.items.append(ReadyDecodeRequest(request_id=str(i),payload=None,cache=cache(1),
                    rope_deltas=torch.zeros((1,1),dtype=torch.int64), cache_position=torch.ones(1,dtype=torch.int64),
                    first_token_tensor=torch.tensor([[1]]), first_token=1, prompt_length=1))
        active_counts = []
        def decode(tokens, *args):
            active_counts.append(arena.num_active)
            return torch.where(tokens >= 3, torch.full_like(tokens,9), tokens+1)
        completed = []
        scheduler = ContinuousDecodeScheduler(arena=arena,decode_fn=decode,max_new_tokens=20)
        steps = scheduler.iter_run_stream(source, on_completion=completed.append,
            ready_buffer_capacity=2, ready_buffer_low_watermark=1, cooperative_refill=True)
        self.assertEqual(next(steps), {'active':0,'graph_calls':0})
        enqueue(range(2))
        self.assertEqual(next(steps), {'active':2,'graph_calls':0})
        enqueue(range(2,4)); source.closed = True
        self.assertEqual(next(steps), {'active':4,'graph_calls':0})
        for _ in range(30):
            try:
                next(steps)
            except StopIteration:
                break
        else:
            self.fail('scheduler did not drain')
        self.assertEqual(active_counts[0],4)
        self.assertEqual({c.ready.request_id for c in completed},{'0','1','2','3'})
