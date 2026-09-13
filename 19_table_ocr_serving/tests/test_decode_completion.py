"""CPU scheduler checks with deterministic token streams, not NPU validation."""
from dataclasses import fields
import inspect
import queue
import threading
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch
import fixed_architecture_reference  # supplies CPU-only dependency stubs for runtime imports

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
sys.modules.setdefault('torch_npu', types.ModuleType('torch_npu'))

from p06_text_prefill_and_decode import LocalPaddleOCRVLStaticCache
from p02_serving_runtime import (
    ContinuousRecognizer, DecodeArena, DecodeCompletion, DecodeSlotState,
    DecodeRequest, RecognitionRequest,
)
from p02_serving_runtime import RecognitionResult
import p02_serving_runtime as continuous_decode


class DecodeCompletionTests(unittest.TestCase):
    def run_workload(self, sequences, *, capacity=512, limit=512, batch_size=1,
                     request_source=None, fail_preparation=()):
        """Use serve() and its real CPU queue/loop, with only model work faked."""
        def cache():
            shape = (1, 1, capacity, 1)
            return LocalPaddleOCRVLStaticCache((torch.zeros(shape),), (torch.zeros(shape),), capacity)

        engine = ContinuousRecognizer.__new__(ContinuousRecognizer)
        active_cache = cache()
        if batch_size > 1:
            active_cache = LocalPaddleOCRVLStaticCache(
                tuple(t.repeat(batch_size,1,1,1) for t in active_cache.key_caches),
                tuple(t.repeat(batch_size,1,1,1) for t in active_cache.value_caches), capacity)
        arena = DecodeArena(cache=active_cache, device=torch.device('cpu'),
                            batch_size=batch_size, eos_token_id=2)
        engine.device=arena.device
        engine.batch_size=batch_size
        engine.decode_arena=arena
        engine.eos_token_id=2
        engine.max_new_tokens=limit
        engine.completion_policy=None
        engine.progress=None
        engine.diagnostic_effective_length=None
        engine.diagnostic_request_id=None
        engine.copy_stream=None
        engine.host_token_ring=None
        engine.cpu_preprocess_max_pending=max(2,batch_size)
        engine.ready_buffer_capacity=batch_size
        engine.ready_buffer_low_watermark=max(1,batch_size//2)
        engine.prefill_cache_pool=types.SimpleNamespace(stats=lambda:dict(active_slots=0))
        releases=[]
        prefills=[]
        def prepare(request,submitted_at):
            if request.request_id in fail_preparation:
                raise ValueError('invalid image')
            return request
        def prefill(request,wait_s):
            # NPU prefill must never run while all decode slots are occupied.
            self.assertLess(arena.num_active,batch_size)
            prefills.append(request.request_id)
            token=sequences[request.request_id][0]
            return DecodeRequest(
                request_id=request.request_id, prompt=request.prompt, crop_size=(1,1),skip_special_tokens=True,
                cache=cache(),cache_lease=types.SimpleNamespace(release=lambda:releases.append(request.request_id)),
                rope_deltas=torch.zeros((1,1),dtype=torch.long),cache_position=torch.tensor([1]),
                first_token_tensor=torch.tensor([[token]]),first_token=token,prompt_length=1,
                projected_image_tokens=1,vision={},text_prefill={},cpu_timing=None,
                prefill_timing=None,request_started=0,prefill_finished=0)
        engine._prepare_cpu=prepare
        engine._prefill_for_decode=prefill
        # Capture the actual completion; detokenization is separately tested.
        engine._build_recognition_result=lambda completion,**kwargs:completion

        # Return the next token for the request currently occupying each slot.
        def decode(*args):
            tokens=[]
            for state in arena.slots:
                if state is None:
                    tokens.append([2]); continue
                sequence=sequences[state.ready.request_id]
                index=state.iterations_launched
                tokens.append([sequence[index] if index<len(sequence) else 2])
            return torch.tensor(tokens)
        engine.text_decode=types.SimpleNamespace(fn=decode)
        if request_source is None:
            request_source=QueueRequests()
            for request_id in sequences:
                request_source.queue.put(RecognitionRequest(request_id,b'image','Table Recognition:'))
            request_source.queue.put(None)
        completed=[]; errors=[]
        fake_npu = types.SimpleNamespace(npu=types.SimpleNamespace(synchronize=lambda device:None))
        with patch.object(continuous_decode, 'torch_npu', fake_npu), \
             patch.object(torch, 'npu', types.SimpleNamespace(current_stream=lambda device:
                 types.SimpleNamespace(synchronize=lambda:None)), create=True):
            summary = engine.serve(request_source,schedule_id='cpu-test',emit_result=completed.append,
                on_request_error=lambda request_id,error:errors.append((request_id,error)))
        self.assertEqual(sorted(releases),sorted(prefills))
        self.assertEqual(len(releases),len(set(releases)))
        self.assertEqual(arena.num_active,0)
        for completion in completed:
            self.assertIsNone(completion.ready.cache)
            self.assertIsNone(completion.ready.first_token_tensor)
        self.assertEqual(engine.output_tokens,sum(len(c.token_ids) for c in completed))
        self.assertEqual(summary.requests,len(completed))
        self.assertEqual(summary.raw_decode_token_slots,summary.effective_decode_tokens+
                         summary.idle_decode_token_slots+summary.lookahead_decode_token_slots)
        return completed,summary,errors

    def run_tokens(self, tokens, *, capacity=512, limit=512):
        completed,_,errors=self.run_workload({'test':tokens},capacity=capacity,limit=limit)
        self.assertEqual(errors,[])
        self.assertEqual(len(completed),1)
        return completed[0]

    def test_empty_shutdown(self):
        completed,summary,errors=self.run_workload({})
        self.assertEqual((completed,summary.graph_calls,errors),([],0,[]))

    def test_first_token_eos(self):
        completed,summary,_=self.run_workload({'first':[2]})
        self.assertEqual(completed[0].token_ids,[2])
        self.assertEqual(summary.graph_calls,0)
        self.assertEqual(summary.prefill_only_completions,1)

    def test_simultaneous_completions_and_slot_reuse(self):
        sequences={str(i):[10+i,20+i,30+i,2] for i in range(9)}
        completed,summary,errors=self.run_workload(sequences,batch_size=3)
        self.assertEqual(errors,[])
        self.assertGreater(summary.hot_swap_admissions,0)
        self.assertEqual({c.ready.request_id:c.token_ids for c in completed},sequences)

    def test_preparation_failure_does_not_stop_other_requests(self):
        completed,_,errors=self.run_workload({'bad':[7,2],'good':[8,2]},fail_preparation=('bad',))
        self.assertEqual([c.ready.request_id for c in completed],['good'])
        self.assertEqual(errors[0][0],'bad')

    def test_idle_does_not_mean_shutdown(self):
        source=QueueRequests()
        producer_errors=[]
        def produce():
            try:
                # Wait until serve actually asks for input before supplying it.
                self.assertTrue(source.waiting.wait(5))
                source.queue.put(RecognitionRequest('later',b'image','Table Recognition:'))
                source.queue.put(None)
            except BaseException as exc:
                producer_errors.append(exc); source.queue.put(None)
        producer=threading.Thread(target=produce)
        producer.start()
        completed,_,_=self.run_workload({'later':[9,2]},request_source=source)
        producer.join(5)
        self.assertEqual(producer_errors,[])
        self.assertEqual(completed[0].token_ids,[9,2])

    def test_repeated_tokens_survive_until_eos(self):
        for period in (1, 3, 32):
            with self.subTest(period=period):
                tokens = list(range(10, 10 + period)) * max(6, 192 // period) + [2]
                completion = self.run_tokens(tokens)
                self.assertEqual(completion.stop_reason, 'eos')
                self.assertEqual(completion.token_ids, tokens)

    def test_repeated_tokens_survive_until_output_limit(self):
        completion = self.run_tokens([7] * 400, limit=200)
        self.assertEqual(completion.stop_reason, 'length')
        self.assertEqual(completion.token_ids, [7] * 200)

    def test_repeated_tokens_survive_until_kv_cap(self):
        completion = self.run_tokens([7] * 400, capacity=200)
        self.assertEqual(completion.stop_reason, 'kv_cache_full')
        self.assertEqual(completion.token_ids, [7] * 200)

    def test_repetition_option_state_and_result_fields_removed(self):
        self.assertNotIn('stop_repetitions', inspect.signature(ContinuousRecognizer).parameters)
        for cls in (DecodeSlotState, DecodeCompletion, RecognitionResult):
            self.assertFalse(any('repetition' in f.name for f in fields(cls)))
        self.assertFalse((EXPERIMENT / '_support/serving/repetition.py').exists())


class QueueRequests:
    """Same empty-versus-closed contract as the HTTP inference worker."""
    def __init__(self):
        self.queue=queue.Queue()
        self.closed=False
        self.waiting=threading.Event()

    def pull(self,*,block):
        if self.closed: return None
        self.waiting.set()
        try:
            request=self.queue.get(timeout=5) if block else self.queue.get_nowait()
        except queue.Empty:
            return None
        if request is None: self.closed=True
        return request


if __name__ == '__main__':
    unittest.main()
