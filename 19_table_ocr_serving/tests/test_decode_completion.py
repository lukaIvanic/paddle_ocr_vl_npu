"""CPU scheduler checks with deterministic token streams, not NPU validation."""
from dataclasses import fields
import inspect
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
    ContinuousDecodeScheduler, DecodeArena, DecodeCompletion, DecodeSlotState,
    ReadyDecodeRequest,
)
from p02_serving_runtime import RecognitionResult
import p02_serving_runtime as continuous_decode


class DecodeCompletionTests(unittest.TestCase):
    def run_tokens(self, tokens, *, capacity=512, limit=512):
        def cache():
            shape = (1, 1, capacity, 1)
            return LocalPaddleOCRVLStaticCache((torch.zeros(shape),), (torch.zeros(shape),), capacity)

        # Return EOS after the supplied tokens, including any completion look-ahead.
        cursor = 0
        def decode(*args):
            nonlocal cursor
            cursor += 1
            return torch.tensor([[tokens[cursor] if cursor < len(tokens) else 2]])

        arena = DecodeArena(cache=cache(), device=torch.device('cpu'),
                            batch_size=1, eos_token_id=2, decode_device_timing=False)
        scheduler = ContinuousDecodeScheduler(arena=arena, decode_fn=decode, max_new_tokens=limit)
        request = ReadyDecodeRequest(
            request_id='test', payload=None, cache=cache(), rope_deltas=torch.zeros((1, 1), dtype=torch.long),
            cache_position=torch.tensor([1]), first_token_tensor=torch.tensor([[tokens[0]]]),
            first_token=tokens[0], prompt_length=1,
        )
        fake_npu = types.SimpleNamespace(npu=types.SimpleNamespace(synchronize=lambda device:None))
        with patch.object(continuous_decode, 'torch_npu', fake_npu), \
             patch.object(torch, 'npu', types.SimpleNamespace(current_stream=lambda device:
                 types.SimpleNamespace(synchronize=lambda:None)), create=True):
            result = scheduler.run([request])
        self.assertEqual(len(result.completions), 1)
        return result.completions[0]

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
        self.assertNotIn('stop_repetitions', inspect.signature(ContinuousDecodeScheduler).parameters)
        for cls in (DecodeSlotState, DecodeCompletion, RecognitionResult):
            self.assertFalse(any('repetition' in f.name for f in fields(cls)))
        self.assertFalse((EXPERIMENT / '_support/serving/repetition.py').exists())


if __name__ == '__main__':
    unittest.main()
