"""CPU control tests; the full decode-loop cases need torch, not an NPU."""
import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from streaming_decode import decode_mask_preflight, decode_step_state, run_decode_stream
from fixed_batch_engine import FixedBatchDecodeEngine


class RefillRegressionTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "CPU torch is required")
    def test_decode_mask_preflight_rejects_fully_masked_rows(self):
        report = decode_mask_preflight(torch.tensor([0, 7, 63]), 64)
        self.assertEqual(report["valid_key_counts"], [1, 8, 64])
        self.assertTrue(report["all_rows_have_finite_attention"])
        with self.assertRaisesRegex(ValueError, "invalid attention rows"):
            decode_mask_preflight(torch.tensor([-1, 7, 63]), 64)

    @unittest.skipIf(torch is None, "CPU torch is required")
    def test_decode_step_state_reports_mixed_boundaries(self):
        report = decode_step_state(
            torch.tensor([1278, 1279, 1280]),
            cache_length=4096,
            boundary_period=1280,
        )
        self.assertEqual(report["effective_lengths"], [1279, 1280, 1281])
        self.assertEqual(report["effective_length_residues"], [1279, 0, 1])
        self.assertEqual(report["boundary_rows"], [1])
        self.assertEqual(report["position_histogram"], {1278: 1, 1279: 1, 1280: 1})

    def test_legacy_window_does_not_lose_free_slots(self):
        tree = ast.parse(Path(__file__).with_name("fixed_batch_engine.py").read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_generate_continuous")
        refill = next(n for n in method.body if isinstance(n, ast.FunctionDef) and n.name == "admit_many")
        outer = ast.parse('''
def case():
    from collections import deque
    from types import SimpleNamespace
    from typing import Any, Sequence
    prefill_s = 0.0
    refill_count = immediate_completion_count = 0
    request_count = 10
    next_request = 1
    vision_ready = deque([(0, SimpleNamespace(max_new_tokens=2))])
    generated = [None] * 10
    outputs = [None] * 10
    slot_requests = [None] * 3
    slot_epochs = [0] * 3
    arena = None
    self = SimpleNamespace(eos_token_id=-1, _prefill_slots=lambda arena, entries: ({slot: {"token_id": 7} for slot, _, _ in entries}, 0, {}))
    def accumulate_prefill_metrics(metrics):
        pass
    def fill_vision_ready():
        nonlocal next_request
        if not vision_ready:
            while next_request < request_count:
                vision_ready.append((next_request, SimpleNamespace(max_new_tokens=2)))
                next_request += 1
    return admit_many([0, 1, 2]), slot_requests
''')
        outer.body[0].body.insert(-1, refill)
        namespace = {}
        exec(compile(ast.fix_missing_locations(outer), "refill_probe", "exec"), namespace)
        admitted, slots = namespace["case"]()
        self.assertEqual(sorted(admitted), [0, 1, 2])
        self.assertEqual(slots, [0, 1, 2])


class FakeSource:
    def __init__(self, requests, children=None):
        self.waiting = deque(requests)
        self.active = set()
        self.results = {}
        self.children = children or {}

    @property
    def closed(self):
        return not self.waiting and not self.active

    @property
    def upstream_exhausted(self):
        return not self.waiting and not any(i in self.children for i in self.active)

    def pull(self, *, block):
        if not self.waiting:
            return None
        i, first, limit = self.waiting.popleft()
        self.active.add(i)
        return i, SimpleNamespace(first=first, max_new_tokens=limit)

    def complete(self, i, ids):
        self.active.remove(i)
        if i in self.results:
            raise AssertionError("duplicate result")
        self.results[i] = ids
        self.waiting.extend(self.children.get(i, []))


class FakeEngine:
    batch_size = 3
    cache_length = 64
    eos_token_id = 99
    pad_token_id = 0
    vision_lookahead = 2
    packed_text_prefill_runtime = True

    def __init__(self):
        self.model = SimpleNamespace(device=torch.device("cpu"))
        self.compiled_decoder = SimpleNamespace(compiled_decode_for=lambda **kw: (self.decode, {}))
        self.key_cache = torch.zeros((self.batch_size, 1, 4, 1))
        self.value_cache = torch.zeros((self.batch_size, 1, 4, 1))
        self.arena = SimpleNamespace(
            flat_tensors=lambda: (self.key_cache, self.value_cache)
        )
        self.decode_inputs = []

    def _arena_for_batch(self):
        return self.arena

    def _duplicate_cache_row_(self, arena, source_slot, destination_slots):
        for tensor in arena.flat_tensors():
            for destination_slot in destination_slots:
                tensor[destination_slot : destination_slot + 1].copy_(
                    tensor[source_slot : source_slot + 1]
                )

    def _prepare_vision_window(self, window):
        return 0.0, {}

    def _prefill_slots(self, arena, entries):
        states = {}
        for slot, _i, req in entries:
            self.key_cache[slot].fill_(req.first)
            self.value_cache[slot].fill_(-req.first)
            states[slot] = {
                "token_id": req.first,
                "token": torch.tensor([[req.first]]),
                "cache_position": torch.tensor([1]),
                "rope_delta": torch.tensor([[req.first + 100]]),
            }
        return states, 0.0, {}

    def decode(self, next_token, cache_position, rope_delta, *cache_tensors):
        self.decode_inputs.append(
            tuple(
                tensor.clone()
                for tensor in (
                    next_token,
                    cache_position,
                    rope_delta,
                    *cache_tensors,
                )
            )
        )
        logits = torch.zeros((self.batch_size, 1, 128))
        for i, value in enumerate(next_token[:, 0].tolist()):
            token = 99 if value % 10 >= 3 else value + 1
            logits[i, 0, token] = 1
        return logits

    def _schedule_token_copy(self, candidate, **kwargs):
        return SimpleNamespace(host_tokens=candidate.reshape(-1).tolist(), **kwargs)

    def _wait_token_copy(self, pending):
        return pending.host_tokens, 0.0


@unittest.skipIf(torch is None, "CPU torch is required; run these tests in the validated 910B environment")
class DecodeTests(unittest.TestCase):
    def run_source(self, source, engine=None):
        sync_module = SimpleNamespace(maybe_sync_device=lambda device: None)
        with patch.dict(sys.modules, {"run_local_model_two_step_extract": sync_module}):
            return run_decode_stream(engine or FakeEngine(), source)

    def test_partial_initial_batch_duplicates_a_real_row_for_increfa(self):
        engine = FakeEngine()
        source = FakeSource([(0, 11, 10)])
        report = self.run_source(source, engine)
        next_token, cache_position, rope_delta, key_cache, value_cache = (
            engine.decode_inputs[0]
        )
        self.assertEqual(next_token[:, 0].tolist(), [11, 11, 11])
        self.assertEqual(cache_position.tolist(), [1, 1, 1])
        self.assertEqual(rope_delta[:, 0].tolist(), [111, 111, 111])
        self.assertTrue(torch.equal(key_cache[0:1], key_cache[1:2]))
        self.assertTrue(torch.equal(key_cache[0:1], key_cache[2:3]))
        self.assertTrue(torch.equal(value_cache[0:1], value_cache[1:2]))
        self.assertTrue(torch.equal(value_cache[0:1], value_cache[2:3]))
        self.assertEqual(
            report["inactive_filler_policy"],
            "duplicate_first_real_row_retain_controls",
        )
        self.assertEqual(report["initial_inactive_filler_rows"], 2)
        self.assertEqual(report["initial_filler_source_slot"], 0)

    def test_cache_row_filler_copies_every_kv_tensor(self):
        keys = (torch.arange(12).reshape(3, 1, 4, 1).float(),)
        values = (-torch.arange(12).reshape(3, 1, 4, 1).float(),)
        arena = SimpleNamespace(
            key_caches=keys,
            value_caches=values,
            flat_tensors=lambda: (*keys, *values),
        )
        FixedBatchDecodeEngine._duplicate_cache_row_(arena, 1, (0, 2))
        for tensor in arena.flat_tensors():
            self.assertTrue(torch.equal(tensor[0:1], tensor[1:2]))
            self.assertTrue(torch.equal(tensor[2:3], tensor[1:2]))

    def test_advance_filler_control_keeps_positions_uniform(self):
        engine = FakeEngine()
        engine.decode_filler_control = "advance"
        source = FakeSource([(0, 11, 10)])
        self.run_source(source, engine)
        self.assertGreaterEqual(len(engine.decode_inputs), 2)
        self.assertEqual(engine.decode_inputs[0][1].tolist(), [1, 1, 1])
        self.assertEqual(engine.decode_inputs[1][1].tolist(), [2, 2, 2])

    def test_retain_filler_control_creates_mixed_second_step(self):
        engine = FakeEngine()
        engine.decode_filler_control = "retain"
        source = FakeSource([(0, 11, 10)])
        self.run_source(source, engine)
        self.assertGreaterEqual(len(engine.decode_inputs), 2)
        self.assertEqual(engine.decode_inputs[0][1].tolist(), [1, 1, 1])
        self.assertEqual(engine.decode_inputs[1][1].tolist(), [2, 1, 1])

    def test_diagnostic_events_expose_second_step_state(self):
        engine = FakeEngine()
        engine.decode_diagnostic_steps = 2
        engine.decode_diagnostic_sync = False
        engine.decode_diagnostic_boundary_period = 2
        source = FakeSource([(0, 11, 10)])
        with patch("streaming_decode.log_phase") as phase:
            report = self.run_source(source, engine)
        states = [
            call.kwargs
            for call in phase.call_args_list
            if call.args[:2] == ("decode_step_state", "finish")
        ]
        self.assertEqual(len(states), 2)
        self.assertEqual(states[0]["step"], 0)
        self.assertEqual(states[0]["cache_positions"], [1, 1, 1])
        self.assertEqual(states[0]["boundary_rows"], [0, 1, 2])
        self.assertEqual(states[1]["step"], 1)
        self.assertEqual(states[1]["cache_positions"], [2, 1, 1])
        self.assertEqual(states[1]["effective_lengths"], [3, 2, 2])
        self.assertEqual(states[1]["boundary_rows"], [1, 2])
        self.assertEqual(report["decode_diagnostic_steps"], 2)
        self.assertEqual(report["decode_diagnostic_boundary_period"], 2)

    def test_windows_epochs_and_caps(self):
        requests = [(i, 1 + (i % 7)*10, 3 if i % 4 == 0 else 10) for i in range(100)]
        source = FakeSource(requests)
        report = self.run_source(source)
        self.assertEqual(report["request_count"], 100)
        self.assertEqual(report["idle_rows_with_ready_work"], 0)
        self.assertLessEqual(report["max_live_generation_requests"], 5)
        for i, first, limit in requests:
            expected = [first, first+1, first+2] + ([] if limit == 3 else [99])
            self.assertEqual(source.results[i], expected)

    def test_layout_completion_creates_new_work(self):
        source = FakeSource([(0, 1, 10)], {0: [(1, 11, 10), (2, 21, 10)]})
        report = self.run_source(source)
        self.assertEqual(set(source.results), {0, 1, 2})
        self.assertEqual(report["request_count"], 3)

    def test_immediate_eos_and_one_token_limit(self):
        source = FakeSource([(0, 99, 10), (1, 11, 1), (2, 21, 10)])
        report = self.run_source(source)
        self.assertEqual(source.results, {0: [99], 1: [11], 2: [21, 22, 23, 99]})
        self.assertEqual(report["immediate_completion_count"], 2)

    def test_all_immediate_and_empty(self):
        for requests in ([], [(0, 99, 10), (1, 1, 1)]):
            report = self.run_source(FakeSource(requests))
            self.assertEqual(report["graph_calls"], 0)
            self.assertEqual(report["request_count"], len(requests))


if __name__ == "__main__":
    unittest.main()
