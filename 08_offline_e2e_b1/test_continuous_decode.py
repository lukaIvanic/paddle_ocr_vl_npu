"""CPU control-plane tests for the continuous decode scheduler."""

from __future__ import annotations

import unittest

import torch

from continuous_decode import (
    ContinuousDecodeScheduler,
    DecodeArena,
    ReadyDecodeRequest,
)
from local_modeling_paddleocr_vl import LocalPaddleOCRVLStaticCache


def make_cache(batch_size: int, fill: float) -> LocalPaddleOCRVLStaticCache:
    key = torch.full((batch_size, 1, 8, 1), fill, dtype=torch.float32)
    value = torch.full((batch_size, 1, 8, 1), fill + 0.5, dtype=torch.float32)
    return LocalPaddleOCRVLStaticCache((key,), (value,), 8)


def make_ready(request_id: str, first_token: int, fill: float) -> ReadyDecodeRequest:
    return ReadyDecodeRequest(
        request_id=request_id,
        payload=request_id,
        cache=make_cache(1, fill),
        rope_deltas=torch.zeros((1, 1), dtype=torch.int64),
        cache_position=torch.tensor([3], dtype=torch.int64),
        first_token_tensor=torch.tensor([[first_token]], dtype=torch.int64),
        first_token=first_token,
        prompt_length=3,
    )


class ContinuousDecodeSchedulerTest(unittest.TestCase):
    def test_reuses_finished_slot_and_discards_delayed_lookahead(self) -> None:
        transitions = {
            10: 11,
            11: 2,
            20: 2,
            30: 31,
            31: 32,
            32: 2,
            2: 99,
        }

        def decode_fn(input_ids, _positions, _rope, *_cache):
            batch_size = int(input_ids.shape[0])
            logits = torch.full((batch_size, 1, 128), -1.0)
            for row, token_id in enumerate(input_ids.reshape(-1).tolist()):
                logits[row, 0, transitions[int(token_id)]] = 1.0
            return logits

        arena = DecodeArena(
            cache=make_cache(2, 0.0),
            device=torch.device("cpu"),
            batch_size=2,
            eos_token_id=2,
        )
        scheduler = ContinuousDecodeScheduler(
            arena=arena,
            decode_fn=decode_fn,
            max_new_tokens=8,
        )
        result = scheduler.run(
            [
                make_ready("a", 10, 1.0),
                make_ready("b", 20, 2.0),
                make_ready("c", 30, 3.0),
            ]
        )

        self.assertEqual(
            [completion.token_ids for completion in result.completions],
            [[10, 11, 2], [20, 2], [30, 31, 32, 2]],
        )
        self.assertEqual([completion.stop_reason for completion in result.completions], ["eos"] * 3)
        self.assertEqual(result.initial_admissions, 2)
        self.assertEqual(result.hot_swap_admissions, 1)
        self.assertEqual(result.graph_calls, 6)
        self.assertEqual(result.raw_decode_token_slots, 12)
        self.assertEqual(result.effective_decode_tokens, 6)
        self.assertEqual(result.active_decode_token_slots, 9)
        self.assertEqual(result.lookahead_decode_token_slots, 3)
        self.assertEqual(result.idle_decode_token_slots, 3)
        self.assertEqual(result.raw_decode_token_slots, result.effective_decode_tokens + result.lookahead_decode_token_slots + result.idle_decode_token_slots)
        self.assertTrue(all(slot is None for slot in arena.slots))

    def test_prefill_only_requests_do_not_launch_decode(self) -> None:
        arena = DecodeArena(
            cache=make_cache(2, 0.0),
            device=torch.device("cpu"),
            batch_size=2,
            eos_token_id=2,
        )
        scheduler = ContinuousDecodeScheduler(
            arena=arena,
            decode_fn=lambda *_args: self.fail("decode must not run"),
            max_new_tokens=1,
        )
        result = scheduler.run(
            [
                make_ready("already-eos", 2, 1.0),
                make_ready("length-one", 10, 2.0),
            ]
        )

        self.assertEqual(result.graph_calls, 0)
        self.assertEqual(result.prefill_only_completions, 2)
        self.assertEqual(
            [completion.stop_reason for completion in result.completions],
            ["eos", "length"],
        )
        self.assertEqual(result.raw_decode_token_slots, 0)

    def test_stream_keeps_slots_filled_across_page_ids(self) -> None:
        transitions = {
            10: 11,
            11: 12,
            12: 13,
            13: 14,
            14: 15,
            15: 2,
            20: 2,
            30: 2,
            40: 2,
            50: 2,
            2: 99,
        }

        def decode_fn(input_ids, _positions, _rope, *_cache):
            batch_size = int(input_ids.shape[0])
            logits = torch.full((batch_size, 1, 128), -1.0)
            for row, token_id in enumerate(input_ids.reshape(-1).tolist()):
                logits[row, 0, transitions[int(token_id)]] = 1.0
            return logits

        arena = DecodeArena(
            cache=make_cache(2, 0.0),
            device=torch.device("cpu"),
            batch_size=2,
            eos_token_id=2,
        )
        scheduler = ContinuousDecodeScheduler(
            arena=arena,
            decode_fn=decode_fn,
            max_new_tokens=8,
        )
        completed: list[str] = []
        source = (
            make_ready(request_id, first_token, float(index + 1))
            for index, (request_id, first_token) in enumerate(
                [
                    ("page0-long", 10),
                    ("page0-short", 20),
                    ("page1-a", 30),
                    ("page1-b", 40),
                    ("page1-c", 50),
                ]
            )
        )
        result = scheduler.run_stream(
            source,
            on_completion=lambda completion: completed.append(
                completion.ready.request_id
            ),
            ready_buffer_capacity=2,
        )

        self.assertEqual(result.submitted_requests, 5)
        self.assertEqual(result.ready_buffer_capacity, 2)
        self.assertLessEqual(result.max_ready_queue_depth, 2)
        self.assertEqual(
            [completion.ready.request_id for completion in result.completions],
            ["page0-long", "page0-short", "page1-a", "page1-b", "page1-c"],
        )
        self.assertEqual(completed[0], "page0-short")
        self.assertIn("page1-a", completed[:-1])
        self.assertGreaterEqual(result.active_decode_token_slots / result.raw_decode_token_slots, 0.9)
        self.assertEqual(
            result.raw_decode_token_slots,
            result.effective_decode_tokens
            + result.lookahead_decode_token_slots
            + result.idle_decode_token_slots,
        )


if __name__ == "__main__":
    unittest.main()
