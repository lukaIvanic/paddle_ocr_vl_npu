#!/usr/bin/env python3

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

if "safetensors" not in sys.modules:
    safetensors_stub = types.ModuleType("safetensors")
    safetensors_stub.safe_open = None
    sys.modules["safetensors"] = safetensors_stub

from modeling_glm52_layer import GLM52Config  # noqa: E402
from modeling_glm52_tp1 import (  # noqa: E402
    GLM52OptimizedTP1Stack,
    expected_indexer_type,
)


def test_config() -> GLM52Config:
    return GLM52Config(
        hidden_size=6144,
        num_hidden_layers=78,
        num_attention_heads=64,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        moe_intermediate_size=2048,
        num_experts=256,
        top_k=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        first_k_dense_replace=3,
        index_topk_freq=4,
        index_skip_topk_offset=3,
    )


class FakeLayer(torch.nn.Module):
    def __init__(self, layer_index: int, *, full: bool, config: GLM52Config):
        super().__init__()
        self.layer_index = layer_index
        self.indexer = object() if full else None
        self.config = config

    def forward_decode_with_topk(
        self,
        hidden_states,
        cache_position,
        primary_cache,
        secondary_cache,
        index_cache,
        shared_topk,
    ):
        del cache_position, primary_cache, secondary_cache, index_cache
        if self.indexer is not None:
            shared_topk = torch.full_like(shared_topk, self.layer_index)
        return hidden_states + 1, shared_topk


class OptimizedTP1StackTest(unittest.TestCase):
    def test_full_indexer_schedule_for_all_78_layers(self) -> None:
        config = test_config()
        full = [
            index
            for index in range(config.num_hidden_layers)
            if expected_indexer_type(config, index) == "full"
        ]
        self.assertEqual(full[:5], [0, 1, 2, 6, 10])
        self.assertEqual(full[-1], 74)
        self.assertEqual(len(full), 21)

    def test_layers_2_10_reuse_the_previous_full_selection(self) -> None:
        config = test_config()
        layers = [
            FakeLayer(
                index,
                full=expected_indexer_type(config, index) == "full",
                config=config,
            )
            for index in range(2, 11)
        ]
        stack = GLM52OptimizedTP1Stack(
            layers,
            first_layer=2,
            last_layer=10,
            cache_length=16,
        )
        self.assertEqual(stack.full_indexer_layers, [2, 6, 10])
        self.assertEqual(stack.shared_indexer_layers, [3, 4, 5, 7, 8, 9])
        dummy_caches = tuple(
            tuple(torch.zeros(1) for _ in layers) for _ in range(3)
        )
        output, topk = stack.forward_decode(
            torch.zeros(1),
            torch.zeros(1, dtype=torch.int64),
            *dummy_caches,
            stack.initial_topk(device=torch.device("cpu")),
        )
        self.assertEqual(float(output.item()), 9.0)
        self.assertTrue(torch.equal(topk, torch.full_like(topk, 10)))


if __name__ == "__main__":
    unittest.main()
