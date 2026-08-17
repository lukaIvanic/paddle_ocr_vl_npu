#!/usr/bin/env python3

from __future__ import annotations

import unittest

from modeling_qwen3_tp2 import Qwen3TPConfig, shard_bounds


class Qwen3TPContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Qwen3TPConfig(
            vocab_size=151936,
            hidden_size=5120,
            intermediate_size=25600,
            num_hidden_layers=64,
            num_attention_heads=64,
            num_key_value_heads=8,
            head_dim=128,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000.0,
            max_position_embeddings=40_960,
            tie_word_embeddings=False,
        )

    def test_tp2_local_dimensions(self) -> None:
        self.config.validate_tp(2)
        self.assertEqual(self.config.num_attention_heads // 2, 32)
        self.assertEqual(self.config.num_key_value_heads // 2, 4)
        self.assertEqual(self.config.intermediate_size // 2, 12800)
        self.assertEqual(self.config.vocab_size // 2, 75968)

    def test_shards_cover_dimension_without_overlap(self) -> None:
        self.assertEqual(shard_bounds(151936, 0, 2), (0, 75968))
        self.assertEqual(shard_bounds(151936, 1, 2), (75968, 151936))

    def test_layer_override_keeps_architecture(self) -> None:
        smoke = self.config.with_num_hidden_layers(1)
        self.assertEqual(smoke.num_hidden_layers, 1)
        self.assertEqual(smoke.hidden_size, self.config.hidden_size)


if __name__ == "__main__":
    unittest.main()
