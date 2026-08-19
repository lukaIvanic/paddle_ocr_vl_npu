#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from absorbed_mla import (  # noqa: E402
    absorb_kv_b_weight,
    materialize_absorbed_kv,
)


class AbsorbedMLATest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(52)
        self.heads = 4
        self.nope = 3
        self.rope = 2
        self.value = 5
        self.latent = 6
        self.cache_length = 7
        self.kv_b_weight = torch.randn(
            self.heads * (self.nope + self.value), self.latent
        )
        self.w_uk_t, self.w_uv = absorb_kv_b_weight(
            self.kv_b_weight,
            local_heads=self.heads,
            qk_nope_head_dim=self.nope,
            v_head_dim=self.value,
            kv_lora_rank=self.latent,
        )

    def test_materialized_kv_matches_original_projection(self) -> None:
        latent_cache = torch.randn(1, self.cache_length, self.latent)
        rope_cache = torch.randn(1, self.cache_length, self.rope)
        key, value = materialize_absorbed_kv(
            latent_cache,
            rope_cache,
            self.w_uk_t,
            self.w_uv,
            used_length=self.cache_length,
        )

        projected = torch.nn.functional.linear(
            latent_cache[0], self.kv_b_weight
        ).view(self.cache_length, self.heads, self.nope + self.value)
        expected_nope, expected_value = projected.split(
            [self.nope, self.value], dim=-1
        )
        expected_key = torch.cat(
            (
                expected_nope.permute(1, 0, 2),
                rope_cache[0].unsqueeze(0).expand(self.heads, -1, -1),
            ),
            dim=-1,
        ).unsqueeze(0)
        expected_value = expected_value.permute(1, 0, 2).unsqueeze(0)

        torch.testing.assert_close(key, expected_key)
        torch.testing.assert_close(value, expected_value)

    def test_rejects_wrong_weight_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected KV-B weight"):
            absorb_kv_b_weight(
                self.kv_b_weight[:-1],
                local_heads=self.heads,
                qk_nope_head_dim=self.nope,
                v_head_dim=self.value,
                kv_lora_rank=self.latent,
            )


if __name__ == "__main__":
    unittest.main()
