#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from absorbed_mla import (  # noqa: E402
    absorb_kv_b_weight,
    manual_absorbed_attention,
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

    def test_absorbed_attention_matches_decompressed_attention(self) -> None:
        latent_cache = torch.randn(1, self.cache_length, self.latent)
        rope_cache = torch.randn(1, self.cache_length, self.rope)
        query_nope = torch.randn(1, 1, self.heads, self.nope)
        query_rope = torch.randn(1, 1, self.heads, self.rope)
        selected = torch.tensor([0, 2, 3, 5, 6], dtype=torch.int64)
        position = torch.tensor([5], dtype=torch.int64)
        scale = (self.nope + self.rope) ** -0.5

        absorbed = manual_absorbed_attention(
            query_nope,
            query_rope,
            latent_cache,
            rope_cache,
            self.w_uk_t,
            self.w_uv,
            selected,
            position,
            scale=scale,
        )

        key, value = materialize_absorbed_kv(
            latent_cache,
            rope_cache,
            self.w_uk_t,
            self.w_uv,
            used_length=self.cache_length,
        )
        selected_key = torch.index_select(key, 2, selected)
        selected_value = torch.index_select(value, 2, selected)
        query = torch.cat((query_nope, query_rope), dim=-1).transpose(1, 2)
        sparse_k = selected.shape[0]
        scores = torch.bmm(
            query.reshape(self.heads, 1, self.nope + self.rope),
            selected_key.reshape(
                self.heads, sparse_k, self.nope + self.rope
            ).transpose(1, 2),
        ).view(1, self.heads, 1, sparse_k)
        scores = scores * scale
        valid = selected.unsqueeze(0) <= position.unsqueeze(1)
        scores = scores.masked_fill(
            ~valid.unsqueeze(1), torch.finfo(scores.dtype).min
        )
        probabilities = torch.softmax(scores, dim=-1)
        expected = torch.bmm(
            probabilities.reshape(self.heads, 1, sparse_k),
            selected_value.reshape(self.heads, sparse_k, self.value),
        ).transpose(0, 1).reshape(1, 1, self.heads * self.value)

        torch.testing.assert_close(absorbed, expected, atol=2e-5, rtol=2e-5)

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
