"""CPU-side contract tests for the experimental GQA AIV graph operator."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.gqa_increfa_aiv import (
    MIN_KV_LENGTH_FOR_48_CORES,
    gqa_incre_flash_attention_aiv,
)
from paddleocr_vl.model import text_decode


class GqaIncrefaAivContractTest(unittest.TestCase):
    def test_nonsplit_megakernel_uses_separate_prep_and_16_aiv_blocks(self) -> None:
        optimization = text_decode.resolve_decode_optimization(
            "paddle_decoder_megakernel_b1_nonsplit_gqa"
        )

        self.assertTrue(optimization.super_kernel_scope)
        self.assertTrue(optimization.ascendc_kv_scatter_query)
        self.assertFalse(optimization.ascendc_decode_gqa)
        self.assertTrue(optimization.ascendc_decode_gqa_attention)
        self.assertEqual(optimization.gqa_aiv_vector_core_count, 16)
        self.assertIn(
            "strict-scope-check=abort",
            optimization.super_kernel_options,
        )

    def test_rejects_unsafe_48_core_short_partition(self) -> None:
        kv_length = MIN_KV_LENGTH_FOR_48_CORES - 1
        query = torch.empty((1, 16, 1, 128), dtype=torch.float16)
        key = torch.empty((1, 2, kv_length, 128), dtype=torch.float16)
        value = torch.empty_like(key)
        mask = torch.empty((1, 1, 1, kv_length), dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "requires KV length >= 1536"):
            gqa_incre_flash_attention_aiv(
                query,
                key,
                value,
                mask,
                num_heads=16,
                num_key_value_heads=2,
                scale_value=128**-0.5,
                vector_core_count=48,
            )

    def test_fused_decode_attention_uses_static_b1_shape(self) -> None:
        query = torch.empty((1, 16, 1, 128), dtype=torch.float16)
        key_state = torch.empty((1, 2, 1, 128), dtype=torch.float16)
        value_state = torch.empty_like(key_state)
        attention = SimpleNamespace(
            num_heads=16,
            num_key_value_heads=2,
            head_dim=128,
            scaling=128**-0.5,
            o_proj=object(),
        )
        optimization = text_decode.resolve_decode_optimization(
            "paddle_decoder_megakernel_b1_fused_gqa"
        )

        with (
            mock.patch.object(
                text_decode,
                "_project_decode_qkv",
                return_value=(query, key_state, value_state),
            ),
            mock.patch.object(
                text_decode,
                "_apply_decode_rotary",
                return_value=(query, key_state),
            ),
            mock.patch.object(
                text_decode,
                "decode_gqa_incre_flash_attention_aiv",
                return_value=query,
            ),
            mock.patch.object(
                text_decode,
                "_linear_tokenwise",
                side_effect=lambda _linear, tensor: tensor,
            ),
        ):
            output = text_decode._decode_attention(
                attention,
                torch.empty((1, 1, 1024), dtype=torch.float16),
                (torch.empty(0), torch.empty(0)),
                None,
                torch.empty((1, 2, 1024, 128), dtype=torch.float16),
                torch.empty((1, 2, 1024, 128), dtype=torch.float16),
                torch.zeros((1,), dtype=torch.int64),
                torch.empty((1, 1, 1, 1024), dtype=torch.bool),
                None,
                None,
                optimization,
            )

        self.assertEqual(output.shape, (1, 1, 2048))


if __name__ == "__main__":
    unittest.main()
