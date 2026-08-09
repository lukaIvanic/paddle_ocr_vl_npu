"""CPU-side contract tests for the experimental GQA AIV graph operator."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.gqa_increfa_aiv import (
    MIN_KV_LENGTH_FOR_48_CORES,
    gqa_incre_flash_attention_aiv,
)


class GqaIncrefaAivContractTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
