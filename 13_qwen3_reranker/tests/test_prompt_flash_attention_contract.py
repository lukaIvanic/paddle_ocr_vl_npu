from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from local_modeling_qwen3_reranker import (  # noqa: E402
    PROMPT_FA_FULL_ATTENTION_TOKENS,
    prompt_flash_attention_bnsd_310p_compatible,
)


class PromptFlashAttentionContractTest(unittest.TestCase):
    def test_310p_contract_expands_gqa_and_omits_unsupported_arguments(self) -> None:
        captured: dict[str, object] = {}

        def fake_prompt_flash_attention(query, key, value, **kwargs):
            captured.update(query=query, key=key, value=value, kwargs=kwargs)
            return query

        fake_torch_npu = SimpleNamespace(npu_prompt_flash_attention=fake_prompt_flash_attention)
        query = torch.randn(2, 4, 3, 8, dtype=torch.float16)
        key = torch.randn(2, 2, 3, 8, dtype=torch.float16)
        value = torch.randn(2, 2, 3, 8, dtype=torch.float16)
        attention_mask = torch.zeros(2, 1, 3, 3, dtype=torch.bool)

        with patch.dict(sys.modules, {"torch_npu": fake_torch_npu}):
            output = prompt_flash_attention_bnsd_310p_compatible(
                query,
                key,
                value,
                attention_mask=attention_mask,
                num_heads=4,
                scale=8**-0.5,
            )

        self.assertIs(output, query)
        self.assertEqual(captured["key"].shape, (2, 4, 3, 8))
        self.assertEqual(captured["value"].shape, (2, 4, 3, 8))
        kwargs = captured["kwargs"]
        self.assertNotIn("actual_seq_lengths", kwargs)
        self.assertNotIn("actual_seq_lengths_kv", kwargs)
        self.assertNotIn("num_key_value_heads", kwargs)
        self.assertEqual(kwargs["num_heads"], 4)
        self.assertEqual(kwargs["input_layout"], "BNSD")
        self.assertEqual(kwargs["pre_tokens"], PROMPT_FA_FULL_ATTENTION_TOKENS)
        self.assertEqual(kwargs["next_tokens"], PROMPT_FA_FULL_ATTENTION_TOKENS)
        self.assertEqual(kwargs["sparse_mode"], 0)
        self.assertEqual(kwargs["atten_mask"].dtype, torch.bool)
        self.assertTrue(kwargs["atten_mask"].is_contiguous())

    def test_310p_contract_rejects_non_fp16_inputs(self) -> None:
        query = torch.randn(1, 2, 3, 8, dtype=torch.float32)
        mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "requires float16"):
            prompt_flash_attention_bnsd_310p_compatible(
                query,
                query,
                query,
                attention_mask=mask,
                num_heads=2,
                scale=8**-0.5,
            )


if __name__ == "__main__":
    unittest.main()
