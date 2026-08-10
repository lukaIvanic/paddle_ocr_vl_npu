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
    LocalQwen3RerankerConfig,
    LocalQwen3RerankerRotaryEmbedding,
    PROMPT_FA_FULL_ATTENTION_TOKENS,
    build_left_padded_causal_bool_mask,
    build_left_padded_causal_bool_mask_chunk,
    linear_tokenwise,
    prompt_flash_attention_bnsd_310p_compatible,
)


class PromptFlashAttentionContractTest(unittest.TestCase):
    def test_chunk_mask_matches_full_mask_slice(self) -> None:
        attention_mask = torch.tensor(
            [[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]],
            dtype=torch.long,
        )
        full_mask = build_left_padded_causal_bool_mask(attention_mask)
        for query_start, query_end in ((0, 2), (2, 4), (4, 6)):
            chunk_mask = build_left_padded_causal_bool_mask_chunk(
                attention_mask,
                query_start=query_start,
                query_end=query_end,
            )
            torch.testing.assert_close(
                chunk_mask,
                full_mask[:, :, query_start:query_end, :query_end],
            )

    def test_tokenwise_linear_matches_rank3_linear(self) -> None:
        torch.manual_seed(0)
        linear = torch.nn.Linear(8, 12, bias=False)
        hidden_states = torch.randn(2, 3, 8)
        torch.testing.assert_close(
            linear_tokenwise(linear, hidden_states),
            linear(hidden_states),
        )

    def test_rotary_embedding_uses_compile_safe_outer_product(self) -> None:
        config = LocalQwen3RerankerConfig(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=False,
        )
        rotary = LocalQwen3RerankerRotaryEmbedding(config)
        position_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
        cos, sin = rotary(position_ids, dtype=torch.float32, device=torch.device("cpu"))
        expected_freqs = position_ids.float().unsqueeze(-1) * rotary.inv_freq.view(1, 1, -1)
        expected = torch.cat((expected_freqs, expected_freqs), dim=-1)
        torch.testing.assert_close(cos, expected.cos())
        torch.testing.assert_close(sin, expected.sin())

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
