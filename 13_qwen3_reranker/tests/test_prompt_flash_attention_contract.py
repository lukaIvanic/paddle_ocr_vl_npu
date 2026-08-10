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
    LocalQwen3RerankerAttention,
    LocalQwen3RerankerForCausalLM,
    LocalQwen3RerankerRotaryEmbedding,
    PROMPT_FA_FULL_ATTENTION_TOKENS,
    build_310p_square_promptfa_mask,
    build_left_padded_causal_bool_mask,
    build_left_padded_causal_bool_mask_chunk,
    linear_tokenwise,
    prompt_flash_attention_bnsd_310p_compatible,
    prompt_flash_attention_bsnd_310p_compatible,
    reranker_transformer_linears,
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

    def test_310p_contract_keeps_native_gqa_and_omits_sequence_lengths(self) -> None:
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
        self.assertEqual(captured["key"].shape, (2, 2, 3, 8))
        self.assertEqual(captured["value"].shape, (2, 2, 3, 8))
        kwargs = captured["kwargs"]
        self.assertNotIn("actual_seq_lengths", kwargs)
        self.assertNotIn("actual_seq_lengths_kv", kwargs)
        self.assertEqual(kwargs["num_key_value_heads"], 2)
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

    def test_310p_rectangular_mask_square_pads_query_and_discards_dummy_rows(self) -> None:
        captured: dict[str, object] = {}

        def fake_prompt_flash_attention(query, key, value, **kwargs):
            captured.update(query=query, key=key, value=value, kwargs=kwargs)
            return query

        fake_torch_npu = SimpleNamespace(npu_prompt_flash_attention=fake_prompt_flash_attention)
        query = torch.randn(2, 4, 2, 8, dtype=torch.float16)
        key = torch.randn(2, 2, 4, 8, dtype=torch.float16)
        value = torch.randn(2, 2, 4, 8, dtype=torch.float16)
        real_mask = torch.tensor(
            [
                [[[False, False, True, True], [False, False, False, True]]],
                [[[False, True, True, True], [False, False, True, True]]],
            ],
            dtype=torch.bool,
        )

        with patch.dict(sys.modules, {"torch_npu": fake_torch_npu}):
            output = prompt_flash_attention_bnsd_310p_compatible(
                query,
                key,
                value,
                attention_mask=real_mask,
                num_heads=4,
                scale=8**-0.5,
            )

        torch.testing.assert_close(output, query)
        self.assertEqual(captured["query"].shape, (2, 4, 4, 8))
        self.assertEqual(captured["key"].shape, (2, 2, 4, 8))
        square_mask = captured["kwargs"]["atten_mask"]
        self.assertEqual(square_mask.shape, (2, 1, 4, 4))
        self.assertFalse(square_mask[:, :, :2].all(dim=-1).any().item())
        torch.testing.assert_close(square_mask[:, :, -2:], real_mask)

    def test_prebuilt_square_mask_skips_per_attention_mask_construction(self) -> None:
        captured: dict[str, object] = {}

        def fake_prompt_flash_attention(query, key, value, **kwargs):
            captured.update(query=query, key=key, value=value, kwargs=kwargs)
            return query

        fake_torch_npu = SimpleNamespace(npu_prompt_flash_attention=fake_prompt_flash_attention)
        query = torch.randn(2, 4, 2, 8, dtype=torch.float16)
        key = torch.randn(2, 2, 4, 8, dtype=torch.float16)
        value = torch.randn(2, 2, 4, 8, dtype=torch.float16)
        real_mask = torch.zeros(2, 1, 2, 4, dtype=torch.bool)
        square_mask = build_310p_square_promptfa_mask(real_mask)

        with patch.dict(sys.modules, {"torch_npu": fake_torch_npu}):
            output = prompt_flash_attention_bnsd_310p_compatible(
                query,
                key,
                value,
                attention_mask=square_mask,
                num_heads=4,
                scale=8**-0.5,
                attention_mask_is_square=True,
            )

        torch.testing.assert_close(output, query)
        self.assertIs(captured["kwargs"]["atten_mask"], square_mask)

    def test_310p_bsnd_contract_square_pads_sequence_axis(self) -> None:
        captured: dict[str, object] = {}

        def fake_prompt_flash_attention(query, key, value, **kwargs):
            captured.update(query=query, key=key, value=value, kwargs=kwargs)
            return query

        fake_torch_npu = SimpleNamespace(npu_prompt_flash_attention=fake_prompt_flash_attention)
        query = torch.randn(2, 2, 4, 8, dtype=torch.float16)
        key = torch.randn(2, 4, 2, 8, dtype=torch.float16)
        value = torch.randn(2, 4, 2, 8, dtype=torch.float16)
        real_mask = torch.zeros(2, 1, 2, 4, dtype=torch.bool)

        with patch.dict(sys.modules, {"torch_npu": fake_torch_npu}):
            output = prompt_flash_attention_bsnd_310p_compatible(
                query,
                key,
                value,
                attention_mask=real_mask,
                num_heads=4,
                scale=8**-0.5,
            )

        torch.testing.assert_close(output, query)
        self.assertEqual(captured["query"].shape, (2, 4, 4, 8))
        self.assertEqual(captured["key"].shape, (2, 4, 2, 8))
        self.assertEqual(captured["value"].shape, (2, 4, 2, 8))
        self.assertEqual(captured["kwargs"]["input_layout"], "BSND")
        self.assertEqual(captured["kwargs"]["atten_mask"].shape, (2, 1, 4, 4))

    def test_project_qkv_bsnd_is_layout_view_of_bnsd_result(self) -> None:
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
        attention = LocalQwen3RerankerAttention(config, attention_impl="prompt_flash_attention")
        rotary = LocalQwen3RerankerRotaryEmbedding(config)
        hidden_states = torch.randn(2, 3, 16)
        position_ids = torch.arange(3).view(1, 3).expand(2, -1)
        cos, sin = rotary(position_ids, dtype=torch.float32, device=torch.device("cpu"))

        bnsd = attention.project_qkv(hidden_states, cos, sin, output_layout="BNSD")
        bsnd = attention.project_qkv(hidden_states, cos, sin, output_layout="BSND")

        for bnsd_tensor, bsnd_tensor in zip(bnsd, bsnd):
            torch.testing.assert_close(bsnd_tensor, bnsd_tensor.transpose(1, 2))

    def test_expanded_prefix_cache_preset_remains_explicit(self) -> None:
        config = LocalQwen3RerankerConfig(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=False,
        )
        model = LocalQwen3RerankerForCausalLM(config)
        selected = model.set_prefill_optimization("expanded_prefix_kv")
        key = torch.randn(1, 2, 3, 8)
        value = torch.randn(1, 2, 3, 8)
        keys, values = model.prepare_prefix_caches((key,), (value,))

        self.assertTrue(selected.expanded_prefix_kv)
        self.assertEqual(keys[0].shape, (1, 4, 3, 8))
        self.assertEqual(values[0].shape, (1, 4, 3, 8))
        torch.testing.assert_close(keys[0][:, 0], key[:, 0])
        torch.testing.assert_close(keys[0][:, 1], key[:, 0])

    def test_combined_bsnd_preset_transposes_compact_prefix_cache_once(self) -> None:
        config = LocalQwen3RerankerConfig(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=False,
        )
        model = LocalQwen3RerankerForCausalLM(config)
        selected = model.set_prefill_optimization("combined_bsnd")
        key = torch.randn(1, 2, 3, 8)
        value = torch.randn(1, 2, 3, 8)
        keys, values = model.prepare_prefix_caches((key,), (value,))

        self.assertEqual(selected.prompt_fa_layout, "BSND")
        self.assertFalse(selected.expanded_prefix_kv)
        self.assertEqual(model.layers[0].self_attn.prompt_fa_layout, "BSND")
        self.assertEqual(keys[0].shape, (1, 3, 2, 8))
        self.assertEqual(values[0].shape, (1, 3, 2, 8))
        torch.testing.assert_close(keys[0][:, :, 0], key[:, 0])
        torch.testing.assert_close(keys[0][:, :, 1], key[:, 1])

    def test_reranker_transformer_linears_selects_seven_projections_per_layer(self) -> None:
        config = LocalQwen3RerankerConfig(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=True,
        )
        model = LocalQwen3RerankerForCausalLM(config)
        modules = reranker_transformer_linears(model)

        self.assertEqual(len(modules), 14)
        self.assertEqual(len({name for name, _module in modules}), 14)
        self.assertFalse(any("embed_tokens" in name or "lm_head" in name for name, _ in modules))


if __name__ == "__main__":
    unittest.main()
