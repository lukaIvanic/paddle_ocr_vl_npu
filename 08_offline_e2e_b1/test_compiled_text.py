"""CPU structural and numerical tests for Experiment 08 compiled text prefill."""

from __future__ import annotations

import unittest

import torch

from config import PaddleOCRTextConfig
from local_modeling_paddleocr_vl import (
    LocalPaddleOCRVLStaticCache,
    PaddleOCRTextModel,
)
from text_compile import (
    DEFAULT_TEXT_BUCKETS,
    StaticTextPrefill,
    parse_text_buckets,
    prepare_text_bucket,
    select_text_bucket,
    unique_bucket_forward,
)


class _TinyConditionalModel(torch.nn.Module):
    def __init__(self, config: PaddleOCRTextConfig):
        super().__init__()
        self.model = PaddleOCRTextModel(config)
        self.config = type("Config", (), {"text_config": config})()


class CompiledTextTest(unittest.TestCase):
    def test_default_buckets_cover_static_shapes(self) -> None:
        self.assertEqual(DEFAULT_TEXT_BUCKETS, (32, 64, 128, 256, 512, 1024, 2048))
        self.assertEqual(select_text_bucket(1, DEFAULT_TEXT_BUCKETS), 32)
        self.assertEqual(select_text_bucket(32, DEFAULT_TEXT_BUCKETS), 32)
        self.assertEqual(select_text_bucket(33, DEFAULT_TEXT_BUCKETS), 64)
        self.assertIsNone(select_text_bucket(2049, DEFAULT_TEXT_BUCKETS))

    def test_bucket_parser_accepts_arbitrary_static_shapes(self) -> None:
        self.assertEqual(parse_text_buckets("32,48,96"), (32, 48, 96))
        self.assertEqual(select_text_bucket(33, (32, 48, 96)), 48)
        for invalid in ("", "32,32", "64,32", "0,32", "-32,32"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_text_buckets(invalid)

    def test_each_bucket_gets_a_distinct_compiler_code_object(self) -> None:
        config = self._tiny_config()
        module = StaticTextPrefill(_TinyConditionalModel(config)).eval()
        first = unique_bucket_forward(module, 32)
        second = unique_bucket_forward(module, 64)
        self.assertIsNot(first.__func__.__code__, second.__func__.__code__)
        self.assertEqual(first.__func__.__code__.co_name, "text_prefill_bucket_32")
        self.assertEqual(second.__func__.__code__.co_name, "text_prefill_bucket_64")

    def test_masked_padding_matches_unpadded_hidden_and_valid_kv(self) -> None:
        torch.manual_seed(19)
        config = self._tiny_config()
        conditional = _TinyConditionalModel(config).eval()
        real_seq_len = 5
        physical_seq_len = 8
        cache_length = 12
        inputs_embeds = torch.randn((1, real_seq_len, config.hidden_size))
        attention_mask = torch.ones((1, real_seq_len), dtype=torch.int64)
        position_ids = torch.arange(real_seq_len, dtype=torch.int64).view(1, 1, -1).expand(3, 1, -1)
        eager_cache = LocalPaddleOCRVLStaticCache.allocate(
            config,
            batch_size=1,
            cache_length=cache_length,
            device=torch.device("cpu"),
            dtype=inputs_embeds.dtype,
        )
        padded_cache = LocalPaddleOCRVLStaticCache.allocate(
            config,
            batch_size=1,
            cache_length=cache_length,
            device=torch.device("cpu"),
            dtype=inputs_embeds.dtype,
        )

        with torch.inference_mode():
            eager_hidden = conditional.model.forward_prefill_static(
                inputs_embeds,
                attention_mask,
                position_ids,
                eager_cache,
            )[:, -1:, :]
            prepared = prepare_text_bucket(
                inputs_embeds,
                attention_mask,
                position_ids,
                physical_seq_len=physical_seq_len,
            )
            padded_hidden = StaticTextPrefill(conditional).eval()(
                prepared.inputs_embeds,
                prepared.attention_mask,
                prepared.position_ids,
                prepared.last_token_index,
                *padded_cache.flat_tensors(),
            )

        torch.testing.assert_close(padded_hidden, eager_hidden, rtol=1e-5, atol=1e-5)
        for eager, padded in zip(eager_cache.flat_tensors(), padded_cache.flat_tensors()):
            torch.testing.assert_close(
                padded[:, :, :real_seq_len],
                eager[:, :, :real_seq_len],
                rtol=1e-5,
                atol=1e-5,
            )
        self.assertEqual(tuple(prepared.inputs_embeds.shape), (1, 8, 16))
        self.assertTrue(torch.equal(prepared.attention_mask[0, 5:], torch.zeros(3, dtype=torch.int64)))
        self.assertTrue(torch.equal(prepared.position_ids[:, :, 5:], torch.ones(3, 1, 3, dtype=torch.int64)))
        self.assertEqual(prepared.last_token_index.item(), real_seq_len - 1)

    @staticmethod
    def _tiny_config() -> PaddleOCRTextConfig:
        return PaddleOCRTextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [1, 1],
            },
        )


if __name__ == "__main__":
    unittest.main()
