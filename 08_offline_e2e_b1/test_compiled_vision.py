"""CPU structural and numerical tests for Experiment 08 compiled vision."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from config import PaddleOCRVisionConfig
from local_modeling_paddleocr_vl import PaddleOCRVisionTransformer
from vision_compile import (
    DEFAULT_VISION_BUCKETS,
    StaticManualVisionEncoder,
    parse_vision_buckets,
    prepare_vision_bucket,
    select_vision_bucket,
    unique_bucket_forward,
)


class CompiledVisionTest(unittest.TestCase):
    def test_default_buckets_cover_requested_static_shapes(self) -> None:
        self.assertEqual(
            DEFAULT_VISION_BUCKETS,
            (16, 32, 64, 128, 256, 512, 1024, 2048),
        )
        self.assertEqual(select_vision_bucket(1, DEFAULT_VISION_BUCKETS), 16)
        self.assertEqual(select_vision_bucket(16, DEFAULT_VISION_BUCKETS), 16)
        self.assertEqual(select_vision_bucket(17, DEFAULT_VISION_BUCKETS), 32)
        self.assertEqual(select_vision_bucket(2048, DEFAULT_VISION_BUCKETS), 2048)
        self.assertIsNone(select_vision_bucket(2049, DEFAULT_VISION_BUCKETS))

    def test_bucket_parser_rejects_ambiguous_or_dynamic_shapes(self) -> None:
        self.assertEqual(parse_vision_buckets("16,32,64"), (16, 32, 64))
        for invalid in ("", "16,16", "32,16", "16,24", "0,16"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_vision_buckets(invalid)

    def test_each_bucket_gets_a_distinct_compiler_code_object(self) -> None:
        config = PaddleOCRVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
        )
        transformer = PaddleOCRVisionTransformer(config).eval()
        model = SimpleNamespace(visual=SimpleNamespace(vision_model=transformer))
        module = StaticManualVisionEncoder(model).eval()
        first = unique_bucket_forward(module, 16)
        second = unique_bucket_forward(module, 32)
        self.assertIsNot(first.__func__.__code__, second.__func__.__code__)
        self.assertEqual(first.__func__.__code__.co_name, "vision_encoder_bucket_16")
        self.assertEqual(second.__func__.__code__.co_name, "vision_encoder_bucket_32")

    def test_masked_padding_matches_unpadded_stock_encoder(self) -> None:
        torch.manual_seed(7)
        config = PaddleOCRVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=8,
            patch_size=2,
        )
        transformer = PaddleOCRVisionTransformer(config).eval()
        model = SimpleNamespace(
            visual=SimpleNamespace(vision_model=transformer),
        )
        prefix = torch.randn((6, config.hidden_size), dtype=torch.float32)
        grid = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        cu_seqlens = torch.tensor([0, 6], dtype=torch.int32)

        with torch.inference_mode():
            expected = transformer.post_layernorm(
                transformer.encoder(prefix, cu_seqlens, grid)
            )
            prepared = prepare_vision_bucket(
                model,
                prefix,
                grid,
                physical_seq_len=8,
            )
            actual = StaticManualVisionEncoder(model).eval()(
                prepared.prefix_hidden_states,
                prepared.rope_cos,
                prepared.rope_sin,
                prepared.attention_mask,
            )[0, :6]

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(tuple(prepared.prefix_hidden_states.shape), (1, 8, 16))
        self.assertTrue(bool(prepared.attention_mask[0, 0, 0, 7]))
        self.assertTrue(bool(prepared.attention_mask[0, 0, 7, 0]))
        self.assertFalse(bool(prepared.attention_mask[0, 0, 7, 7]))
        self.assertTrue(torch.equal(prepared.prefix_hidden_states[0, 6:], torch.zeros(2, 16)))
        self.assertTrue(torch.equal(prepared.rope_cos[0, 6:], torch.ones(2, 4)))
        self.assertTrue(torch.equal(prepared.rope_sin[0, 6:], torch.zeros(2, 4)))


if __name__ == "__main__":
    unittest.main()
