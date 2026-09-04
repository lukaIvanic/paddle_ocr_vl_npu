import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from vision_prefill_compile import StaticMinerUVisionBlocks


class _Visual(nn.Module):
    def __init__(self, *, embed_dim: int = 16, num_heads: int = 2) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        self.blocks = nn.ModuleList()


def _fake_grouped_matmul(inputs, weights, *, bias, **_kwargs):
    output = torch.matmul(inputs[0], weights[0][0])
    if bias is not None:
        output = output + bias[0][0]
    return [output]


class VisionPrefillContractTests(unittest.TestCase):
    def test_manual_fp32_layer_norm_tracks_module(self):
        torch.manual_seed(0)
        layer_norm = nn.LayerNorm(16, eps=1e-6).half()
        values = torch.randn(1, 7, 16, dtype=torch.float16)
        wrapper = StaticMinerUVisionBlocks(
            _Visual(),
            attention_impl="manual",
            layer_norm_impl="manual_fp32",
        )

        expected = layer_norm(values)
        actual = wrapper._layer_norm(layer_norm, values)

        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)

    def test_grouped_matmul_3d_weight_tracks_linear_for_bsh_input(self):
        torch.manual_seed(1)
        linear = nn.Linear(16, 24, bias=True)
        values = torch.randn(1, 9, 16)
        fake_torch_npu = types.SimpleNamespace(
            npu_grouped_matmul=_fake_grouped_matmul,
        )

        with patch.dict(sys.modules, {"torch_npu": fake_torch_npu}):
            actual = StaticMinerUVisionBlocks._grouped_linear(values, linear)

        self.assertEqual(tuple(actual.shape), (1, 9, 24))
        torch.testing.assert_close(actual, linear(values))

    def test_promptfa_padding_preserves_native_prefix(self):
        wrapper = StaticMinerUVisionBlocks(
            _Visual(embed_dim=160, num_heads=2),
            promptfa_pad_head_dim_to=96,
        )
        values = torch.randn(1, 2, 5, 80)

        padded = wrapper._pad_promptfa_head_dim(values)

        self.assertEqual(tuple(padded.shape), (1, 2, 5, 96))
        torch.testing.assert_close(padded[..., :80], values)
        self.assertEqual(int(torch.count_nonzero(padded[..., 80:])), 0)

    def test_promptfa_padding_rejected_for_manual_attention(self):
        with self.assertRaisesRegex(ValueError, "only valid"):
            StaticMinerUVisionBlocks(
                _Visual(embed_dim=160, num_heads=2),
                attention_impl="manual",
                promptfa_pad_head_dim_to=96,
            )


if __name__ == "__main__":
    unittest.main()
