"""CPU algebra/compile-boundary tests, not an NPU performance claim."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paddleocr_vl.model import vision_prefill as v


def model_fixture():
    layers = nn.ModuleList()
    for _ in range(2):
        attention = nn.Module()
        attention.head_dim, attention.num_heads, attention.scaling = 72, 2, 72 ** -0.5
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(attention, name, nn.Linear(144, 144))
        layer = nn.Module()
        layer.self_attn = attention
        layer.layer_norm1 = nn.LayerNorm(144)
        layer.layer_norm2 = nn.LayerNorm(144)
        layer.mlp = nn.Module()
        layer.mlp.fc1, layer.mlp.fc2 = nn.Linear(144, 64), nn.Linear(64, 144)
        layer.mlp.hidden_act = "gelu"
        layers.append(layer)
    transformer = nn.Module()
    transformer.encoder = nn.Module()
    transformer.encoder.layers = layers
    transformer.post_layernorm = nn.LayerNorm(144)
    return SimpleNamespace(visual=SimpleNamespace(vision_model=transformer))


def reference_attention(q, k, value, *, num_heads, scale, atten_mask):
    return (q @ k.transpose(-1, -2) * scale).masked_fill(
        atten_mask, float("-inf")
    ).softmax(-1) @ value


class VisionWeightPaddingTest(unittest.TestCase):
    def test_two_layers_masks_rope_and_compile_boundary(self):
        torch.manual_seed(5)
        original = model_fixture()
        padded = copy.deepcopy(original)
        v.prepare_vision_attention_weight_padding(padded)
        baseline = v.VisionPrefillStage(original, attention_impl="prompt_flash_attention")
        candidate = v.VisionPrefillStage(padded, attention_impl="prompt_flash_attention")
        with patch.object(v, "vision_prompt_flash_attention_bnsd", reference_attention):
            for batch, seq in ((1, 7), (2, 11)):
                hidden = torch.randn(batch, seq, 144)
                angles = torch.randn(batch, seq, 36).repeat(1, 1, 2)
                mask = torch.arange(seq)[None, :] > torch.arange(seq)[:, None]
                args = (hidden, angles.cos(), angles.sin(), mask[None, None])
                with torch.no_grad():
                    expected = baseline(*args)
                    actual = candidate(*args)
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
                    compiled = torch.compile(candidate, backend="eager", fullgraph=True, dynamic=False)
                    torch.testing.assert_close(compiled(*args), actual, atol=0, rtol=0)

    def test_weights_scale_and_neutral_coordinates(self):
        original = model_fixture()
        padded = copy.deepcopy(original)
        v.prepare_vision_attention_weight_padding(padded)
        indices = [h * 80 + i + (4 if i >= 36 else 0) for h in range(2) for i in range(72)]
        zeros = sorted(set(range(160)) - set(indices))
        for old_layer, layer in zip(original.visual.vision_model.encoder.layers,
                                    padded.visual.vision_model.encoder.layers):
            old, new = old_layer.self_attn, layer.self_attn
            self.assertEqual(new.head_dim, 72)
            self.assertEqual(new.scaling, old.scaling)
            for name in ("q_proj", "k_proj", "v_proj"):
                source, target = getattr(old, name), getattr(new, name)
                self.assertTrue(torch.equal(target.weight[indices], source.weight))
                self.assertTrue(torch.equal(target.bias[indices], source.bias))
                self.assertEqual(target.weight[zeros].count_nonzero().item(), 0)
                self.assertEqual(target.bias[zeros].count_nonzero().item(), 0)
            self.assertTrue(torch.equal(new.out_proj.weight[:, indices], old.out_proj.weight))
            self.assertEqual(new.out_proj.weight[:, zeros].count_nonzero().item(), 0)
        with self.assertRaises(ValueError):
            v.prepare_vision_attention_weight_padding(padded)
        factors = torch.randn(1, 9, 72)
        extended = v.pad_vision_rope_halves(factors, 1.0)
        self.assertTrue(torch.equal(extended[..., :36], factors[..., :36]))
        self.assertTrue(torch.equal(extended[..., 40:76], factors[..., 36:]))
        self.assertTrue(torch.all(extended[..., 36:40] == 1).item())
        self.assertTrue(torch.all(extended[..., 76:] == 1).item())


if __name__ == "__main__":
    unittest.main()
