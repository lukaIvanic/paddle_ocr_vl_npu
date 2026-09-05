"""CPU checks for equivalent packing and correct prefetch storage references."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paddleocr_vl.model.text_decode import (
    _decode_mlp, prepare_decode_optimization_modules,
    prepare_decode_weight_prefetch, resolve_decode_optimization,
)

BASE = "combined_apply_complete_layer_prefetch1_rope_lut"
PACKED = BASE + "_packed_mlp"


def make_model():
    layers = []
    for _ in range(2):
        attn = SimpleNamespace(**{name: nn.Linear(16, size, bias=False).double()
            for name, size in (("q_proj", 16), ("k_proj", 4), ("v_proj", 4), ("o_proj", 16))})
        mlp = SimpleNamespace(gate_proj=nn.Linear(16, 32, bias=False).double(),
                              up_proj=nn.Linear(16, 32, bias=False).double(),
                              down_proj=nn.Linear(32, 16, bias=False).double())
        layers.append(SimpleNamespace(self_attn=attn, mlp=mlp))
    return SimpleNamespace(model=SimpleNamespace(layers=layers),
                           lm_head=nn.Linear(16, 64, bias=False).double())


class PackedPrefetchTest(unittest.TestCase):
    def test_equivalent_packing(self):
        torch.manual_seed(93)
        model = make_model()
        config = prepare_decode_optimization_modules(model, PACKED)
        # CPU uses the same gate/up partition; stock NPU SwiGLU is tested in
        # the real server, not replaced by a claimed CPU accelerator test.
        config = replace(config, npu_swiglu=False)
        for batch in (1, 2, 4, 8):
            x = torch.randn(batch, 1, 16, dtype=torch.float64)
            for layer in model.model.layers:
                mlp = layer.mlp
                expected = mlp.down_proj(torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
                torch.testing.assert_close(_decode_mlp(mlp, x, config), expected,
                                           rtol=1e-12, atol=1e-12)

    def test_prefetch_uses_executed_weights(self):
        for preset in (BASE, PACKED):
            model = make_model()
            config = prepare_decode_optimization_modules(model, preset)
            prepare_decode_weight_prefetch(model, config)
            layer0, layer1 = model.model.layers
            expected_mlp = ([layer1.mlp.decode_gate_up_proj.weight] if config.packed_mlp else
                            [layer1.mlp.gate_proj.weight, layer1.mlp.up_proj.weight])
            expected = [layer1.self_attn.decode_qkv_proj.weight, layer1.self_attn.o_proj.weight,
                        *expected_mlp, layer1.mlp.down_proj.weight]
            self.assertEqual([id(w) for w in layer0._decode_prefetch_future_layers],
                             [id(w) for w in expected])
            self.assertEqual([id(w) for w in layer1.self_attn._decode_prefetch_current_mlp],
                             [id(w) for w in [*expected_mlp, layer1.mlp.down_proj.weight]])
            self.assertIs(layer1._decode_prefetch_future_layers[0], model.lm_head.weight)


if __name__ == "__main__":
    unittest.main()
