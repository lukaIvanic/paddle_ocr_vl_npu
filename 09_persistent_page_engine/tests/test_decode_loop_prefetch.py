"""Verify the optional last-layer hint changes no other prefetch selection."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paddleocr_vl.model.text_decode import (
    prepare_decode_weight_prefetch, resolve_decode_optimization,
)

CONTROL = "combined_apply_complete_layer_prefetch1_rope_lut"
CANDIDATE = CONTROL + "_loop_prefetch"


class LoopPrefetchTest(unittest.TestCase):
    def test_only_new_setting_is_next_iteration_hint(self):
        a = asdict(resolve_decode_optimization(CONTROL))
        b = asdict(resolve_decode_optimization(CANDIDATE))
        self.assertEqual({k for k in a if a[k] != b[k]},
                         {"name", "prefetch_next_iteration"})

    def test_last_layer_adds_first_weights_after_head(self):
        def layer():
            return SimpleNamespace(
                self_attn=SimpleNamespace(decode_qkv_proj=nn.Linear(4, 4), o_proj=nn.Linear(4, 4)),
                mlp=SimpleNamespace(gate_proj=nn.Linear(4, 4), up_proj=nn.Linear(4, 4), down_proj=nn.Linear(4, 4)))
        layers = [layer() for _ in range(18)]
        model = SimpleNamespace(model=SimpleNamespace(layers=layers), lm_head=nn.Linear(4, 8))
        prepare_decode_weight_prefetch(model, CONTROL)
        old = [tuple(map(id, x._decode_prefetch_future_layers)) for x in layers]
        prepare_decode_weight_prefetch(model, CANDIDATE)
        new = [tuple(map(id, x._decode_prefetch_future_layers)) for x in layers]
        self.assertEqual(new[:-1], old[:-1])
        first = layers[0]
        expected = (model.lm_head.weight, first.self_attn.decode_qkv_proj.weight,
                    first.self_attn.o_proj.weight, first.mlp.gate_proj.weight,
                    first.mlp.up_proj.weight, first.mlp.down_proj.weight)
        self.assertEqual(new[-1], tuple(map(id, expected)))
        self.assertEqual(old[-1], (id(model.lm_head.weight),))


if __name__ == "__main__":
    unittest.main()
