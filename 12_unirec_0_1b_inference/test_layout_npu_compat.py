#!/usr/bin/env python3
"""CPU-only checks for the PP-DocLayoutV2 NPU compatibility bindings."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent


class _LayoutClass(nn.Module):
    pass


def _load_layout_torchair():
    layout_module = types.ModuleType(
        "transformers.models.pp_doclayout_v2.modeling_pp_doclayout_v2"
    )
    layout_module.PPDocLayoutV2SelfAttention = _LayoutClass
    layout_module.PPDocLayoutV2ReadingOrderSelfAttention = _LayoutClass
    layout_module.PPDocLayoutV2GlobalPointer = _LayoutClass
    layout_module.PPDocLayoutV2SinePositionEmbedding = _LayoutClass
    layout_module.create_bidirectional_mask = (
        lambda *, config, inputs_embeds, attention_mask: attention_mask
    )

    transformers = types.ModuleType("transformers")
    transformers.__path__ = []
    models = types.ModuleType("transformers.models")
    models.__path__ = []
    pp_doclayout = types.ModuleType("transformers.models.pp_doclayout_v2")
    pp_doclayout.__path__ = []
    pp_doclayout.modeling_pp_doclayout_v2 = layout_module

    module_names = {
        "transformers": transformers,
        "transformers.models": models,
        "transformers.models.pp_doclayout_v2": pp_doclayout,
        "transformers.models.pp_doclayout_v2.modeling_pp_doclayout_v2": (
            layout_module
        ),
    }
    previous = {name: sys.modules.get(name) for name in module_names}
    try:
        sys.modules.update(module_names)
        spec = importlib.util.spec_from_file_location(
            "layout_torchair_under_test",
            ROOT / "layout_torchair.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load layout_torchair.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


class _CapturingEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.Identity()
        self.dropout = nn.Identity()
        self.input_ids: torch.Tensor | None = None

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
    ) -> torch.Tensor:
        del bbox
        self.input_ids = input_ids.detach().clone()
        return input_ids.to(torch.float32).unsqueeze(-1)


class _IdentityEncoder(nn.Module):
    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del bbox, attention_mask
        return SimpleNamespace(last_hidden_state=hidden_states)


class _ReadingOrder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            pad_token_id=0,
            start_token_id=1,
            pred_token_id=2,
            end_token_id=3,
        )
        self.embeddings = _CapturingEmbeddings()
        self.label_embeddings = nn.Embedding(4, 1)
        self.label_features_projection = nn.Identity()
        self.encoder = _IdentityEncoder()
        self.relative_head = nn.Identity()


class _AnchorOwner:
    def generate_anchors(self):
        raise AssertionError("unpatched anchor method")


class _Model:
    def __init__(self) -> None:
        self.model = _AnchorOwner()
        self.reading_order = _ReadingOrder()


class LayoutNpuCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layout_torchair = _load_layout_torchair()

    def test_eager_bundle_binds_anchor_and_reading_order_only(self) -> None:
        model = _Model()
        self.layout_torchair.make_eager_npu_compatible(model)

        self.assertIs(
            model.model.generate_anchors.__func__,
            self.layout_torchair._generate_anchors,
        )
        self.assertIs(
            model.reading_order.forward.__func__,
            self.layout_torchair._reading_order,
        )

    def test_reading_order_matches_index_assignment_at_boundary_counts(self) -> None:
        model = _Model()
        self.layout_torchair.make_eager_npu_compatible(model)
        mask = torch.tensor(
            [
                [False, False, False, False],
                [True, False, False, False],
                [True, True, True, True],
            ]
        )
        boxes = torch.zeros((3, 4, 4), dtype=torch.float32)

        output = model.reading_order(boxes=boxes, mask=mask)

        expected = torch.tensor(
            [
                [1, 3, 0, 0, 0, 0],
                [1, 2, 3, 0, 0, 0],
                [1, 2, 2, 2, 2, 3],
            ],
            dtype=torch.long,
        )
        torch.testing.assert_close(model.reading_order.embeddings.input_ids, expected)
        torch.testing.assert_close(output, expected[:, 1:5].to(torch.float32).unsqueeze(-1))

    def test_adapter_installs_the_eager_bundle(self) -> None:
        source = (ROOT / "opendoc_layout_npu.py").read_text(encoding="utf-8")
        self.assertIn("make_eager_npu_compatible(self.model)", source)
        self.assertNotIn("from layout_torchair import _generate_anchors", source)


if __name__ == "__main__":
    unittest.main()
