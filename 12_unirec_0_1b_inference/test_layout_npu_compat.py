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


class _SelfAttentionClass(nn.Module):
    pass


class _ReadingOrderSelfAttentionClass(nn.Module):
    pass


class _GlobalPointerClass(nn.Module):
    pass


class _SinePositionClass(nn.Module):
    pass


class _ReadingOrderEncoderClass(nn.Module):
    pass


def _load_layout_torchair():
    layout_module = types.ModuleType(
        "transformers.models.pp_doclayout_v2.modeling_pp_doclayout_v2"
    )
    layout_module.PPDocLayoutV2SelfAttention = _SelfAttentionClass
    layout_module.PPDocLayoutV2ReadingOrderSelfAttention = (
        _ReadingOrderSelfAttentionClass
    )
    layout_module.PPDocLayoutV2ReadingOrderEncoder = _ReadingOrderEncoderClass
    layout_module.PPDocLayoutV2GlobalPointer = _GlobalPointerClass
    layout_module.PPDocLayoutV2SinePositionEmbedding = _SinePositionClass
    layout_module.PPDocLayoutV2ForObjectDetectionOutput = SimpleNamespace
    def rejected_bidirectional_mask(**kwargs):
        del kwargs
        raise AssertionError("eager reading order must not call the HF mask helper")

    layout_module.create_bidirectional_mask = rejected_bidirectional_mask

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


def _load_layout_adapter():
    spec = importlib.util.spec_from_file_location(
        "opendoc_layout_npu_under_test",
        ROOT / "opendoc_layout_npu.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load opendoc_layout_npu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


class _IdentityEncoder(_ReadingOrderEncoderClass):
    def __init__(self) -> None:
        super().__init__()
        self.attention_mask: torch.Tensor | None = None

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del bbox
        self.attention_mask = attention_mask.detach().clone()
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


class _AnchorOwner(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def generate_anchors(self):
        raise AssertionError("unpatched anchor method")


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _AnchorOwner()
        self.reading_order = _ReadingOrder()

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("unpatched object-detection forward")


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

        expected_valid = torch.tensor(
            [
                [True, True, False, False, False, False],
                [True, True, True, False, False, False],
                [True, True, True, True, True, True],
            ]
        )
        zeros = torch.zeros_like(expected_valid, dtype=torch.float32)
        negative = torch.full_like(
            expected_valid, torch.finfo(torch.float32).min, dtype=torch.float32
        )
        expected_key_bias = torch.where(expected_valid, zeros, negative)
        expected_attention_mask = expected_key_bias.unsqueeze(1).unsqueeze(1).expand(
            3, 1, 6, 6
        )
        torch.testing.assert_close(
            model.reading_order.encoder.attention_mask, expected_attention_mask
        )

    def test_table_lookup_matches_advanced_indexing_for_float_and_int(self) -> None:
        indices = torch.tensor([[0, 2, 1], [1, 0, 2]], dtype=torch.long)
        for table in (
            torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32),
            torch.tensor([7, 3, 11], dtype=torch.int32),
            torch.arange(12, dtype=torch.float32).reshape(3, 4),
        ):
            expected = table[indices]
            actual = self.layout_torchair._table_lookup(table, indices)
            torch.testing.assert_close(actual, expected)

    def test_compile_bundle_rewrites_all_index_by_tensor_sources(self) -> None:
        model = _Model()
        self.layout_torchair.make_compile_compatible(model)

        self.assertIs(
            model.forward.__func__,
            self.layout_torchair._object_detection_forward,
        )
        self.assertIs(
            model.reading_order.encoder._cal_1d_pos_emb.__func__,
            self.layout_torchair._reading_order_cal_1d_pos_emb,
        )

    def test_adapter_installs_the_eager_bundle(self) -> None:
        source = (ROOT / "opendoc_layout_npu.py").read_text(encoding="utf-8")
        self.assertIn("make_eager_npu_compatible(self.model)", source)
        self.assertNotIn("from layout_torchair import _generate_anchors", source)

    def test_layout_postprocess_copies_only_required_outputs_to_cpu(self) -> None:
        module = _load_layout_adapter()
        outputs = SimpleNamespace(
            logits=torch.randn(1, 3, 4, requires_grad=True),
            pred_boxes=torch.randn(1, 3, 4, requires_grad=True),
            order_logits=torch.randn(1, 3, 3, requires_grad=True),
            ignored=torch.randn(1024),
        )

        copied = module._layout_outputs_for_cpu_postprocess(outputs)

        self.assertEqual(set(vars(copied)), {"logits", "pred_boxes", "order_logits"})
        for name in vars(copied):
            tensor = getattr(copied, name)
            self.assertEqual(tensor.device.type, "cpu")
            self.assertFalse(tensor.requires_grad)

    def test_depthwise_rewrites_are_exact_block_diagonal_convolutions(self) -> None:
        module = _load_layout_adapter()
        torch.manual_seed(17)
        inputs = torch.randn(1, 64, 8, 8)
        reference_conv = nn.Conv2d(
            64, 64, kernel_size=5, padding=2, groups=64, bias=True
        ).eval()
        with torch.inference_mode():
            reference = reference_conv(inputs)

        for requested in ("group16", "group32", "group64", "dense"):
            candidate = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=5, padding=2, groups=64, bias=True)
            ).eval()
            candidate[0].load_state_dict(reference_conv.state_dict())
            summary = module._rewrite_layout_depthwise_convs(
                candidate,
                requested=requested,
            )
            with torch.inference_mode():
                actual = candidate(inputs)
            torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)
            self.assertEqual(summary["target_count"], 1)
            self.assertEqual(summary["rewritten_count"], 1)
            self.assertEqual(candidate[0].groups, 64 // summary["modules"][0]["group_width"])


if __name__ == "__main__":
    unittest.main()
