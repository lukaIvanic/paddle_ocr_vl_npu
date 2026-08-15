#!/usr/bin/env python3
"""CPU-only checks for the PP-DocLayoutV2 NPU compatibility bindings."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
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


class _HybridEncoderClass(nn.Module):
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
    layout_module.PPDocLayoutV2HybridEncoder = _HybridEncoderClass
    layout_module.PPDocLayoutV2ForObjectDetectionOutput = SimpleNamespace
    layout_module.BaseModelOutput = SimpleNamespace
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


class PPDocLayoutV2FrozenBatchNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.randn(channels))
        self.register_buffer("bias", torch.randn(channels))
        self.register_buffer("running_mean", torch.randn(channels))
        self.register_buffer("running_var", torch.rand(channels) + 0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = self.weight.reshape(1, -1, 1, 1) * (
            self.running_var.reshape(1, -1, 1, 1) + 1e-5
        ).rsqrt()
        bias = self.bias.reshape(1, -1, 1, 1)
        bias = bias - self.running_mean.reshape(1, -1, 1, 1) * scale
        return inputs * scale + bias


class _ConvFrozenNorm(nn.Module):
    def __init__(self, channels: int, *, conv_bias: bool) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=conv_bias
        )
        self.normalization = PPDocLayoutV2FrozenBatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.normalization(self.convolution(inputs))


class _ConvEvalNorm(nn.Module):
    def __init__(self, channels: int, *, affine: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(channels, eps=1e-3, affine=affine)
        self.norm.running_mean.copy_(torch.randn(channels))
        self.norm.running_var.copy_(torch.rand(channels) + 0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(inputs))


class LayoutNpuCompatibilityTest(unittest.TestCase):
    def test_index_free_nearest_upsample_is_bit_exact(self) -> None:
        layout_torchair = _load_layout_torchair()
        torch.manual_seed(20260813)
        inputs = torch.randn(2, 7, 11, 13, dtype=torch.float16)

        reference = torch.nn.functional.interpolate(
            inputs,
            scale_factor=2.0,
            mode="nearest",
        )
        actual = layout_torchair._nearest_upsample2d_2x_exact(inputs)

        torch.testing.assert_close(actual, reference, atol=0.0, rtol=0.0)
        self.assertEqual(actual.stride(), reference.stride())

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

    def test_explicit_layout_preprocess_is_contiguous_bchw(self) -> None:
        module = _load_layout_adapter()
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as tv_functional

        rng = np.random.default_rng(44)
        image = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
        resized_uint8 = module.prepare_layout_resized_uint8_exact([image])[
            "pixel_values"
        ]
        actual = module.prepare_layout_pixel_values_exact([image])["pixel_values"]
        reference = torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0)
        reference = tv_functional.resize(
            reference,
            [800, 800],
            interpolation=InterpolationMode.BICUBIC,
            antialias=False,
        ).to(torch.float32).div_(255.0)

        torch.testing.assert_close(actual, reference, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            resized_uint8.to(torch.float32).div_(255.0),
            reference,
            atol=0.0,
            rtol=0.0,
        )
        self.assertEqual(resized_uint8.dtype, torch.uint8)
        self.assertEqual(
            resized_uint8.numel() * resized_uint8.element_size(),
            1_920_000,
        )
        self.assertEqual(actual.shape, (1, 3, 800, 800))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.stride(), (1_920_000, 640_000, 800, 1))

    def test_explicit_layout_preprocess_skips_cat_for_batch_one(self) -> None:
        module = _load_layout_adapter()
        image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)

        with mock.patch.object(module.torch, "cat", wraps=module.torch.cat) as cat:
            pixel_values = module.prepare_layout_resized_uint8_exact([image])[
                "pixel_values"
            ]

        self.assertEqual(tuple(pixel_values.shape), (1, 3, 800, 800))
        self.assertTrue(pixel_values.is_contiguous())
        cat.assert_not_called()

    def test_explicit_layout_preprocess_concatenates_larger_batches(self) -> None:
        module = _load_layout_adapter()
        first = np.zeros((3, 3, 3), dtype=np.uint8)
        second = np.full((3, 3, 3), 255, dtype=np.uint8)

        with mock.patch.object(module.torch, "cat", wraps=module.torch.cat) as cat:
            pixel_values = module.prepare_layout_resized_uint8_exact(
                [first, second]
            )["pixel_values"]

        self.assertEqual(tuple(pixel_values.shape), (2, 3, 800, 800))
        self.assertTrue(pixel_values.is_contiguous())
        cat.assert_called_once()

    def test_fast_layout_postprocess_matches_transformers_math(self) -> None:
        module = _load_layout_adapter()
        torch.manual_seed(45)
        outputs = SimpleNamespace(
            logits=torch.randn(2, 17, 5, dtype=torch.float16),
            pred_boxes=torch.rand(2, 17, 4, dtype=torch.float16),
            order_logits=torch.randn(2, 17, 17, dtype=torch.float16),
        )
        target_sizes = [(1536, 1024), (2200, 1700)]

        order_scores = torch.sigmoid(outputs.order_logits)
        order_votes = order_scores.triu(diagonal=1).sum(dim=1) + (
            1.0 - order_scores.transpose(1, 2)
        ).tril(diagonal=-1).sum(dim=1)
        order_pointers = torch.argsort(order_votes, dim=1)
        order_sequences = torch.empty_like(order_pointers)
        ranks = torch.arange(17, dtype=order_pointers.dtype).expand(2, -1)
        order_sequences.scatter_(1, order_pointers, ranks)

        centers, dimensions = torch.split(outputs.pred_boxes, 2, dim=-1)
        boxes = torch.cat(
            [centers - 0.5 * dimensions, centers + 0.5 * dimensions],
            dim=-1,
        )
        heights, widths = torch.as_tensor(target_sizes).unbind(1)
        scale = torch.stack([widths, heights, widths, heights], dim=1)
        boxes = boxes * scale[:, None, :]
        scores, flat_indices = torch.topk(
            torch.sigmoid(outputs.logits).flatten(1),
            17,
            dim=-1,
        )
        labels = flat_indices % 5
        query_indices = flat_indices // 5
        boxes = boxes.gather(
            1,
            query_indices.unsqueeze(-1).repeat(1, 1, 4),
        )
        order_sequences = order_sequences.gather(1, query_indices)
        reference = []
        for score, label, box, order in zip(
            scores,
            labels,
            boxes,
            order_sequences,
        ):
            keep = score >= 0.4
            order = order[keep]
            order, indices = torch.sort(order)
            reference.append(
                {
                    "scores": score[keep][indices],
                    "labels": label[keep][indices],
                    "boxes": box[keep][indices],
                    "order_seq": order,
                }
            )

        timing_s = {}
        actual = module.post_process_layout_object_detection_exact(
            outputs,
            threshold=0.4,
            target_sizes=target_sizes,
            timing_s=timing_s,
        )
        for expected, candidate in zip(reference, actual):
            for name in expected:
                torch.testing.assert_close(
                    candidate[name],
                    expected[name],
                    atol=0.0,
                    rtol=0.0,
                )
        self.assertEqual(
            set(timing_s),
            {
                "box_order_sigmoid_s",
                "box_order_votes_s",
                "box_order_rank_s",
                "box_xyxy_scale_s",
                "box_class_topk_s",
                "box_gather_s",
                "box_filter_sort_s",
            },
        )
        self.assertTrue(all(value >= 0.0 for value in timing_s.values()))

    def test_native_depthwise_rewrite_is_a_noop(self) -> None:
        module = _load_layout_adapter()
        candidate = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=2, groups=64, bias=True)
        ).eval()
        original = candidate[0]
        summary = module._rewrite_layout_depthwise_convs(
            candidate,
            requested="native",
        )
        self.assertIs(candidate[0], original)
        self.assertEqual(
            summary,
            {
                "requested": "native",
                "target_count": 0,
                "rewritten_count": 0,
                "modules": [],
            },
        )

    def test_constant_grouped_rewrites_all_native_layout_depthwise_convs(self) -> None:
        module = _load_layout_adapter()
        torch.manual_seed(23)
        candidate = nn.Sequential(
            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=16,
                bias=False,
            ),
            nn.Conv2d(
                16,
                16,
                kernel_size=5,
                padding=2,
                groups=16,
                bias=False,
            ),
        ).eval()
        reference = copy.deepcopy(candidate)
        inputs = torch.randn(2, 16, 17, 19)

        with mock.patch.object(
            module,
            "register_focal_depthwise_constant_converter",
        ):
            summary = module._rewrite_layout_depthwise_convs(
                candidate,
                requested="constant_grouped",
            )

        with torch.inference_mode():
            expected = reference(inputs)
            actual = candidate(inputs)
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
        self.assertEqual(summary["target_count"], 2)
        self.assertEqual(summary["rewritten_count"], 2)
        self.assertEqual(
            [row["kernel"] for row in summary["modules"]],
            [[3, 3], [5, 5]],
        )
        self.assertEqual(
            [row["stride"] for row in summary["modules"]],
            [[2, 2], [1, 1]],
        )
        self.assertTrue(
            all(
                row["weight_binding"]
                == "frozen_prepacked_fractal_z_grouped"
                for row in summary["modules"]
            )
        )

    def test_adapter_defaults_to_torchair_internal_weights(self) -> None:
        module = _load_layout_adapter()
        default = inspect.signature(
            module.PPDocLayoutV2NpuAdapter
        ).parameters["weight_format"].default
        self.assertEqual(default, module.DEFAULT_LAYOUT_WEIGHT_FORMAT)
        self.assertEqual(default, "torchair_internal")
        self.assertEqual(
            module.LAYOUT_WEIGHT_FORMAT_CHOICES,
            ("native", "torchair_internal"),
        )
        self.assertEqual(
            module.LAYOUT_DEPTHWISE_REWRITE_CHOICES,
            ("native", "constant_grouped"),
        )

    def test_frozen_batch_norm_folding_matches_unfused_module(self) -> None:
        module = _load_layout_adapter()
        torch.manual_seed(29)
        inputs = torch.randn(2, 16, 9, 11)
        for conv_bias in (False, True):
            candidate = nn.Sequential(
                _ConvFrozenNorm(16, conv_bias=conv_bias)
            ).eval()
            with torch.inference_mode():
                reference = candidate(inputs)
            summary = module._fuse_layout_frozen_batch_norms(candidate)
            with torch.inference_mode():
                actual = candidate(inputs)
            torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
            self.assertEqual(summary["fused_count"], 1)
            self.assertIsInstance(candidate[0].normalization, nn.Identity)
            self.assertIsNotNone(candidate[0].convolution.bias)

    def test_precomputed_frozen_batch_norm_affine_matches_unfused(self) -> None:
        module = _load_layout_adapter()
        torch.manual_seed(30)
        candidate = nn.Sequential(_ConvFrozenNorm(16, conv_bias=False)).eval()
        inputs = torch.randn(2, 16, 9, 11)
        with torch.inference_mode():
            reference = candidate(inputs)
        summary = module._precompute_layout_frozen_bn_affines(
            candidate,
            preformat_nc1hwc0=False,
        )
        with torch.inference_mode():
            actual = candidate(inputs)
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        self.assertEqual(summary["replaced_count"], 1)
        self.assertIsInstance(
            candidate[0].normalization,
            module._PrecomputedLayoutAffine2d,
        )

    def test_eval_batch_norm_folding_matches_unfused_module(self) -> None:
        module = _load_layout_adapter()
        torch.manual_seed(31)
        inputs = torch.randn(2, 16, 9, 11)
        for affine in (False, True):
            candidate = nn.Sequential(_ConvEvalNorm(16, affine=affine)).eval()
            with torch.inference_mode():
                reference = candidate(inputs)
            summary = module._fuse_layout_eval_batch_norms(candidate)
            with torch.inference_mode():
                actual = candidate(inputs)
            torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
            self.assertEqual(summary["fused_count"], 1)
            self.assertIsInstance(candidate[0].norm, nn.Identity)
            self.assertIsNotNone(candidate[0].conv.bias)


if __name__ == "__main__":
    unittest.main()
