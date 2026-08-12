#!/usr/bin/env python3
"""CPU correctness checks for exact UniRec focal-depthwise rewrites."""

from __future__ import annotations

import copy
import sys
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from vision_focal_depthwise import (  # noqa: E402
    AlignedSpatialDepthwiseConv,
    ConstantFocalDepthwiseConv,
    grouped_fz_storage_shape,
    pack_grouped_fz_host,
    rewrite_vision_focal_depthwise_convs,
)


def _fake_vision_encoder() -> SimpleNamespace:
    stages = []
    for stage_index, (channels, depth) in enumerate(
        ((96, 2), (192, 2), (384, 9), (768, 2))
    ):
        blocks = []
        for _ in range(depth):
            focals = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Conv2d(
                            channels,
                            channels,
                            kernel_size=kernel,
                            padding=kernel // 2,
                            groups=channels,
                            bias=False,
                        ),
                        torch.nn.GELU(),
                    )
                    for kernel in (3, 5, 7)
                ]
            )
            blocks.append(
                SimpleNamespace(
                    modulation=SimpleNamespace(focal_layers=focals)
                )
            )
        stages.append(SimpleNamespace(blocks=blocks, stage=stage_index))
    return SimpleNamespace(layers=stages)


class VisionFocalDepthwiseRewriteTest(unittest.TestCase):
    def test_grouped_fz_storage_shape_matches_cann_filter_layout(self) -> None:
        self.assertEqual(
            grouped_fz_storage_shape((384, 1, 7, 7), groups=384),
            (1176, 1, 16, 16),
        )
        self.assertEqual(
            grouped_fz_storage_shape((192, 1, 5, 5), groups=192),
            (300, 1, 16, 16),
        )

    def test_native_depthwise_grouped_pack_places_each_filter_once(self) -> None:
        weight = torch.arange(32 * 1 * 5 * 5, dtype=torch.float16).reshape(
            32, 1, 5, 5
        ).numpy()
        packed = pack_grouped_fz_host(weight, groups=32)
        self.assertEqual(packed.shape, (50, 1, 16, 16))
        recovered = torch.zeros(32, 1, 5, 5, dtype=torch.float16).numpy()
        for output_channel in range(32):
            group_block = output_channel // 16
            group_lane = output_channel % 16
            for kernel_h in range(5):
                for kernel_w in range(5):
                    recovered[output_channel, 0, kernel_h, kernel_w] = packed[
                        group_block * 25 + kernel_h * 5 + kernel_w,
                        0,
                        group_lane,
                        group_lane,
                    ]
        self.assertTrue((recovered == weight).all())
        self.assertEqual(int((packed != 0).sum()), int((weight != 0).sum()))
    def test_constant_wrapper_matches_native_depthwise_convolution(self) -> None:
        torch.manual_seed(5)
        source = torch.nn.Conv2d(
            16,
            16,
            kernel_size=7,
            padding=3,
            groups=16,
            bias=False,
        ).eval()
        inputs = torch.randn(2, 16, 11, 13)
        wrapped = ConstantFocalDepthwiseConv(source, weight_id=10_000).eval()
        torch.testing.assert_close(
            wrapped(inputs),
            source(inputs),
            atol=1e-6,
            rtol=1e-5,
        )

    def test_prepacked_wrapper_keeps_physical_grouped_storage(self) -> None:
        torch.manual_seed(6)
        source = torch.nn.Conv2d(
            32,
            32,
            kernel_size=5,
            padding=2,
            groups=32,
            bias=False,
        ).eval()
        inputs = torch.randn(2, 32, 9, 11)
        wrapped = ConstantFocalDepthwiseConv(
            source,
            weight_id=10_001,
            prepack_grouped=True,
        ).eval()
        self.assertEqual(tuple(wrapped.packed_weight.shape), (50, 1, 16, 16))
        self.assertEqual(
            wrapped.packed_weight.untyped_storage().nbytes(),
            50 * 1 * 16 * 16 * 2,
        )
        torch.testing.assert_close(
            wrapped(inputs),
            source(inputs),
            atol=1e-6,
            rtol=1e-5,
        )

    def test_aligned_spatial_filters_are_exact(self) -> None:
        torch.manual_seed(7)
        for kernel in (5, 7):
            source = torch.nn.Conv2d(
                16,
                16,
                kernel_size=kernel,
                padding=kernel // 2,
                groups=16,
                bias=False,
            ).eval()
            inputs = torch.randn(2, 16, 11, 13)
            expected = source(inputs)
            rewritten = AlignedSpatialDepthwiseConv(source).eval()
            actual = rewritten(inputs)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            self.assertEqual(rewritten.weight.shape[-2] * rewritten.weight.shape[-1] % 16, 0)

    def test_group16_rewrite_is_exact_and_targets_only_stage2_and_stage3(self) -> None:
        torch.manual_seed(11)
        vision = _fake_vision_encoder()
        original = copy.deepcopy(
            vision.layers[2].blocks[0].modulation.focal_layers[2][0]
        )
        summary = rewrite_vision_focal_depthwise_convs(
            vision,
            requested="group16",
        )
        self.assertEqual(summary["target_count"], 22)
        self.assertEqual(summary["rewritten_count"], 22)
        self.assertEqual({row["stage"] for row in summary["modules"]}, {2, 3})
        rewritten = vision.layers[2].blocks[0].modulation.focal_layers[2][0]
        self.assertEqual(rewritten.groups, 24)
        self.assertEqual(tuple(rewritten.weight.shape), (384, 16, 7, 7))
        inputs = torch.randn(1, 384, 7, 9)
        torch.testing.assert_close(
            rewritten(inputs),
            original(inputs),
            atol=1e-6,
            rtol=1e-5,
        )

    def test_native_lane_records_targets_without_mutation(self) -> None:
        vision = _fake_vision_encoder()
        before = vision.layers[3].blocks[1].modulation.focal_layers[1][0]
        summary = rewrite_vision_focal_depthwise_convs(
            vision,
            requested="native",
        )
        after = vision.layers[3].blocks[1].modulation.focal_layers[1][0]
        self.assertIs(after, before)
        self.assertEqual(summary["target_count"], 22)
        self.assertEqual(summary["rewritten_count"], 0)

    def test_constant_grouped_all_targets_every_focal_depthwise_filter(self) -> None:
        vision = _fake_vision_encoder()
        with mock.patch(
            "vision_focal_depthwise.register_focal_depthwise_constant_converter"
        ):
            summary = rewrite_vision_focal_depthwise_convs(
                vision,
                requested="constant_grouped_all",
            )
        self.assertEqual(summary["target_count"], 45)
        self.assertEqual(summary["rewritten_count"], 45)
        self.assertEqual(
            {row["stage"] for row in summary["modules"]},
            {0, 1, 2, 3},
        )
        self.assertEqual(
            {tuple(row["source_kernel"]) for row in summary["modules"]},
            {(3, 3), (5, 5), (7, 7)},
        )
        self.assertTrue(
            all(
                row["weight_binding"]
                == "frozen_prepacked_fractal_z_grouped"
                for row in summary["modules"]
            )
        )


if __name__ == "__main__":
    unittest.main()
