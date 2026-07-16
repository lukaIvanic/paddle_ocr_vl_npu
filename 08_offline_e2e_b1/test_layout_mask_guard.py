"""Contract tests for the narrow PP-DocLayout empty-mask compatibility guard."""

from __future__ import annotations

import unittest

import numpy as np

from layout_mask_guard import find_empty_mask_crops, install_layout_mask_guard
from run_with_layout_mask_guard import parse_args


class FakeProcessor:
    calls: list[list[list[float]]] = []

    def _extract_polygon_points_by_masks(self, boxes, masks, scale_ratio):
        type(self).calls.append(np.asarray(boxes).tolist())
        return [
            np.array([[100 + index, 200 + index]], dtype=np.float32)
            for index in range(len(boxes))
        ]


class LayoutMaskGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeProcessor.calls = []

    def test_finds_positive_box_that_collapses_at_mask_resolution(self) -> None:
        boxes = np.array([[1.0, 1.0, 2.0, 10.0]], dtype=np.float32)
        masks = np.ones((1, 20, 20), dtype=np.float32)

        records = find_empty_mask_crops(boxes, masks, (0.4, 0.4))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["scaled_x_clipped"], [0, 0])
        self.assertEqual(records[0]["box_int"], [1, 1, 2, 10])

    def test_guard_preserves_valid_original_and_rect_falls_back_only_invalid(self) -> None:
        boxes = np.array(
            [[1.0, 1.0, 2.0, 10.0], [10.0, 10.0, 20.0, 20.0]],
            dtype=np.float32,
        )
        masks = np.ones((2, 20, 20), dtype=np.float32)
        state = install_layout_mask_guard(FakeProcessor)

        polygons = FakeProcessor()._extract_polygon_points_by_masks(
            boxes, masks, (0.4, 0.4)
        )

        np.testing.assert_array_equal(
            polygons[0],
            np.array([[1, 1], [2, 1], [2, 10], [1, 10]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            polygons[1],
            np.array([[100, 200]], dtype=np.float32),
        )
        self.assertEqual(FakeProcessor.calls, [[[10.0, 10.0, 20.0, 20.0]]])
        self.assertEqual(state.snapshot()["fallback_regions"], 1)

    def test_install_is_idempotent(self) -> None:
        first = install_layout_mask_guard(FakeProcessor)
        second = install_layout_mask_guard(FakeProcessor)
        self.assertIs(first, second)

    def test_stock_wrapper_preserves_runner_arguments(self) -> None:
        args = parse_args(
            [
                "--guard-report",
                "guard.json",
                "stock_runner.py",
                "--offset",
                "12",
                "--limit",
                "34",
            ]
        )
        self.assertEqual(args.runner_args, ["--offset", "12", "--limit", "34"])


if __name__ == "__main__":
    unittest.main()
