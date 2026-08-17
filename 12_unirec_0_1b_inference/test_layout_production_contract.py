from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from layout_detector_lab import (  # noqa: E402
    CURRENT_PRODUCTION_CONTRACT,
    _resolve_contract,
    parse_args,
)
from layout_page_input import materialize_layout_bgr  # noqa: E402
from layout_page_input import materialize_layout_rgb  # noqa: E402
from layout_msda_aclnn import (  # noqa: E402
    STATIC_LAYOUT_LEVEL_CUMSUM,
    STATIC_LAYOUT_LEVEL_PRODUCTS,
    STATIC_LAYOUT_SPATIAL_SHAPES,
    _build_310p_descriptor_bridge_shapes,
    _uses_310p_internal_layout,
)


class LayoutProductionContractTest(unittest.TestCase):
    def test_static_layout_metadata_matches_800px_feature_pyramid(self) -> None:
        self.assertEqual(
            tuple(height * width for height, width in STATIC_LAYOUT_SPATIAL_SHAPES),
            STATIC_LAYOUT_LEVEL_PRODUCTS,
        )
        running = 0
        cumulative = []
        for product in STATIC_LAYOUT_LEVEL_PRODUCTS:
            running += product
            cumulative.append(running)
        self.assertEqual(tuple(cumulative), STATIC_LAYOUT_LEVEL_CUMSUM)
        self.assertEqual((0, *STATIC_LAYOUT_LEVEL_CUMSUM[:-1]), (0, 10000, 12500))

    def test_msda_internal_layout_is_310p_only(self) -> None:
        self.assertTrue(_uses_310p_internal_layout("Ascend310P3"))
        self.assertTrue(_uses_310p_internal_layout("ascend310p"))
        self.assertFalse(_uses_310p_internal_layout("Ascend910B2"))

    def test_310p_msda_descriptor_bridge_preserves_allocation_size(self) -> None:
        infer_location, internal_output = _build_310p_descriptor_bridge_shapes(
            (1, 13125, 8, 32),
            (1, 300, 8, 3, 4, 2),
            (1, 300, 256),
        )
        self.assertEqual(infer_location, (1, 8, 3, 4, 2, 300))
        self.assertEqual(internal_output, (1, 256, 300))
        original_location_numel = 1 * 300 * 8 * 3 * 4 * 2
        infer_location_numel = 1
        for dim in infer_location:
            infer_location_numel *= dim
        self.assertEqual(infer_location_numel, original_location_numel)
        self.assertEqual(1 * 256 * 300, 1 * 300 * 256)
        # Reproduce the installed broken callback's transposed branch:
        # [value.B, location[-1], location.H * value.D]. The bridge makes its
        # wrong indexing produce the correct public allocation descriptor.
        broken_host_infer_output = (
            1,
            infer_location[5],
            infer_location[1] * 32,
        )
        self.assertEqual(broken_host_infer_output, (1, 300, 256))

    def test_310p_msda_descriptor_bridge_rejects_inconsistent_shapes(self) -> None:
        with self.assertRaises(RuntimeError):
            _build_310p_descriptor_bridge_shapes(
                (1, 13125, 8, 32),
                (1, 300, 8, 3, 4, 2),
                (1, 300, 128),
            )

    def test_production_contract_is_the_lab_default(self) -> None:
        argv = [
            "layout_detector_lab.py",
            "--openocr-root",
            "/tmp/openocr",
            "--output",
            "/tmp/result.json",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.contract, "current_production")
        self.assertEqual(
            {name: getattr(args, name) for name in CURRENT_PRODUCTION_CONTRACT},
            CURRENT_PRODUCTION_CONTRACT,
        )
        self.assertEqual(args.depthwise_rewrite, "constant_grouped")

    def test_production_contract_fills_every_model_setting(self) -> None:
        args = SimpleNamespace(
            contract="current_production",
            **{name: None for name in CURRENT_PRODUCTION_CONTRACT},
        )
        _resolve_contract(argparse.ArgumentParser(), args)
        self.assertEqual(
            {name: getattr(args, name) for name in CURRENT_PRODUCTION_CONTRACT},
            CURRENT_PRODUCTION_CONTRACT,
        )

    def test_production_contract_rejects_drift(self) -> None:
        values = {name: None for name in CURRENT_PRODUCTION_CONTRACT}
        values["dtype"] = "float32"
        args = SimpleNamespace(contract="current_production", **values)
        with self.assertRaises(SystemExit):
            _resolve_contract(argparse.ArgumentParser(), args)

    def test_shared_bgr_materialization_is_exact_and_contiguous(self) -> None:
        rgb = np.array(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            dtype=np.uint8,
        )
        bgr = materialize_layout_bgr(rgb)
        np.testing.assert_array_equal(bgr, rgb[..., ::-1])
        self.assertTrue(bgr.flags.c_contiguous)
        self.assertFalse(np.shares_memory(rgb, bgr))

    def test_shared_rgb_materialization_reuses_contiguous_decoder_output(self) -> None:
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        materialized = materialize_layout_rgb(rgb)
        self.assertIs(materialized, rgb)
        self.assertTrue(materialized.flags.c_contiguous)


if __name__ == "__main__":
    unittest.main()
