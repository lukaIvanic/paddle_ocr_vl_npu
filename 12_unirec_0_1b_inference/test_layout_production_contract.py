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


class LayoutProductionContractTest(unittest.TestCase):
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
