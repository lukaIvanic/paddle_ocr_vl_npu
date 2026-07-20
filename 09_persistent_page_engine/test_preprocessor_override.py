#!/usr/bin/env python3
"""CPU-only checks for the Experiment 09 min_pixels override."""

from __future__ import annotations

import unittest

from preprocessing import apply_min_pixels_override


class MinPixelsOverrideTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model_config = {
            "min_pixels": 112896,
            "max_pixels": 1003520,
            "patch_size": 14,
            "merge_size": 2,
        }

    def test_none_preserves_values_without_aliasing(self) -> None:
        effective = apply_min_pixels_override(self.model_config, None)
        self.assertEqual(effective, self.model_config)
        self.assertIsNot(effective, self.model_config)

    def test_half_override_changes_only_min_pixels(self) -> None:
        effective = apply_min_pixels_override(self.model_config, 56448)
        self.assertEqual(effective["min_pixels"], 56448)
        self.assertEqual(effective["max_pixels"], 1003520)
        self.assertEqual(effective["patch_size"], 14)
        self.assertEqual(effective["merge_size"], 2)
        self.assertEqual(self.model_config["min_pixels"], 112896)

    def test_nonpositive_override_is_rejected(self) -> None:
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                apply_min_pixels_override(self.model_config, value)

    def test_override_above_max_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_min_pixels_override(self.model_config, 1003521)


if __name__ == "__main__":
    unittest.main()
