#!/usr/bin/env python3
"""CPU checks for the canonical-RGB production layout frontend."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from layout_process_pool import _crop_margin_rgb  # noqa: E402


def _legacy_crop_margin_bgr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    maximum = gray.max()
    minimum = gray.min()
    if maximum == minimum:
        return image
    data = ((gray - minimum) / (maximum - minimum) * 255).astype(np.uint8)
    _, binary = cv2.threshold(data, 200, 255, cv2.THRESH_BINARY_INV)
    coordinates = cv2.findNonZero(binary)
    if coordinates is None:
        return image
    x, y, width, height = cv2.boundingRect(coordinates)
    return image[y : y + height, x : x + width]


class LayoutRgbFrontendTest(unittest.TestCase):
    def test_rgb_formula_margin_matches_legacy_bgr_geometry_and_pixels(self) -> None:
        rgb = np.full((32, 40, 3), 255, dtype=np.uint8)
        rgb[7:25, 9:31] = np.array([20, 80, 160], dtype=np.uint8)
        rgb[12:19, 15:24] = np.array([230, 210, 190], dtype=np.uint8)
        bgr = np.ascontiguousarray(rgb[..., ::-1])

        legacy_rgb = np.ascontiguousarray(
            _legacy_crop_margin_bgr(bgr)[..., ::-1]
        )
        direct_rgb = np.ascontiguousarray(_crop_margin_rgb(rgb))

        np.testing.assert_array_equal(direct_rgb, legacy_rgb)


if __name__ == "__main__":
    unittest.main()
