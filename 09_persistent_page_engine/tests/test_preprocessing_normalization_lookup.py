"""Bitwise checks: faster normalization is not a different image recipe."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paddleocr_vl.model import preprocessing as p


def reference(array, cfg):
    array = array.astype(np.float32)
    if cfg["do_rescale"]:
        array = array * float(cfg["rescale_factor"])
    if cfg["do_normalize"]:
        array = (array - np.array(cfg["image_mean"], dtype=np.float32)) / np.array(cfg["image_std"], dtype=np.float32)
    return array


class NormalizationLookupTest(unittest.TestCase):
    def test_all_channel_values_and_fallbacks(self):
        cfg = p.load_preprocessor_config(Path("/does-not-exist"))
        image = np.repeat(np.arange(256, dtype=np.uint8).reshape(16, 16, 1), 3, axis=2)
        for mean, std in [([0.5] * 3, [0.5] * 3), ([0.1] * 3, [0.3] * 3),
                          ([0.1, 0.2, 0.3], [0.9, 0.8, 0.7])]:
            for rescale in (False, True):
                for normalize in (False, True):
                    cfg.update(image_mean=mean, image_std=std, do_rescale=rescale, do_normalize=normalize)
                    actual = p._normalize_image_array(image, cfg)
                    expected = reference(image, cfg)
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertTrue(np.array_equal(actual.view(np.uint32), expected.view(np.uint32)))

    def test_full_resize_and_patchification_are_bit_exact(self):
        cfg = p.load_preprocessor_config(Path("/does-not-exist"))
        cfg.update(min_pixels=28224, max_pixels=802816)
        rng = np.random.default_rng(1)
        for size in ((63, 117), (509, 811), (1081, 1973)):
            image = Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8))
            actual, grid = p.preprocess_pil_image(image, cfg)
            with patch.object(p, "_normalize_image_array", reference):
                expected, old_grid = p.preprocess_pil_image(image, cfg)
            self.assertTrue(torch.equal(grid, old_grid))
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(actual.dtype, torch.float32)

    def test_lookup_is_immutable_arithmetic_not_image_cache(self):
        table = p._uint8_normalization_table(True, 1 / 255, 0.5, 0.5)
        self.assertEqual(table.shape, (256,))
        self.assertFalse(table.flags.writeable)


if __name__ == "__main__":
    unittest.main()
