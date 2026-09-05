"""Content-independent CPU algebra tests; NPU drift/performance tested separately."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paddleocr_vl.model.vision_prefill import PaddleOCRVisionEmbeddings


class LinearPatchTest(unittest.TestCase):
    def test_projection_and_position_embedding_algebra(self):
        torch.manual_seed(91)
        for channels, patch_size in ((1, 2), (3, 14)):
            model = PaddleOCRVisionEmbeddings(SimpleNamespace(
                hidden_size=16, image_size=patch_size*2,
                patch_size=patch_size, num_channels=channels)).double()
            original_weights = {k: v.clone() for k, v in model.state_dict().items()}
            for height, width in ((2, 3), (3, 2), (1, 7)):
                pixels = torch.randn(1, height*width, channels, patch_size, patch_size, dtype=torch.float64)
                grid = torch.tensor([[1, height, width]])
                model.linear_patch_projection = False
                expected = model(pixels, grid)
                model.linear_patch_projection = True
                actual = model(pixels, grid)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            self.assertEqual(set(model.state_dict()), set(original_weights))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, original_weights[key], rtol=0, atol=0)

    def test_rejects_non_patch_geometry(self):
        model = PaddleOCRVisionEmbeddings(SimpleNamespace(
            hidden_size=8, image_size=28, patch_size=14, num_channels=3))
        model.linear_patch_projection = True
        with self.assertRaises(ValueError):
            model.project_patches(torch.randn(2, 3, 28, 14))


if __name__ == "__main__":
    unittest.main()
