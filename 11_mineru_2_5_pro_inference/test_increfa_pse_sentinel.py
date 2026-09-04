"""CPU contract tests for the 310P IncreFA exact-tile workaround."""

import unittest

import torch

from local_modeling_mineru import (
    apply_increfa_310p_pse_sentinel,
    build_static_decode_mask,
    increfa_310p_inner_tile_size,
)


class IncreFAPseSentinelTests(unittest.TestCase):
    def test_public_310p_tiler_reproduces_paddle_and_mineru_boundaries(self):
        self.assertEqual(
            increfa_310p_inner_tile_size(
                num_attention_heads=16,
                num_key_value_heads=2,
            ),
            1280,
        )
        self.assertEqual(
            increfa_310p_inner_tile_size(
                num_attention_heads=14,
                num_key_value_heads=2,
            ),
            1408,
        )

    def test_mineru_sentinel_changes_only_exact_tile_boundaries(self):
        inputs = torch.zeros((4, 1, 64), dtype=torch.float16)
        positions = torch.tensor([1406, 1407, 1408, 2815], dtype=torch.int64)
        mask = build_static_decode_mask(inputs, positions, 4096)
        adjusted, pse_shift, tile_size = apply_increfa_310p_pse_sentinel(
            mask,
            inputs,
            positions,
            cache_length=4096,
            num_attention_heads=14,
            num_key_value_heads=2,
        )
        self.assertEqual(tile_size, 1408)
        sentinel_rows = (1, 3)
        for row in range(4):
            sentinel_position = int(positions[row]) + 1
            if row in sentinel_rows:
                self.assertEqual(float(adjusted[row, 0, 0, sentinel_position]), 0.0)
                self.assertEqual(
                    float(pse_shift[row, 0, 0, sentinel_position]),
                    float(torch.finfo(torch.float16).min),
                )
            else:
                self.assertEqual(
                    float(adjusted[row, 0, 0, sentinel_position]),
                    float(torch.finfo(torch.float16).min),
                )
                self.assertEqual(float(pse_shift[row, 0, 0, sentinel_position]), 0.0)

    def test_full_cache_does_not_create_out_of_range_sentinel(self):
        inputs = torch.zeros((1, 1, 64), dtype=torch.float16)
        positions = torch.tensor([4095], dtype=torch.int64)
        mask = build_static_decode_mask(inputs, positions, 4096)
        adjusted, pse_shift, _tile_size = apply_increfa_310p_pse_sentinel(
            mask,
            inputs,
            positions,
            cache_length=4096,
            num_attention_heads=14,
            num_key_value_heads=2,
        )
        self.assertTrue(torch.equal(adjusted, mask))
        self.assertEqual(int(torch.count_nonzero(pse_shift)), 0)


if __name__ == "__main__":
    unittest.main()
