"""CPU tests for the packed text attention contract."""

import unittest

try:
    import torch
except ImportError:
    torch = None

from text_prefill_compile import build_packed_text_allowed_mask


@unittest.skipIf(torch is None, "CPU torch is required")
class PackedTextMaskTests(unittest.TestCase):
    def test_padding_rows_are_valid_and_isolated(self):
        segment_ids = torch.tensor([0, 0, 1, 1, -1, -1])
        local_positions = torch.tensor([0, 1, 0, 1, 0, 0])

        allowed = build_packed_text_allowed_mask(segment_ids, local_positions)

        self.assertTrue(bool(allowed.any(dim=-1).all()))
        self.assertEqual(
            allowed[:4, :4].tolist(),
            [
                [True, False, False, False],
                [True, True, False, False],
                [False, False, True, False],
                [False, False, True, True],
            ],
        )
        self.assertFalse(bool(allowed[:4, 4:].any()))
        self.assertFalse(bool(allowed[4:, :4].any()))
        self.assertTrue(bool(allowed[4:, 4:].all()))

    def test_unpadded_rows_remain_block_causal(self):
        segment_ids = torch.tensor([0, 0, 0])
        local_positions = torch.tensor([0, 1, 2])
        allowed = build_packed_text_allowed_mask(segment_ids, local_positions)
        self.assertTrue(torch.equal(allowed, torch.tril(torch.ones((3, 3), dtype=torch.bool))))

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            build_packed_text_allowed_mask(torch.tensor([0]), torch.tensor([0, 1]))


if __name__ == "__main__":
    unittest.main()
