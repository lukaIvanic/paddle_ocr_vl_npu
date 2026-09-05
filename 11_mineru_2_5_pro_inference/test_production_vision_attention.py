import sys
import types
import unittest
from unittest.mock import patch

import torch

from bench_production_vision_attention import mask_segments, unpad_attention


def mask_for(lengths):
    labels = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
    return (labels[:, None] != labels[None, :])[None, None]


class AttentionContracts(unittest.TestCase):
    def test_lengths_include_shared_padding_component(self):
        self.assertEqual(mask_segments(mask_for([480, 192, 96])), [480, 192, 96])
        self.assertEqual(mask_segments(mask_for([5476, 156])), [5476, 156])

    def test_reject_fully_masked_row_and_cross_component_leak(self):
        mask = mask_for([3, 2])
        mask[..., 1, :] = True
        with self.assertRaises(ValueError):
            mask_segments(mask)
        mask = mask_for([3, 2])
        mask[..., 4, 0] = False
        with self.assertRaises(ValueError):
            mask_segments(mask)

    def test_unpad_repacking_matches_masked_attention(self):
        torch.manual_seed(7)
        lengths = [3, 2, 1]
        q, k, v = [torch.randn(1, 2, 6, 8) for _ in range(3)]
        mask = mask_for(lengths)
        scale = 8 ** -.5
        expected = torch.softmax((q @ k.transpose(-1, -2) * scale).masked_fill(mask, -float('inf')), -1) @ v

        def fake_unpad(*, query, key, value, seq_len, scale_value, num_heads, num_kv_heads, out):
            self.assertEqual(seq_len.device.type, 'cpu')
            self.assertEqual(seq_len.dtype, torch.int32)
            self.assertEqual(num_heads, num_kv_heads)
            start = 0
            for length in seq_len.tolist():
                end = start + length
                a, b, c = [t[start:end].transpose(0, 1) for t in (query, key, value)]
                out[start:end] = (torch.softmax(a @ b.transpose(-1, -2) * scale_value, -1) @ c).transpose(0, 1)
                start = end

        with patch.dict(sys.modules, {'torch_npu': types.SimpleNamespace(_npu_flash_attention_unpad=fake_unpad)}):
            actual = unpad_attention(lengths)(q, k, v, num_heads=2, scale=scale, atten_mask=mask, sparse_mode=1)
        torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
