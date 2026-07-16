"""Small contract tests for the official-PaddleX integration boundary."""

from __future__ import annotations

import unittest

import numpy as np

from paddlex_adapter import PaddleXContinuousRecognizerAdapter
from run_omnidocbench_paddlex import parse_args, selected_text_buckets
from runtime_defaults import (
    OMNIDOCBENCH_CACHE_LENGTH,
    OMNIDOCBENCH_DECODE_BATCH_SIZE,
    OMNIDOCBENCH_MAX_NEW_TOKENS,
    OPTIMIZED_TEXT_BUCKETS,
)


class PaddleXAdapterTest(unittest.TestCase):
    def test_benchmark_cli_uses_the_named_production_profile(self) -> None:
        args = parse_args(["--output-dir", "unused-test-output"])

        self.assertEqual(args.batch_size, OMNIDOCBENCH_DECODE_BATCH_SIZE)
        self.assertEqual(args.cache_length, OMNIDOCBENCH_CACHE_LENGTH)
        self.assertEqual(args.max_new_tokens, OMNIDOCBENCH_MAX_NEW_TOKENS)
        self.assertEqual(selected_text_buckets(args), OPTIMIZED_TEXT_BUCKETS[4:])

    def test_request_conversion_preserves_order_and_converts_bgr_to_rgb(self) -> None:
        bgr = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)

        requests = PaddleXContinuousRecognizerAdapter._build_requests(
            [
                {"image": bgr, "query": "OCR:"},
                {"image": bgr, "query": "Table Recognition:"},
            ],
            batch_index=7,
            skip_special_tokens=False,
        )

        self.assertEqual(
            [request.request_id for request in requests],
            [
                "paddlex_batch_000007_item_000000",
                "paddlex_batch_000007_item_000001",
            ],
        )
        self.assertEqual([request.layout_order for request in requests], [0, 1])
        self.assertEqual([request.label for request in requests], ["text", "table"])
        self.assertEqual(requests[0].crop.getpixel((0, 0)), (3, 2, 1))
        self.assertFalse(requests[0].skip_special_tokens)

    def test_request_conversion_rejects_unknown_queries(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported PaddleX"):
            PaddleXContinuousRecognizerAdapter._build_requests(
                [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "query": "Nope:"}],
                batch_index=0,
                skip_special_tokens=True,
            )


if __name__ == "__main__":
    unittest.main()
