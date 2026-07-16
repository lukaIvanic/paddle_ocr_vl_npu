"""Contract tests for the cleaned Experiment 08 runtime profile."""

from __future__ import annotations

import unittest
from dataclasses import fields

from engine import PrefilledRecognition
from run_offline_e2e import parse_args
from runtime_defaults import (
    DEFAULT_DECODE_BATCH_SIZE,
    DEFAULT_TEXT_BACKEND,
    DEFAULT_VISION_BACKEND,
    OPTIMIZED_TEXT_BUCKETS,
    OPTIMIZED_VISION_BUCKETS,
)
from text_compile import parse_text_buckets
from vision_compile import parse_vision_buckets


class RuntimeDefaultsTest(unittest.TestCase):
    def test_cli_defaults_to_the_validated_optimized_profile(self) -> None:
        args = parse_args(["--image", "unused-test-page.png"])

        self.assertEqual(args.batch_size, DEFAULT_DECODE_BATCH_SIZE)
        self.assertEqual(args.vision_backend, DEFAULT_VISION_BACKEND)
        self.assertEqual(args.text_backend, DEFAULT_TEXT_BACKEND)
        self.assertEqual(
            parse_vision_buckets(args.vision_compile_buckets),
            OPTIMIZED_VISION_BUCKETS,
        )
        self.assertEqual(
            parse_text_buckets(args.text_compile_buckets),
            OPTIMIZED_TEXT_BUCKETS,
        )

    def test_dense_bucket_ranges_match_the_measured_policy(self) -> None:
        self.assertEqual(OPTIMIZED_VISION_BUCKETS[:3], (32, 64, 96))
        self.assertEqual(OPTIMIZED_VISION_BUCKETS[-3:], (1792, 1920, 2048))
        self.assertEqual(len(OPTIMIZED_VISION_BUCKETS), 32)

    def test_text_buckets_match_measured_prompt_distribution(self) -> None:
        self.assertEqual(OPTIMIZED_TEXT_BUCKETS[:4], (32, 64, 96, 128))
        self.assertEqual(OPTIMIZED_TEXT_BUCKETS[4:9], (160, 176, 192, 208, 224))
        self.assertEqual(OPTIMIZED_TEXT_BUCKETS[-2:], (1280, 1312))

    def test_prefilled_state_does_not_retain_the_pil_request(self) -> None:
        self.assertNotIn("request", {field.name for field in fields(PrefilledRecognition)})


if __name__ == "__main__":
    unittest.main()
