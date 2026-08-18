from __future__ import annotations

import unittest

from optimize_vision_bucket_frontier import make_variants


class OptimizeVisionBucketFrontierTest(unittest.TestCase):
    def test_candidate_generation_rejects_partial_final_stage_tiles(self) -> None:
        pages = [[(960, 448), (960, 512), (448, 192)]]
        variants, rejected, adjustments = make_variants(
            pages,
            [64 * 64, 960 * 512 * 4],
            [1.0, 10.0],
        )
        keys = {variant.key for variant in variants}

        self.assertNotIn("960x448_b1", keys)
        self.assertIn("960x512_b1", keys)
        self.assertNotIn("448x192_b2", keys)
        self.assertIn("512x192_b2", keys)
        self.assertIn("1024x448_b1", keys)
        self.assertIn("960x448_b1", rejected)
        self.assertIn("448x192_b2", rejected)
        replacements = {
            row["source_key"]: row["replacement_keys"] for row in adjustments
        }
        self.assertEqual(
            replacements["960x448_b1"],
            ["1024x448_b1"],
        )
        self.assertEqual(
            replacements["448x192_b2"],
            ["512x192_b2"],
        )
        self.assertTrue(
            all(
                (
                    variant.batch_size
                    * (variant.width // 32)
                    * (variant.height // 32)
                )
                % 16
                == 0
                for variant in variants
            )
        )


if __name__ == "__main__":
    unittest.main()
