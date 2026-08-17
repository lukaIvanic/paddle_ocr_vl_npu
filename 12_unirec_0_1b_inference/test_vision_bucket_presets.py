from __future__ import annotations

import unittest

from vision_bucket_presets import (
    VISION_BUCKETS_310P_K10_L1,
    VisionBucketSpec,
    plan_canvas_bucket_calls,
    resolve_vision_bucket_specs,
)


class VisionBucketPresetTest(unittest.TestCase):
    def test_k10_has_ten_unique_graph_variants(self) -> None:
        specs = resolve_vision_bucket_specs("310p_k10_l1")
        self.assertEqual(len(specs), 10)
        self.assertEqual(len({spec.key for spec in specs}), 10)
        self.assertEqual(specs, VISION_BUCKETS_310P_K10_L1)

    def test_same_canvas_planner_uses_b2_and_b4(self) -> None:
        specs = tuple(
            spec
            for spec in VISION_BUCKETS_310P_K10_L1
            if (spec.width, spec.height) == (960, 64)
        )
        expected = {
            1: [2],
            2: [2],
            3: [4],
            4: [4],
            5: [2, 4],
            6: [2, 4],
            7: [4, 4],
            8: [4, 4],
        }
        for real_rows, batches in expected.items():
            planned = plan_canvas_bucket_calls(specs, real_rows)
            self.assertEqual(
                sorted(spec.batch_size for spec in planned),
                batches,
            )
            self.assertGreaterEqual(
                sum(spec.batch_size for spec in planned), real_rows
            )

    def test_single_variant_repeats_and_pads_tail(self) -> None:
        spec = VisionBucketSpec(960, 128, 1)
        self.assertEqual(plan_canvas_bucket_calls((spec,), 3), (spec, spec, spec))
        b4 = VisionBucketSpec(448, 64, 4)
        self.assertEqual(plan_canvas_bucket_calls((b4,), 5), (b4, b4))

    def test_multiple_variants_require_costs(self) -> None:
        with self.assertRaisesRegex(ValueError, "require planning costs"):
            plan_canvas_bucket_calls(
                (VisionBucketSpec(64, 64, 1), VisionBucketSpec(64, 64, 2)),
                2,
            )


if __name__ == "__main__":
    unittest.main()
