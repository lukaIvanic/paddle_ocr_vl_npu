from __future__ import annotations

import ast
import unittest
from pathlib import Path

from vision_bucket_presets import (
    VISION_BUCKETS_310P_K10_L1,
    VISION_BUCKETS_310P_K10_L4_ALL,
    VisionBucketSpec,
    plan_canvas_bucket_calls,
    resolve_vision_bucket_specs,
)


class VisionBucketPresetTest(unittest.TestCase):
    def test_full_vision_has_ten_static_bucket_code_objects(self) -> None:
        source = Path(__file__).with_name("vision_full_batch.py").read_text()
        tree = ast.parse(source)
        encoder = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "_MaskedFullVisionEncoder"
        )
        slots = {
            node.name
            for node in encoder.body
            if isinstance(node, ast.FunctionDef)
            and node.name.startswith("_forward_bucket_slot_")
        }
        self.assertEqual(
            slots,
            {f"_forward_bucket_slot_{index}" for index in range(10)},
        )
        factory = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_new_masked_full_encoder_module"
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "exec"
                for node in ast.walk(factory)
            )
        )

    def test_worker_setup_warms_eager_fallback_twice(self) -> None:
        source = Path(__file__).with_name("layout_process_pool.py").read_text()
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "warmup_eager_fallback"
        ]
        self.assertEqual(len(calls), 1)
        passes = next(
            keyword.value
            for keyword in calls[0].keywords
            if keyword.arg == "passes"
        )
        self.assertIsInstance(passes, ast.Constant)
        self.assertEqual(passes.value, 2)

    def test_k10_has_ten_unique_graph_variants(self) -> None:
        specs = resolve_vision_bucket_specs("310p_k10_l1")
        self.assertEqual(len(specs), 10)
        self.assertEqual(len({spec.key for spec in specs}), 10)
        self.assertEqual(specs, VISION_BUCKETS_310P_K10_L1)

    def test_k10_l4_all_has_ten_unique_graph_variants(self) -> None:
        specs = resolve_vision_bucket_specs("310p_k10_l4_all")
        self.assertEqual(len(specs), 10)
        self.assertEqual(len({spec.key for spec in specs}), 10)
        self.assertEqual(specs, VISION_BUCKETS_310P_K10_L4_ALL)
        self.assertTrue(any(spec.height == 1408 for spec in specs))
        for width, height in ((64, 1408), (448, 1152), (896, 576)):
            self.assertTrue(any(spec.accepts(width, height) for spec in specs))

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

    def test_final_stage_row_alignment_identifies_310p_tail_shapes(self) -> None:
        cases = {
            (960, 448, 1): False,
            (960, 512, 1): True,
            (960, 64, 2): False,
            (960, 64, 4): True,
            (448, 192, 2): False,
            (448, 192, 4): True,
        }
        for (width, height, batch_size), expected in cases.items():
            spec = VisionBucketSpec(width, height, batch_size)
            self.assertEqual(
                spec.has_aligned_final_stage_rows,
                expected,
                spec.key,
            )

    def test_multiple_variants_require_costs(self) -> None:
        with self.assertRaisesRegex(ValueError, "require planning costs"):
            plan_canvas_bucket_calls(
                (VisionBucketSpec(64, 64, 1), VisionBucketSpec(64, 64, 2)),
                2,
            )


if __name__ == "__main__":
    unittest.main()
