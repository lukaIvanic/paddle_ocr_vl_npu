from __future__ import annotations

import ast
import unittest
from pathlib import Path

from vision_bucket_presets import (
    VISION_BUCKETS_310P_K10_L1,
    VISION_BUCKETS_310P_K10_L4_ALL,
    VISION_BUCKETS_310P_K10_L4_ALIGNED,
    VISION_BUCKETS_310P_K20_L4,
    VisionBucketSpec,
    assign_vision_bucket_cache_slots,
    plan_canvas_bucket_calls,
    resolve_vision_bucket_specs,
)


class VisionBucketPresetTest(unittest.TestCase):
    def test_full_vision_preserves_ten_legacy_static_code_objects(self) -> None:
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

    def test_flat_global_context_has_stable_affected_bucket_methods(self) -> None:
        source = Path(__file__).with_name("vision_full_batch.py").read_text()
        tree = ast.parse(source)
        encoder = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "_FlatGlobalContextFullVisionEncoder"
        )
        methods = {
            node.name
            for node in encoder.body
            if isinstance(node, ast.FunctionDef)
            and node.name.startswith("_forward_flat_bucket_slot_")
        }
        self.assertEqual(
            methods,
            {
                "_forward_flat_bucket_slot_6",
                "_forward_flat_bucket_slot_8",
            },
        )

        flat_keys = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "FLAT_GLOBAL_CONTEXT_BUCKET_KEYS"
                for target in node.targets
            )
        )
        self.assertIsInstance(flat_keys, ast.Call)
        self.assertEqual(
            {
                element.value
                for element in flat_keys.args[0].elts
                if isinstance(element, ast.Constant)
            },
            {
                "1024x704_b1",
                "1024x1408_b1",
            },
        )

        extended = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "_ExtendedFlatGlobalContextFullVisionEncoder"
        )
        extended_methods = {
            node.name
            for node in extended.body
            if isinstance(node, ast.FunctionDef)
            and node.name.startswith("_forward_flat_bucket_slot_")
        }
        self.assertEqual(
            extended_methods,
            {
                f"_forward_flat_bucket_slot_{index}"
                for index in range(20)
                if index not in {6, 8}
            },
        )
        extended_flat_keys = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "EXTENDED_FLAT_GLOBAL_CONTEXT_BUCKET_KEYS"
                for target in node.targets
            )
        )
        self.assertEqual(
            {
                element.value
                for element in extended_flat_keys.args[0].elts
                if isinstance(element, ast.Constant)
            },
            {
                "128x1408_b1",
                "192x64_b4",
                "320x320_b2",
                "448x192_b2",
                "448x576_b1",
                "512x128_b4",
                "512x768_b1",
                "576x256_b2",
                "960x192_b1",
                "960x384_b1",
                "960x704_b1",
                "960x896_b1",
                "960x1152_b1",
                "960x1344_b1",
            },
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

    def test_k10_l4_aligned_has_ten_safe_graph_variants(self) -> None:
        specs = resolve_vision_bucket_specs("310p_k10_l4_aligned")
        self.assertEqual(specs, VISION_BUCKETS_310P_K10_L4_ALIGNED)
        self.assertEqual(len(specs), 10)
        self.assertEqual(len({spec.key for spec in specs}), 10)
        self.assertTrue(all(spec.has_aligned_final_stage_rows for spec in specs))
        self.assertTrue(any(spec.width == 1024 for spec in specs))
        for width, height in ((64, 1408), (448, 1152), (896, 576)):
            self.assertTrue(any(spec.accepts(width, height) for spec in specs))

    def test_k20_l4_has_twenty_unrestricted_graph_variants(self) -> None:
        specs = resolve_vision_bucket_specs("310p_k20_l4")
        self.assertEqual(specs, VISION_BUCKETS_310P_K20_L4)
        self.assertEqual(len(specs), 20)
        self.assertEqual(len({spec.key for spec in specs}), 20)
        self.assertTrue(any(not spec.has_aligned_final_stage_rows for spec in specs))
        for width, height in ((64, 1408), (448, 1152), (896, 576)):
            self.assertTrue(any(spec.accepts(width, height) for spec in specs))

    def test_k20_preserves_six_validated_k10_cache_slots(self) -> None:
        aligned_slots = dict(
            zip(
                (spec.key for spec in VISION_BUCKETS_310P_K10_L4_ALIGNED),
                assign_vision_bucket_cache_slots(
                    VISION_BUCKETS_310P_K10_L4_ALIGNED
                ),
            )
        )
        k20_slots = dict(
            zip(
                (spec.key for spec in VISION_BUCKETS_310P_K20_L4),
                assign_vision_bucket_cache_slots(
                    VISION_BUCKETS_310P_K20_L4,
                    slot_count=20,
                ),
            )
        )
        shared = set(aligned_slots) & set(k20_slots)
        self.assertEqual(
            shared,
            {
                "448x384_b2",
                "512x64_b4",
                "960x64_b4",
                "960x128_b2",
                "960x256_b1",
                "960x512_b1",
            },
        )
        for key in shared:
            self.assertEqual(k20_slots[key], aligned_slots[key], key)
        self.assertEqual(len(set(k20_slots.values())), 20)

    def test_aligned_k10_preserves_shared_l4_cache_slots(self) -> None:
        old_slots = dict(
            zip(
                (spec.key for spec in VISION_BUCKETS_310P_K10_L4_ALL),
                assign_vision_bucket_cache_slots(VISION_BUCKETS_310P_K10_L4_ALL),
            )
        )
        aligned_slots = dict(
            zip(
                (spec.key for spec in VISION_BUCKETS_310P_K10_L4_ALIGNED),
                assign_vision_bucket_cache_slots(
                    VISION_BUCKETS_310P_K10_L4_ALIGNED
                ),
            )
        )
        shared = set(old_slots) & set(aligned_slots)
        self.assertEqual(shared, {
            "448x384_b2",
            "512x64_b4",
            "960x64_b4",
            "960x128_b2",
            "960x256_b1",
        })
        for key in shared:
            self.assertEqual(aligned_slots[key], old_slots[key], key)
        self.assertEqual(
            tuple(aligned_slots.values()),
            (1, 2, 0, 3, 4, 5, 9, 7, 8, 6),
        )

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
