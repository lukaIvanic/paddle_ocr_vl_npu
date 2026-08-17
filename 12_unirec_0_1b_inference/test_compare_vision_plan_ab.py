from __future__ import annotations

import unittest

from compare_vision_plan_ab import comparison, summarize_events


def event(
    *,
    name: str,
    bucket: str,
    batch: int,
    height: int,
    width: int,
    members: list[tuple[int, int]],
    graph_s: float,
    input_s: float,
) -> dict:
    return {
        "event": name,
        "bucket": bucket,
        "real_rows": len(members),
        "physical_rows": batch,
        "physical_input_shape": [batch, 3, height, width],
        "members": [
            {"processed_image_size": [member_width, member_height]}
            for member_width, member_height in members
        ],
        "device_stage_s": {
            "graph_s": graph_s,
            "input_h2d_normalize_s": input_s,
            "output_compact_s": 0.001 if name == "vision_bucket_call" else 0.0,
        },
        "h2d_bytes": {"total": batch * height * width * 3},
    }


class VisionPlanComparisonTest(unittest.TestCase):
    def test_event_summary_measures_spatial_and_slot_padding(self) -> None:
        rows = [
            event(
                name="vision_bucket_call",
                bucket="128x64_b2",
                batch=2,
                height=64,
                width=128,
                members=[(64, 64)],
                graph_s=0.020,
                input_s=0.004,
            ),
            event(
                name="vision_fallback_call",
                bucket="fallback_eager",
                batch=1,
                height=128,
                width=64,
                members=[(64, 128)],
                graph_s=0.030,
                input_s=0.002,
            ),
        ]
        result = summarize_events(rows)
        self.assertEqual(result["calls"], 2)
        self.assertEqual(result["real_rows"], 2)
        self.assertEqual(result["physical_rows"], 3)
        self.assertAlmostEqual(result["slot_efficiency"], 2 / 3)
        self.assertEqual(result["effective_pixels"], 64 * 64 + 64 * 128)
        self.assertEqual(result["physical_pixels"], 2 * 64 * 128 + 64 * 128)
        self.assertAlmostEqual(result["pixel_efficiency"], 0.5)
        self.assertAlmostEqual(result["graph_s"], 0.050)
        self.assertAlmostEqual(result["input_device_s"], 0.006)

    def test_comparison_reconciles_full_vision_delta(self) -> None:
        config = {
            key: value
            for key, value in {
                "page_count": 1,
                "workers": 1,
                "recognition_preprocess_threads": 8,
                "layout_batch_size": 2,
                "layout_cpu_threads": 16,
                "layout_execution": "torchair",
                "layout_dtype": "float16",
                "layout_reading_order_dtype": "float32",
                "layout_threshold": 0.5,
                "vision_focal_depthwise_rewrite": "constant_grouped_all",
                "vision_weight_format": "torchair_internal",
                "cross_cache_length": 1320,
                "self_cache_length": 2048,
            }.items()
        }

        def lane(graph: float, input_s: float, residual: float) -> dict:
            vision = summarize_events(
                [
                    event(
                        name="vision_bucket_call",
                        bucket="64x64_b1",
                        batch=1,
                        height=64,
                        width=64,
                        members=[(64, 64)],
                        graph_s=graph,
                        input_s=input_s,
                    )
                ]
            )
            vision["vision_wall_s"] = vision["device_total_s"] + residual
            vision["vision_wall_residual_s"] = residual
            return {
                "config": config,
                "prefill": {"wall_s": 1.0, "pages_per_s": 1.0},
                "page_identity": {
                    "available": True,
                    "identity_digest": "same",
                    "workload_digest": "same",
                },
                "vision": vision,
            }

        result = comparison(lane(0.10, 0.02, 0.03), lane(0.08, 0.03, 0.02))
        decomposition = result["vision_wall_decomposition_s"]
        self.assertAlmostEqual(decomposition["graph_delta"], -0.02)
        self.assertAlmostEqual(decomposition["input_device_delta"], 0.01)
        self.assertAlmostEqual(decomposition["wall_residual_delta"], -0.01)
        self.assertAlmostEqual(decomposition["observed_wall_delta"], -0.02)
        self.assertAlmostEqual(
            decomposition["reconstructed_wall_delta"],
            decomposition["observed_wall_delta"],
        )


if __name__ == "__main__":
    unittest.main()
