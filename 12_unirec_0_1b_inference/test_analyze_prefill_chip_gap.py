#!/usr/bin/env python3
"""CPU-only tests for UniRec 310P/910B prefill-gap attribution."""

from __future__ import annotations

import copy
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from analyze_prefill_chip_gap import (
    EXPECTED_WORKLOAD,
    REFERENCE_910B,
    analyze_gap,
    print_analysis,
)


def _summary_from_reference(
    name: str,
    *,
    workers: int,
    threads: int,
    producer_scale: float,
    stage_scales: dict[str, float],
) -> dict:
    reference = REFERENCE_910B[name]
    stages = reference["stages"]
    source_keys = {
        "layout": "worker_detector_call_sum_s",
        "cpu_crop": "worker_recognition_input_prepare_sum_s",
        "prefill_including_d2h": "worker_recognition_prefill_sum_s",
        "d2h_substage": "worker_recognition_prefill_cache_d2h_sum_s",
        "shared_pack": "worker_shared_pack_sum_s",
        "ipc_delivery": None,
    }
    worker_stages = {
        "worker_file_read_sum_s": stages["file_io"] * 0.1,
        "worker_direct_rgb_decode_sum_s": stages["file_io"] * 0.9,
        "worker_layout_crop_views_sum_s": stages["frontend_cpu"] * 0.1,
        "worker_document_image_index_sum_s": stages["frontend_cpu"] * 0.1,
        "worker_recognition_crop_build_sum_s": stages["frontend_cpu"] * 0.8,
    }
    for name_key, source in source_keys.items():
        if source is not None:
            worker_stages[source] = stages[name_key] * stage_scales.get(name_key, 1.0)
    producer_wall_s = reference["producer_wall_s"] * producer_scale
    return {
        "status": "ok",
        "offset": 0,
        "limit": 128,
        "workers": workers,
        "recognition_preprocess_threads": threads,
        "artifact_storage": "discard",
        "cross_cache_length": 512,
        "layout_execution": "torchair",
        "layout_batch_size": 1,
        "vision_full_batches": True,
        "recognition_input_contract": "compact_uint8_hwc",
        "validation": {"passed": True},
        "artifact": {
            "page_count": EXPECTED_WORKLOAD["pages"],
            "crop_count": EXPECTED_WORKLOAD["crops"],
            "rejected_crop_count": EXPECTED_WORKLOAD["rejected"],
            "real_source_tokens": EXPECTED_WORKLOAD["real_source_tokens"],
            "physical_source_tokens": EXPECTED_WORKLOAD["physical_source_tokens"],
        },
        "producer_wall_s": producer_wall_s,
        "throughput": {
            "pages_per_s": 128 / producer_wall_s,
            "real_source_tokens_per_s": (
                EXPECTED_WORKLOAD["real_source_tokens"] / producer_wall_s
            ),
        },
        "worker_summary": {
            "worker_count": workers,
            "worker_page_counts": [128 // workers] * workers,
            "worker_busy_s": [producer_wall_s * 0.95] * workers,
            "prefix_diagnostics": {"new_first_call_count": 0},
            "stage_s": worker_stages,
            "ipc_delivery_sum_s": stages["ipc_delivery"]
            * stage_scales.get("ipc_delivery", 1.0),
            "vision_batching": {
                "compiled_real_rows": reference["vision_real_rows"],
                "compiled_physical_rows": reference["vision_physical_rows"],
                "compiled_slot_efficiency": reference[
                    "vision_slot_efficiency"
                ],
            },
        },
    }


class PrefillChipGapAnalysisTest(unittest.TestCase):
    def test_layout_is_ranked_as_primary_sequential_gap(self) -> None:
        w1 = _summary_from_reference(
            "w1_t16",
            workers=1,
            threads=16,
            producer_scale=2.0,
            stage_scales={"layout": 3.0, "prefill_including_d2h": 1.5},
        )
        w8 = _summary_from_reference(
            "w8_t8",
            workers=8,
            threads=8,
            producer_scale=4.0,
            stage_scales={"layout": 4.0, "prefill_including_d2h": 3.0},
        )

        analysis = analyze_gap(w1, w8)

        self.assertEqual(analysis["primary_w1_stage"], "layout")
        self.assertAlmostEqual(analysis["w1_producer_ratio"], 2.0)
        self.assertAlmostEqual(analysis["w8_producer_ratio"], 4.0)
        self.assertGreater(analysis["scaling_penalty"], 1.0)

    def test_rejects_an_unmatched_workload(self) -> None:
        w1 = _summary_from_reference(
            "w1_t16",
            workers=1,
            threads=16,
            producer_scale=2.0,
            stage_scales={},
        )
        w8 = _summary_from_reference(
            "w8_t8",
            workers=8,
            threads=8,
            producer_scale=4.0,
            stage_scales={},
        )
        w1 = copy.deepcopy(w1)
        w1["artifact"]["crop_count"] -= 1

        with self.assertRaisesRegex(ValueError, "workload differs"):
            analyze_gap(w1, w8)

    def test_prints_a_zero_gap_control(self) -> None:
        w1 = _summary_from_reference(
            "w1_t16",
            workers=1,
            threads=16,
            producer_scale=1.0,
            stage_scales={},
        )
        w8 = _summary_from_reference(
            "w8_t8",
            workers=8,
            threads=8,
            producer_scale=1.0,
            stage_scales={},
        )
        output = io.StringIO()

        with redirect_stdout(output):
            print_analysis(analyze_gap(w1, w8))

        self.assertIn("w1_310p_over_910b=1.00x", output.getvalue())
        self.assertIn("primary_gap_share=0.0%", output.getvalue())


if __name__ == "__main__":
    unittest.main()
