#!/usr/bin/env python3
"""Compare existing 310P UniRec prefill summaries with matched 910B controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REFERENCE_910B: dict[str, dict[str, Any]] = {
    "w1_t16": {
        "commit": "7821ad5",
        "physical_npu": 7,
        "producer_wall_s": 23.127915844786912,
        "pages_per_s": 5.534437294696898,
        "real_source_tokens_per_s": 2659.9889247636966,
        "worker_busy_s": [22.89660197496413],
        "worker_page_counts": [128],
        "stages": {
            "layout": 6.675654815975577,
            "cpu_crop": 1.755044178571552,
            "prefill_including_d2h": 4.868537549860775,
            "d2h_substage": 0.5914551797322929,
            "shared_pack": 2.6466299882158637,
            "file_io": 1.0646912283264101,
            "frontend_cpu": 0.4535658733096719,
            "ipc_delivery": 0.9389518746174872,
        },
        "vision_real_rows": 950,
        "vision_physical_rows": 1424,
        "vision_slot_efficiency": 0.6671348314606742,
    },
    "w8_t8": {
        "commit": "4747d8e",
        "physical_npu": 6,
        "producer_wall_s": 4.679882558062673,
        "pages_per_s": 27.351113711064592,
        "real_source_tokens_per_s": 13145.629027380419,
        "worker_busy_s": [
            4.424791673664004,
            4.137954497244208,
            4.124018770176917,
            4.617129490245133,
            4.456353510264308,
            3.9863158208318064,
            4.642300918698312,
            4.244310915935785,
        ],
        "worker_page_counts": [16, 16, 16, 16, 16, 16, 16, 16],
        "stages": {
            "layout": 13.881140884011984,
            "cpu_crop": 2.0222724163904786,
            "prefill_including_d2h": 7.746465802658351,
            "d2h_substage": 0.7117893914692104,
            "shared_pack": 2.8600587719120085,
            "file_io": 1.1276209368370473,
            "frontend_cpu": 0.5254891230724752,
            "ipc_delivery": 1.9411654435098171,
        },
        "vision_real_rows": 950,
        "vision_physical_rows": 1456,
        "vision_slot_efficiency": 0.6524725274725275,
    },
}

EXPECTED_WORKLOAD = {
    "pages": 128,
    "crops": 950,
    "rejected": 6,
    "real_source_tokens": 61520,
    "physical_source_tokens": 129024,
}

INDEPENDENT_STAGE_NAMES = (
    "layout",
    "cpu_crop",
    "prefill_including_d2h",
    "shared_pack",
    "file_io",
    "frontend_cpu",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu310-w1", type=Path, required=True)
    parser.add_argument("--npu310-w8", type=Path, required=True)
    return parser.parse_args()


def _read_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _extract_summary(
    summary: dict[str, Any],
    *,
    workers: int,
    threads: int,
) -> dict[str, Any]:
    if summary["status"] != "ok":
        raise ValueError("prefill summary status is not ok")
    expected_config = {
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
    }
    for key, expected in expected_config.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"unmatched 310P configuration: {key}={summary.get(key)!r}, "
                f"expected {expected!r}"
            )
    if summary["validation"]["passed"] is not True:
        raise ValueError("310P descriptor validation did not pass")

    artifact = summary["artifact"]
    workload = {
        "pages": int(artifact["page_count"]),
        "crops": int(artifact["crop_count"]),
        "rejected": int(artifact["rejected_crop_count"]),
        "real_source_tokens": int(artifact["real_source_tokens"]),
        "physical_source_tokens": int(artifact["physical_source_tokens"]),
    }
    if workload != EXPECTED_WORKLOAD:
        raise ValueError(
            f"310P workload differs from the 910B reference: {workload!r}"
        )

    worker = summary["worker_summary"]
    if int(worker["worker_count"]) != workers:
        raise ValueError("310P worker summary has the wrong worker count")
    if worker["prefix_diagnostics"]["new_first_call_count"] != 0:
        raise ValueError("310P measured window contains unexpected graph first calls")
    page_counts = [int(value) for value in worker["worker_page_counts"]]
    if len(page_counts) != workers or any(value <= 0 for value in page_counts):
        raise ValueError(f"310P workers were not all active: {page_counts!r}")

    stages = worker["stage_s"]
    vision = worker["vision_batching"]
    return {
        "producer_wall_s": float(summary["producer_wall_s"]),
        "pages_per_s": float(summary["throughput"]["pages_per_s"]),
        "real_source_tokens_per_s": float(
            summary["throughput"]["real_source_tokens_per_s"]
        ),
        "worker_busy_s": [float(value) for value in worker["worker_busy_s"]],
        "worker_page_counts": page_counts,
        "stages": {
            "layout": float(stages["worker_detector_call_sum_s"]),
            "cpu_crop": float(stages["worker_recognition_input_prepare_sum_s"]),
            "prefill_including_d2h": float(
                stages["worker_recognition_prefill_sum_s"]
            ),
            "d2h_substage": float(
                stages["worker_recognition_prefill_cache_d2h_sum_s"]
            ),
            "shared_pack": float(stages["worker_shared_pack_sum_s"]),
            "file_io": float(stages["worker_file_read_sum_s"])
            + float(stages["worker_direct_rgb_decode_sum_s"]),
            "frontend_cpu": float(stages["worker_layout_crop_views_sum_s"])
            + float(stages["worker_document_image_index_sum_s"])
            + float(stages["worker_recognition_crop_build_sum_s"]),
            "ipc_delivery": float(worker["ipc_delivery_sum_s"]),
        },
        "vision_real_rows": int(vision["compiled_real_rows"]),
        "vision_physical_rows": int(vision["compiled_physical_rows"]),
        "vision_slot_efficiency": float(vision["compiled_slot_efficiency"]),
    }


def analyze_gap(
    summary_310p_w1: dict[str, Any],
    summary_310p_w8: dict[str, Any],
) -> dict[str, Any]:
    w1 = _extract_summary(summary_310p_w1, workers=1, threads=16)
    w8 = _extract_summary(summary_310p_w8, workers=8, threads=8)
    ref_w1 = REFERENCE_910B["w1_t16"]
    ref_w8 = REFERENCE_910B["w8_t8"]

    w1_producer_ratio = w1["producer_wall_s"] / ref_w1["producer_wall_s"]
    w8_producer_ratio = w8["producer_wall_s"] / ref_w8["producer_wall_s"]
    scaling_310p = w8["pages_per_s"] / w1["pages_per_s"]
    scaling_910b = ref_w8["pages_per_s"] / ref_w1["pages_per_s"]
    scaling_penalty = scaling_910b / scaling_310p
    producer_gap_s = w1["producer_wall_s"] - ref_w1["producer_wall_s"]

    w1_stages = {}
    for name, value in w1["stages"].items():
        reference = float(ref_w1["stages"][name])
        w1_stages[name] = {
            "npu310_s": value,
            "npu910_s": reference,
            "ratio": value / reference if reference else None,
            "gap_s": value - reference,
            "producer_gap_share": (
                (value - reference) / producer_gap_s if producer_gap_s else None
            ),
        }

    primary_stage = max(
        INDEPENDENT_STAGE_NAMES,
        key=lambda name: float(w1_stages[name]["gap_s"]),
    )
    w8_stage_ratios = {
        name: w8["stages"][name] / float(ref_w8["stages"][name])
        for name in w8["stages"]
    }
    return {
        "w1_producer_ratio": w1_producer_ratio,
        "w8_producer_ratio": w8_producer_ratio,
        "scaling_310p": scaling_310p,
        "scaling_910b": scaling_910b,
        "scaling_penalty": scaling_penalty,
        "w1_stage_comparison": w1_stages,
        "w8_stage_ratios": w8_stage_ratios,
        "primary_w1_stage": primary_stage,
        "vision": {
            "npu310_w1": {
                "real_rows": w1["vision_real_rows"],
                "physical_rows": w1["vision_physical_rows"],
                "slot_efficiency": w1["vision_slot_efficiency"],
            },
            "npu310_w8": {
                "real_rows": w8["vision_real_rows"],
                "physical_rows": w8["vision_physical_rows"],
                "slot_efficiency": w8["vision_slot_efficiency"],
            },
            "npu910_w1": {
                "real_rows": ref_w1["vision_real_rows"],
                "physical_rows": ref_w1["vision_physical_rows"],
                "slot_efficiency": ref_w1["vision_slot_efficiency"],
            },
            "npu910_w8": {
                "real_rows": ref_w8["vision_real_rows"],
                "physical_rows": ref_w8["vision_physical_rows"],
                "slot_efficiency": ref_w8["vision_slot_efficiency"],
            },
        },
    }


def _ratio_list(values: dict[str, float]) -> str:
    return ",".join(
        f"{name}:{ratio:.2f}x"
        for name, ratio in sorted(
            values.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )


def print_analysis(analysis: dict[str, Any]) -> None:
    stage_comparison = analysis["w1_stage_comparison"]
    primary_name = str(analysis["primary_w1_stage"])
    primary = stage_comparison[primary_name]
    primary_gap_share = primary["producer_gap_share"]
    primary_gap_share_pct = (
        float(primary_gap_share) * 100.0
        if primary_gap_share is not None
        else 0.0
    )
    independent_ratios = {
        name: float(stage_comparison[name]["ratio"])
        for name in INDEPENDENT_STAGE_NAMES
    }
    print(
        "UNIREC_GAP_HEADLINE "
        f"w1_310p_over_910b={analysis['w1_producer_ratio']:.2f}x "
        f"w8_310p_over_910b={analysis['w8_producer_ratio']:.2f}x "
        f"scaling_310p={analysis['scaling_310p']:.2f}x "
        f"scaling_910b={analysis['scaling_910b']:.2f}x "
        f"scaling_penalty={analysis['scaling_penalty']:.2f}x"
    )
    print(
        "UNIREC_GAP_W1 "
        f"primary={primary_name} "
        f"primary_ratio={primary['ratio']:.2f}x "
        f"primary_gap_s={primary['gap_s']:.2f} "
        f"primary_gap_share={primary_gap_share_pct:.1f}% "
        f"stage_ratios={_ratio_list(independent_ratios)} "
        f"d2h_substage={stage_comparison['d2h_substage']['ratio']:.2f}x "
        f"ipc_delivery={stage_comparison['ipc_delivery']['ratio']:.2f}x"
    )
    print(
        "UNIREC_GAP_W8 "
        f"aggregate_stage_ratios={_ratio_list(analysis['w8_stage_ratios'])}"
    )
    print(
        "UNIREC_GAP_DIAGNOSIS "
        f"sequential_primary={primary_name} "
        f"multiworker_scaling_loss={analysis['scaling_penalty']:.2f}x "
        "note=w8_stage_sums_overlap_and_include_contention"
    )


def main() -> None:
    args = parse_args()
    analysis = analyze_gap(
        _read_summary(args.npu310_w1),
        _read_summary(args.npu310_w8),
    )
    print_analysis(analysis)


if __name__ == "__main__":
    main()
