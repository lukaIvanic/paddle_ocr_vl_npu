#!/usr/bin/env python3
"""Compare a 310P UniRec prefill profile with the committed 910B reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REFERENCE_GRAPH = (
    HERE / "references/prefill_graph_profile_910b_26f6e90_pipe.json"
)
REFERENCE_LAYOUT = (
    HERE / "references/layout_detector_first128_910b_26f6e90.json"
)

REFERENCE_PRODUCER = {
    "layout_s": 6.675654815975577,
    "recognition_prefill_s": 4.868537549860775,
    "d2h_s": 0.5914551797322929,
}

EXPECTED_LANES = (
    "layout_b1_800x800_fp32",
    "vision_960x64_b16_fp16",
    "vision_512x256_b16_fp16",
    "vision_960x256_b4_fp16",
    "vision_512x512_b8_fp16",
    "vision_960x512_b4_fp16",
    "cross_kv_packed_b1_s1024_fp16",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu310-graph-profile", type=Path, required=True)
    parser.add_argument("--npu310-layout-lab", type=Path, required=True)
    parser.add_argument("--npu310-producer-w1", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _group_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["name"]): row for row in rows}


def _extract_profile(lane: dict[str, Any]) -> dict[str, Any]:
    runs = lane["parsed_profile"]["summary"]["runs"]
    if len(runs) != 1:
        raise ValueError(f"lane {lane['name']} must contain exactly one profile run")
    run = runs[0]
    kernel = run["kernel_details"]
    operator = run.get("operator_details", {})
    api = run.get("api_statistic", {})
    return {
        "kernel": {
            "row_count": int(kernel["row_count"]),
            "total_duration_us": float(kernel["total_duration_us"]),
            "total_wait_us": float(kernel["total_wait_us"]),
            "total_aicore_time_us": float(kernel["total_aicore_time_us"]),
            "weighted_cube_utilization_pct": float(
                kernel["weighted_cube_utilization_pct"]
            ),
            "top_kernel_types": kernel["top_kernel_types"],
            "top_core_types": kernel["top_core_types"],
            "top_shape_signatures": kernel["top_shape_signatures"],
            "top_matmul_shape_signatures": kernel[
                "top_matmul_shape_signatures"
            ],
            "top_transdata_shape_signatures": kernel[
                "top_transdata_shape_signatures"
            ],
        },
        "operator": {
            "row_count": int(operator.get("row_count", 0)),
            "total_device_us": float(operator.get("total_device_us", 0.0)),
            "total_host_us": float(operator.get("total_host_us", 0.0)),
            "top_by_device_total_us": operator.get(
                "top_by_device_total_us", []
            ),
        },
        "api": {
            "row_count": int(api.get("row_count", 0)),
            "top_apis": api.get("top_apis", []),
        },
    }


def _normalize_310_graph(raw: dict[str, Any]) -> dict[str, Any]:
    if raw.get("format") != "unirec_prefill_graph_profile_suite_v1":
        raise ValueError("310P graph profile has the wrong format")
    config = raw["config"]
    if config.get("profile_metric") != "pipe":
        raise ValueError("310P comparison requires profile metric pipe")
    if int(config.get("profile_steps", 0)) != 1:
        raise ValueError("310P comparison requires exactly one profiled replay")
    device_name = str(raw["environment"]["device_name"])
    if "310" not in device_name:
        raise ValueError(f"expected a 310P profile, got device={device_name!r}")
    lanes = {str(lane["name"]): lane for lane in raw["lanes"]}
    if set(lanes) != set(EXPECTED_LANES):
        raise ValueError(f"310P profile lane mismatch: {sorted(lanes)}")
    normalized = []
    for name in EXPECTED_LANES:
        lane = lanes[name]
        normalized.append(
            {
                "name": name,
                "first128_calls": int(lane["first128_calls"]),
                "steady_device_event_mean_ms": float(
                    lane["steady_device_event_mean_ms"]
                ),
                "weighted_first128_device_s": float(
                    lane["weighted_first128_device_s"]
                ),
                "profile_steps": int(lane["profile_steps"]),
                "profile": _extract_profile(lane),
            }
        )
    return {
        "environment": raw["environment"],
        "first128_workload": raw["first128_workload"],
        "lanes": normalized,
    }


def _validate_graph_contract(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> None:
    if candidate["first128_workload"] != reference["first128_workload"]:
        raise ValueError("310P and 910B first-128 graph weighting differs")
    ref_lanes = {lane["name"]: lane for lane in reference["lanes"]}
    for lane in candidate["lanes"]:
        ref = ref_lanes[lane["name"]]
        if lane["first128_calls"] != int(ref["first128_calls"]):
            raise ValueError(f"call-count mismatch for {lane['name']}")


def _compare_groups(
    candidate_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    calls: int,
    limit: int = 12,
) -> list[dict[str, Any]]:
    candidate = _group_map(candidate_rows)
    reference = _group_map(reference_rows)
    compared = []
    for name in set(candidate) | set(reference):
        npu310_us = float(candidate.get(name, {}).get("duration_us", 0.0))
        npu910_us = float(reference.get(name, {}).get("duration_us", 0.0))
        delta_us = npu310_us - npu910_us
        compared.append(
            {
                "name": name,
                "npu310_us": npu310_us,
                "npu910_us": npu910_us,
                "ratio": _ratio(npu310_us, npu910_us),
                "delta_us_per_replay": delta_us,
                "weighted_first128_delta_s": delta_us * calls / 1_000_000.0,
            }
        )
    return sorted(
        compared,
        key=lambda row: float(row["weighted_first128_delta_s"]),
        reverse=True,
    )[:limit]


def _compare_lane(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    calls = int(reference["first128_calls"])
    npu310_ms = float(candidate["steady_device_event_mean_ms"])
    npu910_ms = float(reference["steady_device_event_mean_ms"])
    kernel_310 = candidate["profile"]["kernel"]
    kernel_910 = reference["profile"]["kernel"]
    return {
        "name": reference["name"],
        "first128_calls": calls,
        "npu310_device_ms": npu310_ms,
        "npu910_device_ms": npu910_ms,
        "device_ratio": _ratio(npu310_ms, npu910_ms),
        "npu310_weighted_s": float(candidate["weighted_first128_device_s"]),
        "npu910_weighted_s": float(reference["weighted_first128_device_s"]),
        "weighted_gap_s": float(candidate["weighted_first128_device_s"])
        - float(reference["weighted_first128_device_s"]),
        "kernel_count_310": int(kernel_310["row_count"]),
        "kernel_count_910": int(kernel_910["row_count"]),
        "kernel_profile_duration_ratio": _ratio(
            float(kernel_310["total_duration_us"]),
            float(kernel_910["total_duration_us"]),
        ),
        "cube_utilization_310_pct": float(
            kernel_310["weighted_cube_utilization_pct"]
        ),
        "cube_utilization_910_pct": float(
            kernel_910["weighted_cube_utilization_pct"]
        ),
        "top_kernel_type_deltas": _compare_groups(
            kernel_310["top_kernel_types"],
            kernel_910["top_kernel_types"],
            calls=calls,
        ),
        "top_shape_deltas": _compare_groups(
            kernel_310["top_shape_signatures"],
            kernel_910["top_shape_signatures"],
            calls=calls,
        ),
        "top_matmul_shape_deltas": _compare_groups(
            kernel_310["top_matmul_shape_signatures"],
            kernel_910["top_matmul_shape_signatures"],
            calls=calls,
        ),
        "top_transdata_shape_deltas": _compare_groups(
            kernel_310["top_transdata_shape_signatures"],
            kernel_910["top_transdata_shape_signatures"],
            calls=calls,
        ),
    }


def _validate_layout_lab(raw: dict[str, Any]) -> None:
    config = raw["config"]
    expected = {
        "dtype": "float32",
        "execution": "torchair",
        "offset": 0,
        "limit": 128,
        "threshold": 0.4,
        "scheduling": "sequential_b1_same_process",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"310P layout lab mismatch: {key}={config.get(key)!r}, "
                f"expected={value!r}"
            )
    if int(raw["summary"]["page_count"]) != 128:
        raise ValueError("310P layout lab did not process 128 pages")


def _compare_layout_lab(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    _validate_layout_lab(candidate)
    ref_contracts = reference["page_contracts"]
    candidate_contracts = [
        {
            key: page[key]
            for key in ("page_index", "height", "width", "box_count")
        }
        for page in candidate["pages"]
    ]
    expected_contracts = [
        {
            key: page[key]
            for key in ("page_index", "height", "width", "box_count")
        }
        for page in ref_contracts
    ]
    if candidate_contracts != expected_contracts:
        raise ValueError("310P layout pages, shapes, or box counts differ from 910B")
    candidate_stages = candidate["summary"]["stages"]
    reference_stages = reference["summary"]["stages"]
    stages = {}
    for name in sorted(set(candidate_stages) | set(reference_stages)):
        npu310_s = float(candidate_stages.get(name, {}).get("total_s", 0.0))
        npu910_s = float(reference_stages.get(name, {}).get("total_s", 0.0))
        stages[name] = {
            "npu310_s": npu310_s,
            "npu910_s": npu910_s,
            "ratio": _ratio(npu310_s, npu910_s),
            "gap_s": npu310_s - npu910_s,
        }
    return {
        "npu310_page_wall_s": float(candidate["summary"]["measured_page_wall_s"]),
        "npu910_page_wall_s": float(reference["summary"]["measured_page_wall_s"]),
        "page_wall_ratio": _ratio(
            float(candidate["summary"]["measured_page_wall_s"]),
            float(reference["summary"]["measured_page_wall_s"]),
        ),
        "stages": stages,
    }


def _extract_producer_w1(raw: dict[str, Any]) -> dict[str, float]:
    expected = {
        "status": "ok",
        "offset": 0,
        "limit": 128,
        "workers": 1,
        "recognition_preprocess_threads": 16,
        "artifact_storage": "discard",
        "cross_cache_length": 512,
        "layout_execution": "torchair",
        "layout_batch_size": 1,
        "vision_full_batches": True,
        "recognition_input_contract": "compact_uint8_hwc",
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(
                f"310P producer mismatch: {key}={raw.get(key)!r}, "
                f"expected={value!r}"
            )
    stage = raw["worker_summary"]["stage_s"]
    return {
        "layout_s": float(stage["worker_detector_call_sum_s"]),
        "recognition_prefill_s": float(
            stage["worker_recognition_prefill_sum_s"]
        ),
        "d2h_s": float(stage["worker_recognition_prefill_cache_d2h_sum_s"]),
    }


def analyze(
    graph_310_raw: dict[str, Any],
    layout_310_raw: dict[str, Any],
    producer_310_raw: dict[str, Any],
    *,
    graph_reference: dict[str, Any],
    layout_reference: dict[str, Any],
) -> dict[str, Any]:
    graph_310 = _normalize_310_graph(graph_310_raw)
    _validate_graph_contract(graph_reference, graph_310)
    ref_lanes = {lane["name"]: lane for lane in graph_reference["lanes"]}
    lane_comparisons = [
        _compare_lane(lane, ref_lanes[lane["name"]])
        for lane in graph_310["lanes"]
    ]
    by_name = {lane["name"]: lane for lane in lane_comparisons}
    vision_names = [name for name in EXPECTED_LANES if name.startswith("vision_")]

    def lane_sum(names: list[str], field: str) -> float:
        return sum(float(by_name[name][field]) for name in names)

    vision_310_s = lane_sum(vision_names, "npu310_weighted_s")
    vision_910_s = lane_sum(vision_names, "npu910_weighted_s")
    text = by_name["cross_kv_packed_b1_s1024_fp16"]
    recognition_310_s = vision_310_s + float(text["npu310_weighted_s"])
    recognition_910_s = vision_910_s + float(text["npu910_weighted_s"])
    layout_graph = by_name["layout_b1_800x800_fp32"]

    producer_310 = _extract_producer_w1(producer_310_raw)
    layout_stage_gap = producer_310["layout_s"] - REFERENCE_PRODUCER["layout_s"]
    layout_graph_gap = float(layout_graph["weighted_gap_s"])
    recognition_stage_gap = (
        producer_310["recognition_prefill_s"]
        - REFERENCE_PRODUCER["recognition_prefill_s"]
    )
    recognition_graph_gap = recognition_310_s - recognition_910_s
    accounting = {
        "layout": {
            "npu310_total_s": producer_310["layout_s"],
            "npu910_total_s": REFERENCE_PRODUCER["layout_s"],
            "total_ratio": _ratio(
                producer_310["layout_s"], REFERENCE_PRODUCER["layout_s"]
            ),
            "npu310_graph_s": float(layout_graph["npu310_weighted_s"]),
            "npu910_graph_s": float(layout_graph["npu910_weighted_s"]),
            "graph_ratio": layout_graph["device_ratio"],
            "graph_gap_share": _ratio(layout_graph_gap, layout_stage_gap),
            "npu310_surrounding_s": producer_310["layout_s"]
            - float(layout_graph["npu310_weighted_s"]),
            "npu910_surrounding_s": REFERENCE_PRODUCER["layout_s"]
            - float(layout_graph["npu910_weighted_s"]),
        },
        "recognition_prefill": {
            "npu310_total_s": producer_310["recognition_prefill_s"],
            "npu910_total_s": REFERENCE_PRODUCER["recognition_prefill_s"],
            "total_ratio": _ratio(
                producer_310["recognition_prefill_s"],
                REFERENCE_PRODUCER["recognition_prefill_s"],
            ),
            "npu310_graph_s": recognition_310_s,
            "npu910_graph_s": recognition_910_s,
            "graph_ratio": _ratio(recognition_310_s, recognition_910_s),
            "graph_gap_share": _ratio(recognition_graph_gap, recognition_stage_gap),
            "npu310_surrounding_s": producer_310["recognition_prefill_s"]
            - recognition_310_s,
            "npu910_surrounding_s": REFERENCE_PRODUCER[
                "recognition_prefill_s"
            ]
            - recognition_910_s,
            "vision_graph_ratio": _ratio(vision_310_s, vision_910_s),
            "cross_kv_graph_ratio": text["device_ratio"],
            "npu310_d2h_s": producer_310["d2h_s"],
            "npu910_d2h_s": REFERENCE_PRODUCER["d2h_s"],
            "d2h_ratio": _ratio(producer_310["d2h_s"], REFERENCE_PRODUCER["d2h_s"]),
        },
    }
    for section in accounting.values():
        section["surrounding_ratio"] = _ratio(
            float(section["npu310_surrounding_s"]),
            float(section["npu910_surrounding_s"]),
        )

    layout_lab = _compare_layout_lab(layout_310_raw, layout_reference)
    ranked_layout_stages = sorted(
        (
            {"name": name, **values}
            for name, values in layout_lab["stages"].items()
            if name not in {"detector_total_s", "page_file_read_s", "page_image_decode_s"}
        ),
        key=lambda row: float(row["gap_s"]),
        reverse=True,
    )
    layout_lab["ranked_detector_stage_gaps"] = ranked_layout_stages

    candidates = [
        {
            "name": "layout_compiled_graph",
            "gap_s": layout_graph_gap,
            "ratio": layout_graph["device_ratio"],
        },
        {
            "name": "layout_surrounding_work",
            "gap_s": float(accounting["layout"]["npu310_surrounding_s"])
            - float(accounting["layout"]["npu910_surrounding_s"]),
            "ratio": accounting["layout"]["surrounding_ratio"],
        },
        {
            "name": "vision_compiled_graphs",
            "gap_s": vision_310_s - vision_910_s,
            "ratio": accounting["recognition_prefill"]["vision_graph_ratio"],
        },
        {
            "name": "cross_kv_compiled_graph",
            "gap_s": float(text["weighted_gap_s"]),
            "ratio": text["device_ratio"],
        },
        {
            "name": "recognition_surrounding_work",
            "gap_s": float(
                accounting["recognition_prefill"]["npu310_surrounding_s"]
            )
            - float(accounting["recognition_prefill"]["npu910_surrounding_s"]),
            "ratio": accounting["recognition_prefill"]["surrounding_ratio"],
        },
    ]
    candidates.sort(key=lambda row: float(row["gap_s"]), reverse=True)
    return {
        "format": "unirec_prefill_profile_gap_analysis_v1",
        "npu310_environment": graph_310["environment"],
        "npu910_environment": graph_reference["environment"],
        "accounting": accounting,
        "layout_lab": layout_lab,
        "lanes": lane_comparisons,
        "ranked_optimization_targets": candidates,
    }


def _pct(value: float | None) -> float:
    return 100.0 * float(value) if value is not None else 0.0


def _format_top(rows: list[dict[str, Any]], limit: int = 5) -> str:
    values = []
    for row in rows[:limit]:
        name = str(row["name"]).replace(" ", "_")[:64]
        ratio = row.get("ratio")
        ratio_text = f"{float(ratio):.2f}x" if ratio is not None else "new"
        values.append(
            f"{name}:{ratio_text}:{float(row['weighted_first128_delta_s']):.3f}s"
        )
    return ",".join(values)


def print_analysis(analysis: dict[str, Any]) -> None:
    layout = analysis["accounting"]["layout"]
    recognition = analysis["accounting"]["recognition_prefill"]
    lanes = {lane["name"]: lane for lane in analysis["lanes"]}
    top_target = analysis["ranked_optimization_targets"][0]
    print(
        "UNIREC_PROFILE_GAP_HEADLINE "
        f"layout_total={layout['total_ratio']:.2f}x "
        f"layout_graph={layout['graph_ratio']:.2f}x "
        f"layout_surrounding={layout['surrounding_ratio']:.2f}x "
        f"recognition_total={recognition['total_ratio']:.2f}x "
        f"vision_graphs={recognition['vision_graph_ratio']:.2f}x "
        f"cross_kv_graph={recognition['cross_kv_graph_ratio']:.2f}x "
        f"recognition_surrounding={recognition['surrounding_ratio']:.2f}x "
        f"d2h={recognition['d2h_ratio']:.2f}x"
    )
    print(
        "UNIREC_PROFILE_GAP_ACCOUNTING "
        f"layout_graph_gap_share={_pct(layout['graph_gap_share']):.1f}% "
        f"recognition_graph_gap_share={_pct(recognition['graph_gap_share']):.1f}% "
        f"largest_target={top_target['name']} "
        f"largest_target_gap_s={top_target['gap_s']:.3f} "
        f"largest_target_ratio={top_target['ratio']:.2f}x"
    )
    top_layout_stages = analysis["layout_lab"]["ranked_detector_stage_gaps"][:6]
    print(
        "UNIREC_PROFILE_LAYOUT_STAGES "
        + ",".join(
            f"{stage['name']}:{stage['ratio']:.2f}x:{stage['gap_s']:.3f}s"
            for stage in top_layout_stages
        )
    )
    for name in ("layout_b1_800x800_fp32", "vision_960x64_b16_fp16"):
        lane = lanes[name]
        print(
            "UNIREC_PROFILE_KERNEL_DELTA "
            f"lane={name} device_ratio={lane['device_ratio']:.2f}x "
            f"cube_310={lane['cube_utilization_310_pct']:.1f}% "
            f"cube_910={lane['cube_utilization_910_pct']:.1f}% "
            f"top_types={_format_top(lane['top_kernel_type_deltas'])}"
        )
        print(
            "UNIREC_PROFILE_SHAPE_DELTA "
            f"lane={name} top_shapes={_format_top(lane['top_shape_deltas'])}"
        )
    print(
        "UNIREC_PROFILE_NEXT_TARGET "
        f"target={top_target['name']} gap_s={top_target['gap_s']:.3f} "
        f"ratio={top_target['ratio']:.2f}x"
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    analysis = analyze(
        _read(args.npu310_graph_profile),
        _read(args.npu310_layout_lab),
        _read(args.npu310_producer_w1),
        graph_reference=_read(REFERENCE_GRAPH),
        layout_reference=_read(REFERENCE_LAYOUT),
    )
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print_analysis(analysis)


if __name__ == "__main__":
    main()
