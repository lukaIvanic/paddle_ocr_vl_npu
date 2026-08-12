#!/usr/bin/env python3
"""Compare a matched 310P production-vision profile with the 910B2 reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REFERENCE_PRODUCTION = (
    HERE / "references/vision_production_profile_first32_910b_d629c87.json"
)
REFERENCE_GRAPHS = (
    HERE / "references/vision_graph_profile_910b_d629c87_pipe.json"
)
EXPECTED_LANES = (
    "vision_960x64_b16_fp16",
    "vision_512x256_b16_fp16",
    "vision_960x256_b4_fp16",
    "vision_512x512_b8_fp16",
    "vision_960x512_b4_fp16",
)
STRICT_WORKLOAD_KEYS = (
    "page_offset",
    "page_count",
    "page_group_count",
    "page_group_size_histogram",
)
WORKLOAD_KEYS = (
    *STRICT_WORKLOAD_KEYS,
    "crop_count",
    "bucket_calls_per_replay",
    "bucket_real_rows_per_replay",
    "compiled_physical_rows_per_replay",
    "compiled_real_rows_per_replay",
    "fallback_rows_per_replay",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu310-production", type=Path, required=True)
    parser.add_argument("--npu310-graphs", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--topn", type=int, default=12)
    args = parser.parse_args(argv)
    if args.topn < 1:
        parser.error("--topn must be positive")
    return args


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _profile_run(profile: dict[str, Any]) -> dict[str, Any]:
    parsed = profile.get("parsed", profile)
    runs = parsed["summary"]["runs"]
    if len(runs) != 1:
        raise ValueError("each profile must contain exactly one profiler run")
    return runs[0]


def _lane_map(graphs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(lane["name"]): lane for lane in graphs["lanes"]}


def _validate_production(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    for label, report in (("910B2", reference), ("310P", candidate)):
        if report.get("status") != "ok":
            raise ValueError(f"{label} production status is not ok")
        if report.get("profile_scope") != "workload" or not report.get("profile"):
            raise ValueError(f"{label} must contain a workload profile")
        contract = report["production_contract"]
        expected = {
            "worker_count": 1,
            "measured_boundary": "BucketedFullVisionRuntime.encode",
            "input_contract": "compact_uint8_hwc",
            "page_lookahead": 4,
            "page_lookahead_matches_production": True,
            "npu_jit_compile": False,
        }
        for key, value in expected.items():
            if contract.get(key) != value:
                raise ValueError(
                    f"{label} production contract mismatch: "
                    f"{key}={contract.get(key)!r}, expected={value!r}"
                )
    for key in STRICT_WORKLOAD_KEYS:
        if candidate["workload"].get(key) != reference["workload"].get(key):
            raise ValueError(f"production page-group contract mismatch for {key}")
    return {
        key: {
            "npu310": candidate["workload"].get(key),
            "npu910": reference["workload"].get(key),
        }
        for key in WORKLOAD_KEYS
        if candidate["workload"].get(key) != reference["workload"].get(key)
    }


def _validate_graphs(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> None:
    for label, report in (("910B2", reference), ("310P", candidate)):
        if report.get("format") != "unirec_prefill_graph_profile_suite_v1":
            raise ValueError(f"{label} graph profile format mismatch")
        config = report["config"]
        expected = {
            "lane": "vision",
            "warmup": 2,
            "control_repeats": 10,
            "profile_steps": 1,
            "profile_metric": "pipe",
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(
                    f"{label} graph profile mismatch: "
                    f"{key}={config.get(key)!r}, expected={value!r}"
                )
        if set(_lane_map(report)) != set(EXPECTED_LANES):
            raise ValueError(f"{label} graph lane set mismatch")
    ref_lanes = _lane_map(reference)
    candidate_lanes = _lane_map(candidate)
    for name in EXPECTED_LANES:
        ref = ref_lanes[name]
        row = candidate_lanes[name]
        for key in ("input_contract", "first128_calls"):
            if row.get(key) != ref.get(key):
                raise ValueError(f"graph contract mismatch for {name}: {key}")


def _group_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["name"]): row for row in rows}


def _compare_groups(
    candidate_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    topn: int,
    weight: int = 1,
) -> list[dict[str, Any]]:
    candidate = _group_map(candidate_rows)
    reference = _group_map(reference_rows)
    compared = []
    for name in set(candidate) | set(reference):
        npu310_us = float(candidate.get(name, {}).get("duration_us", 0.0))
        npu910_us = float(reference.get(name, {}).get("duration_us", 0.0))
        count310 = int(candidate.get(name, {}).get("count", 0))
        count910 = int(reference.get(name, {}).get("count", 0))
        gap_us = npu310_us - npu910_us
        compared.append(
            {
                "name": name,
                "npu310_us": npu310_us,
                "npu910_us": npu910_us,
                "ratio": _ratio(npu310_us, npu910_us),
                "count310": count310,
                "count910": count910,
                "gap_us": gap_us,
                "weighted_gap_s": gap_us * weight / 1_000_000.0,
            }
        )
    return sorted(compared, key=lambda row: row["weighted_gap_s"], reverse=True)[
        :topn
    ]


def _compare_production_profile(
    reference: dict[str, Any], candidate: dict[str, Any], *, topn: int
) -> dict[str, Any]:
    ref_run = _profile_run(reference["profile"])
    candidate_run = _profile_run(candidate["profile"])
    ref_kernel = ref_run["kernel_details"]
    candidate_kernel = candidate_run["kernel_details"]
    return {
        "kernel_count310": int(candidate_kernel["row_count"]),
        "kernel_count910": int(ref_kernel["row_count"]),
        "kernel_duration_ratio": _ratio(
            float(candidate_kernel["total_duration_us"]),
            float(ref_kernel["total_duration_us"]),
        ),
        "cube_utilization310_pct": float(
            candidate_kernel["weighted_cube_utilization_pct"]
        ),
        "cube_utilization910_pct": float(
            ref_kernel["weighted_cube_utilization_pct"]
        ),
        "kernel_types": _compare_groups(
            candidate_kernel["top_kernel_types"],
            ref_kernel["top_kernel_types"],
            topn=topn,
        ),
        "shape_signatures": _compare_groups(
            candidate_kernel["top_shape_signatures"],
            ref_kernel["top_shape_signatures"],
            topn=topn,
        ),
        "transdata_shapes": _compare_groups(
            candidate_kernel["top_transdata_shape_signatures"],
            ref_kernel["top_transdata_shape_signatures"],
            topn=topn,
        ),
        "matmul_shapes": _compare_groups(
            candidate_kernel["top_matmul_shape_signatures"],
            ref_kernel["top_matmul_shape_signatures"],
            topn=topn,
        ),
    }


def _compare_graphs(
    reference: dict[str, Any], candidate: dict[str, Any], *, topn: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ref_lanes = _lane_map(reference)
    candidate_lanes = _lane_map(candidate)
    lanes = []
    all_kernel_types: dict[str, dict[str, float | int]] = {}
    all_shapes: dict[str, dict[str, float | int]] = {}
    for name in EXPECTED_LANES:
        ref = ref_lanes[name]
        row = candidate_lanes[name]
        calls = int(ref["first128_calls"])
        ref_ms = float(ref["steady_device_event_mean_ms"])
        npu310_ms = float(row["steady_device_event_mean_ms"])
        ref_kernel = _profile_run(ref["parsed_profile"])["kernel_details"]
        candidate_kernel = _profile_run(row["parsed_profile"])["kernel_details"]
        lane = {
            "name": name,
            "first128_calls": calls,
            "npu310_device_ms": npu310_ms,
            "npu910_device_ms": ref_ms,
            "device_ratio": _ratio(npu310_ms, ref_ms),
            "weighted_gap_s": (npu310_ms - ref_ms) * calls / 1000.0,
            "kernel_count310": int(candidate_kernel["row_count"]),
            "kernel_count910": int(ref_kernel["row_count"]),
            "cube_utilization310_pct": float(
                candidate_kernel["weighted_cube_utilization_pct"]
            ),
            "cube_utilization910_pct": float(
                ref_kernel["weighted_cube_utilization_pct"]
            ),
            "top_kernel_types": _compare_groups(
                candidate_kernel["top_kernel_types"],
                ref_kernel["top_kernel_types"],
                topn=topn,
                weight=calls,
            ),
            "top_shape_signatures": _compare_groups(
                candidate_kernel["top_shape_signatures"],
                ref_kernel["top_shape_signatures"],
                topn=topn,
                weight=calls,
            ),
        }
        lanes.append(lane)
        for target, candidate_rows, ref_rows in (
            (
                all_kernel_types,
                candidate_kernel["top_kernel_types"],
                ref_kernel["top_kernel_types"],
            ),
            (
                all_shapes,
                candidate_kernel["top_shape_signatures"],
                ref_kernel["top_shape_signatures"],
            ),
        ):
            candidate_groups = _group_map(candidate_rows)
            ref_groups = _group_map(ref_rows)
            for group_name in set(candidate_groups) | set(ref_groups):
                item = target.setdefault(
                    group_name,
                    {"npu310_us": 0.0, "npu910_us": 0.0, "count310": 0, "count910": 0},
                )
                item["npu310_us"] += float(
                    candidate_groups.get(group_name, {}).get("duration_us", 0.0)
                ) * calls
                item["npu910_us"] += float(
                    ref_groups.get(group_name, {}).get("duration_us", 0.0)
                ) * calls
                item["count310"] += int(
                    candidate_groups.get(group_name, {}).get("count", 0)
                ) * calls
                item["count910"] += int(
                    ref_groups.get(group_name, {}).get("count", 0)
                ) * calls

    def rank(groups: dict[str, dict[str, float | int]]) -> list[dict[str, Any]]:
        result = []
        for name, values in groups.items():
            npu310_us = float(values["npu310_us"])
            npu910_us = float(values["npu910_us"])
            result.append(
                {
                    "name": name,
                    **values,
                    "ratio": _ratio(npu310_us, npu910_us),
                    "weighted_gap_s": (npu310_us - npu910_us) / 1_000_000.0,
                }
            )
        return sorted(result, key=lambda row: row["weighted_gap_s"], reverse=True)[
            :topn
        ]

    aggregate = {
        "weighted_kernel_types": rank(all_kernel_types),
        "weighted_shape_signatures": rank(all_shapes),
    }
    return lanes, aggregate


def _summary(
    production_ref: dict[str, Any],
    production310: dict[str, Any],
    graphs_ref: dict[str, Any],
    graphs310: dict[str, Any],
    lanes: list[dict[str, Any]],
) -> dict[str, Any]:
    ref_device_ms = float(
        production_ref["timing_ms"]["device_timeline_span"]["p50"]
    )
    npu310_device_ms = float(
        production310["timing_ms"]["device_timeline_span"]["p50"]
    )
    ref_calls = production_ref["workload"]["bucket_calls_per_replay"]
    npu310_calls = production310["workload"]["bucket_calls_per_replay"]
    ref_crops = int(production_ref["workload"]["crop_count"])
    npu310_crops = int(production310["workload"]["crop_count"])
    by_lane = {lane["name"]: lane for lane in lanes}
    graph_ref_ms = 0.0
    graph310_ms = 0.0
    for bucket in production_ref["production_contract"]["buckets"]:
        lane = by_lane[f"vision_{bucket}_fp16"]
        graph_ref_ms += float(lane["npu910_device_ms"]) * int(
            ref_calls.get(bucket, 0)
        )
        graph310_ms += float(lane["npu310_device_ms"]) * int(
            npu310_calls.get(bucket, 0)
        )
    surrounding_ref_ms = ref_device_ms - graph_ref_ms
    surrounding310_ms = npu310_device_ms - graph310_ms
    production_per_crop_ref_ms = ref_device_ms / ref_crops
    production_per_crop310_ms = npu310_device_ms / npu310_crops
    graph_per_crop_ref_ms = graph_ref_ms / ref_crops
    graph_per_crop310_ms = graph310_ms / npu310_crops
    surrounding_per_crop_ref_ms = surrounding_ref_ms / ref_crops
    surrounding_per_crop310_ms = surrounding310_ms / npu310_crops
    production_per_crop_gap_ms = (
        production_per_crop310_ms - production_per_crop_ref_ms
    )
    graph_per_crop_gap_ms = graph_per_crop310_ms - graph_per_crop_ref_ms
    return {
        "crop_count": {"npu310": npu310_crops, "npu910": ref_crops},
        "production_device_ms": {"npu310": npu310_device_ms, "npu910": ref_device_ms},
        "production_device_ratio_raw": _ratio(npu310_device_ms, ref_device_ms),
        "production_device_ms_per_crop": {
            "npu310": production_per_crop310_ms,
            "npu910": production_per_crop_ref_ms,
        },
        "production_device_ratio_per_crop": _ratio(
            production_per_crop310_ms, production_per_crop_ref_ms
        ),
        "production_wall_ratio_raw": _ratio(
            float(
                production310["timing_ms"]["production_boundary_wall"]["p50"]
            ),
            float(
                production_ref["timing_ms"]["production_boundary_wall"]["p50"]
            ),
        ),
        "crops_per_s": {
            "npu310": float(production310["throughput"]["crops_per_s_wall_p50"]),
            "npu910": float(production_ref["throughput"]["crops_per_s_wall_p50"]),
        },
        "first32_graph_device_ms": {"npu310": graph310_ms, "npu910": graph_ref_ms},
        "first32_graph_ratio_raw": _ratio(graph310_ms, graph_ref_ms),
        "first32_graph_device_ms_per_crop": {
            "npu310": graph_per_crop310_ms,
            "npu910": graph_per_crop_ref_ms,
        },
        "first32_graph_ratio_per_crop": _ratio(
            graph_per_crop310_ms, graph_per_crop_ref_ms
        ),
        "first32_surrounding_device_ms": {
            "npu310": surrounding310_ms,
            "npu910": surrounding_ref_ms,
        },
        "first32_surrounding_ratio_raw": (
            _ratio(surrounding310_ms, surrounding_ref_ms)
            if surrounding_ref_ms > 0 and surrounding310_ms >= 0
            else None
        ),
        "first32_surrounding_device_ms_per_crop": {
            "npu310": surrounding_per_crop310_ms,
            "npu910": surrounding_per_crop_ref_ms,
        },
        "first32_surrounding_ratio_per_crop": (
            _ratio(surrounding_per_crop310_ms, surrounding_per_crop_ref_ms)
            if surrounding_per_crop_ref_ms > 0 and surrounding_per_crop310_ms >= 0
            else None
        ),
        "graph_share_of_per_crop_production_gap": _ratio(
            graph_per_crop_gap_ms, production_per_crop_gap_ms
        ),
        "first128_graph_device_s": {
            "npu310": float(graphs310["weighted_first128_device_s"]["vision_graphs"]),
            "npu910": float(graphs_ref["weighted_first128_device_s"]["vision_graphs"]),
        },
        "first128_graph_ratio": _ratio(
            float(graphs310["weighted_first128_device_s"]["vision_graphs"]),
            float(graphs_ref["weighted_first128_device_s"]["vision_graphs"]),
        ),
    }


def _fmt_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}x"


def _fmt_fraction(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    production_ref = _read(REFERENCE_PRODUCTION)
    graphs_ref = _read(REFERENCE_GRAPHS)
    production310 = _read(args.npu310_production)
    graphs310 = _read(args.npu310_graphs)
    workload_drift = _validate_production(production_ref, production310)
    _validate_graphs(graphs_ref, graphs310)
    lanes, graph_profile = _compare_graphs(
        graphs_ref, graphs310, topn=args.topn
    )
    summary = _summary(
        production_ref, production310, graphs_ref, graphs310, lanes
    )
    report = {
        "format": "unirec_vision_profile_comparison_v1",
        "devices": {
            "npu310": graphs310["environment"],
            "npu910": graphs_ref["environment"],
        },
        "summary": summary,
        "workload_comparison": {
            "exact_match": not workload_drift,
            "differences": workload_drift,
            "interpretation": (
                "exact production workload"
                if not workload_drift
                else "same page-group contract; production ratios normalized "
                "per crop using each run's own bucket-call histogram"
            ),
        },
        "lanes": lanes,
        "production_profile": _compare_production_profile(
            production_ref, production310, topn=args.topn
        ),
        "graph_profile": graph_profile,
        "references": {
            "production": str(REFERENCE_PRODUCTION),
            "graphs": str(REFERENCE_GRAPHS),
        },
    }
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(
        "UNIREC_VISION_GAP_HEADLINE "
        f"production_per_crop="
        f"{_fmt_ratio(summary['production_device_ratio_per_crop'])} "
        f"graphs_first32_per_crop="
        f"{_fmt_ratio(summary['first32_graph_ratio_per_crop'])} "
        f"surrounding_per_crop="
        f"{_fmt_ratio(summary['first32_surrounding_ratio_per_crop'])} "
        f"graph_gap_share="
        f"{_fmt_fraction(summary['graph_share_of_per_crop_production_gap'])} "
        f"graphs_first128={_fmt_ratio(summary['first128_graph_ratio'])}"
    )
    print(
        "UNIREC_VISION_WORKLOAD_COMPARISON "
        f"exact_match={str(not workload_drift).lower()} "
        f"crops={summary['crop_count']['npu310']}/"
        f"{summary['crop_count']['npu910']} "
        f"differences={json.dumps(workload_drift, sort_keys=True)}"
    )
    for lane in lanes:
        print(
            "UNIREC_VISION_BUCKET_GAP "
            f"name={lane['name']} ratio={_fmt_ratio(lane['device_ratio'])} "
            f"npu310_ms={lane['npu310_device_ms']:.6f} "
            f"npu910_ms={lane['npu910_device_ms']:.6f} "
            f"weighted_gap_s={lane['weighted_gap_s']:.6f} "
            f"kernels={lane['kernel_count310']}/{lane['kernel_count910']} "
            f"cube_pct={lane['cube_utilization310_pct']:.2f}/"
            f"{lane['cube_utilization910_pct']:.2f}"
        )
    for row in graph_profile["weighted_kernel_types"][:6]:
        print(
            "UNIREC_VISION_KERNEL_GAP "
            f"name={json.dumps(row['name'])} ratio={_fmt_ratio(row['ratio'])} "
            f"weighted_gap_s={row['weighted_gap_s']:.6f} "
            f"counts={row['count310']}/{row['count910']}"
        )
    for row in graph_profile["weighted_shape_signatures"][:6]:
        print(
            "UNIREC_VISION_SHAPE_GAP "
            f"name={json.dumps(row['name'])} ratio={_fmt_ratio(row['ratio'])} "
            f"weighted_gap_s={row['weighted_gap_s']:.6f} "
            f"counts={row['count310']}/{row['count910']}"
        )
    print(
        "UNIREC_VISION_PROFILE_COUNTS "
        f"kernels={report['production_profile']['kernel_count310']}/"
        f"{report['production_profile']['kernel_count910']} "
        f"cube_pct={report['production_profile']['cube_utilization310_pct']:.2f}/"
        f"{report['production_profile']['cube_utilization910_pct']:.2f}"
    )
    if args.output:
        print(f"UNIREC_VISION_COMPARISON_OUTPUT {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
