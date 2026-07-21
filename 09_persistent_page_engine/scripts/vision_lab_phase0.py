#!/usr/bin/env python3
"""Run the zero-new-graph vision-throughput Phase 0 experiment set."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.modeling import LocalPaddleOCRVLForConditionalGeneration
from paddleocr_vl.model.vision_prefill import (
    VisionPrefillRuntime,
    select_vision_bucket,
    vision_cache_dir_for_bucket,
)
from paddleocr_vl.serving.runtime_defaults import OPTIMIZED_VISION_BUCKETS
from utils.timing import synchronize
from vision_lab import (
    DEFAULT_MODEL,
    _environment,
    _materialize_inputs,
    _memory_baseline,
    _peak_memory_delta,
    _route,
    _run_group,
)


DEFAULT_CORPUS = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/vision_lab"
    / "corpus_recognition_trace_variants.json"
)
DEFAULT_BASELINE_REPORT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/vision_lab"
    / "calibration_single_torchair_pfa_3a7dc3c.json"
)
DEFAULT_PACKED_REPORT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/vision_lab"
    / "matrix_packed_pfa_torchair_3a7dc3c.json"
)
DEFAULT_CACHE_ROOT = REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_torchair"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/vision_lab"
    / "phase0_throughput_report.json"
)
POLICIES = ("first_fit", "first_fit_decreasing", "best_fit")
DEFAULT_TARGETS = (704, 1920, 2560, 3072, 4096, 6144)
VARIANT_TARGETS = (1024, 1536, 1920)
VALIDATION_TARGET = 1920
MAX_COMPILED_LADDER = max(OPTIMIZED_VISION_BUCKETS)


def _csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(piece) for piece in value.split(",") if piece.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--packed-report", type=Path, default=DEFAULT_PACKED_REPORT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variant", default="min_pixels_28224")
    parser.add_argument("--default-targets", type=_csv_ints, default=DEFAULT_TARGETS)
    parser.add_argument("--variant-targets", type=_csv_ints, default=VARIANT_TARGETS)
    parser.add_argument("--validation-target", type=int, default=VALIDATION_TARGET)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--control-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--skip-like-for-like",
        action="store_true",
        help="Skip numerics and run the speed/padding study only.",
    )
    parser.add_argument("--pipeline-vision-s", type=float, default=10.7)
    parser.add_argument("--pipeline-e2e-s", type=float, default=32.8)
    parser.add_argument("--device-occupancy", type=float, default=0.92)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats positive")
    if args.validation_target not in OPTIMIZED_VISION_BUCKETS:
        parser.error("--validation-target must already exist in the compiled ladder")
    if not 0.0 < args.device_occupancy <= 1.0:
        parser.error("--device-occupancy must be in (0, 1]")
    return args


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "min": min(values) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values) if values else 0.0,
        "mean": statistics.mean(values) if values else 0.0,
    }


def _load_corpora(path: Path, variant: str) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not payload.get("self_check", {}).get("passed"):
        raise ValueError("default corpus regression self-check is not marked passed")
    if variant not in payload.get("variants", {}):
        raise KeyError(f"corpus does not contain requested variant {variant!r}")
    corpora = {
        "default": [dict(item) for item in payload["items"]],
        variant: [dict(item) for item in payload["variants"][variant]["items"]],
    }
    for items in corpora.values():
        for index, item in enumerate(items):
            item["source_index"] = int(item.get("source_index", index))
    return payload, corpora


def _corpus_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = [int(item["real_vision_tokens"]) for item in items]
    overflows = [value for value in tokens if value > MAX_COMPILED_LADDER]
    return {
        "items": len(items),
        "total_real_vision_tokens": sum(tokens),
        "tokens_per_crop": _summary(tokens),
        "eager_overflow_crops_excluded": len(overflows),
        "eager_overflow_tokens_excluded": sum(overflows),
    }


def _packing_groups(
    items: list[dict[str, Any]],
    *,
    target: int,
    policy: str,
) -> list[list[dict[str, Any]]]:
    if policy not in POLICIES:
        raise ValueError(f"unknown packing policy {policy!r}")
    order = list(items)
    if policy == "first_fit_decreasing":
        order.sort(key=lambda item: int(item["real_vision_tokens"]), reverse=True)
    groups: list[list[dict[str, Any]]] = []
    totals: list[int] = []
    for item in order:
        length = int(item["real_vision_tokens"])
        if length > target:
            groups.append([item])
            totals.append(length)
            continue
        candidates = [
            index for index, total in enumerate(totals) if total + length <= target
        ]
        if not candidates:
            groups.append([item])
            totals.append(length)
            continue
        selected = (
            max(candidates, key=lambda index: totals[index])
            if policy == "best_fit"
            else candidates[0]
        )
        groups[selected].append(item)
        totals[selected] += length
    if sum(len(group) for group in groups) != len(items):
        raise AssertionError("packing policy lost corpus items")
    return groups


def _group_kind(group: list[dict[str, Any]], target: int) -> str:
    total = sum(int(item["real_vision_tokens"]) for item in group)
    if total > target:
        return "passthrough"
    if total <= min(MAX_COMPILED_LADDER, target // 2):
        return "leftover_tail"
    return "target_graph"


def _tensor_delta(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - reference.float()).abs()
    relative = delta / reference.float().abs().clamp_min(1e-6)
    return {
        "max_abs": float(delta.max().item()),
        "max_rel": float(relative.max().item()),
        "mean_abs": float(delta.mean().item()),
    }


def _ratio_histogram(values: Iterable[float]) -> dict[str, int]:
    histogram = {"le_0.5": 0, "le_1": 0, "le_2": 0, "le_4": 0, "gt_4": 0}
    for value in values:
        if value <= 0.5:
            histogram["le_0.5"] += 1
        elif value <= 1.0:
            histogram["le_1"] += 1
        elif value <= 2.0:
            histogram["le_2"] += 1
        elif value <= 4.0:
            histogram["le_4"] += 1
        else:
            histogram["gt_4"] += 1
    return histogram


def _padding_audit(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    real_tokens = sum(int(case["real_tokens"]) for case in cases)
    physical_tokens = sum(int(case["physical_tokens"]) for case in cases)
    if physical_tokens < real_tokens:
        raise AssertionError(
            f"physical token accounting underflow: real={real_tokens}, "
            f"physical={physical_tokens}"
        )
    padding_tokens = physical_tokens - real_tokens
    for case in cases:
        real = int(case["real_tokens"])
        physical = int(case["physical_tokens"])
        if physical < real:
            raise AssertionError(
                f"case padding underflow: real={real}, physical={physical}"
            )
        if case["kind"] == "target_graph" and physical != int(case["target"]):
            raise AssertionError(
                "target-graph call did not use its exact static length: "
                f"target={case['target']}, physical={physical}"
            )
    return {
        "real_tokens": real_tokens,
        "physical_tokens": physical_tokens,
        "padding_tokens": padding_tokens,
        "real_token_fraction": (
            real_tokens / physical_tokens if physical_tokens else 1.0
        ),
        "padding_fraction": (
            padding_tokens / physical_tokens if physical_tokens else 0.0
        ),
        "max_case_padding_tokens": max(
            (int(case["physical_tokens"]) - int(case["real_tokens"]) for case in cases),
            default=0,
        ),
        "checked_cases": len(cases),
        "passed": True,
    }


def _cache_preflight(
    *,
    cache_root: Path,
    buckets: Iterable[int],
    model: LocalPaddleOCRVLForConditionalGeneration,
    model_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    hidden_size = int(model.config.vision_config.hidden_size)
    head_dim = hidden_size // int(model.config.vision_config.num_attention_heads)
    rows: list[dict[str, Any]] = []
    missing: list[int] = []
    for bucket in sorted(set(int(value) for value in buckets)):
        cache_dir = vision_cache_dir_for_bucket(
            cache_root,
            bucket=bucket,
            dtype=dtype,
            device=device,
            model_dir=model_dir,
            attention_impl="prompt_flash_attention",
            head_dim=head_dim,
        )
        populated = cache_dir.is_dir() and any(cache_dir.iterdir())
        rows.append({"bucket": bucket, "path": str(cache_dir), "populated": populated})
        if not populated:
            missing.append(bucket)
    if missing:
        raise RuntimeError(
            "Phase 0 refuses to compile missing graphs; absent warm buckets: "
            + ",".join(map(str, missing))
        )
    return {"new_graph_budget_spent": 0, "required_warm_buckets": rows}


def _report_bucket_costs(path: Path) -> tuple[dict[int, float], float]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    configuration = next(
        item for item in payload["configurations"] if item["strategy"] == "single"
    )
    by_bucket: dict[int, list[float]] = defaultdict(list)
    overflow_s = 0.0
    for case in configuration["cases"]:
        mean_ms = float(case["device_ms"]["vision_tower"]["mean"])
        bucket = case["route"].get("bucket")
        if bucket is None:
            overflow_s += mean_ms / 1000.0
        else:
            by_bucket[int(bucket)].append(mean_ms)
    return {
        bucket: statistics.mean(samples) for bucket, samples in by_bucket.items()
    }, overflow_s


def _packed_anchor_cost(path: Path, target: int) -> float:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    configuration = next(
        item
        for item in payload["configurations"]
        if item["strategy"] == "packed" and int(item["target_length"]) == target
    )
    samples = [
        float(case["device_ms"]["vision_tower"]["mean"])
        for case in configuration["cases"]
        if case["route"].get("bucket") == target
    ]
    if not samples:
        raise ValueError(f"packed report has no physical-{target} cases")
    return statistics.mean(samples)


def _compiled_single_output(
    model: LocalPaddleOCRVLForConditionalGeneration,
    runtime: VisionPrefillRuntime,
    item: dict[str, Any],
    grid: torch.Tensor,
    pixels: torch.Tensor,
    *,
    physical: int,
    buckets: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    route = _route(
        strategy="single",
        lengths=[int(item["real_vision_tokens"])],
        execution="torchair",
        buckets=buckets,
        forced_bucket=physical,
    )
    run = _run_group(
        model,
        runtime,
        [item],
        strategy="single",
        route=route,
        grids=[grid],
        pixels=[pixels],
        device=device,
    )
    return run["projector"][0].detach().cpu()


def _like_for_like_validation(
    *,
    name: str,
    items: list[dict[str, Any]],
    target: int,
    model: LocalPaddleOCRVLForConditionalGeneration,
    runtime: VisionPrefillRuntime,
    buckets: tuple[int, ...],
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    multiplier: float,
) -> dict[str, Any]:
    eligible = [
        item
        for item in items
        if int(item["real_vision_tokens"]) <= target
        and int(item["real_vision_tokens"]) <= MAX_COMPILED_LADDER
    ]
    skipped = [item for item in items if item not in eligible]
    groups = _packing_groups(eligible, target=target, policy="best_fit")
    rows: list[dict[str, Any]] = []
    timing_s = 0.0
    for group in groups:
        grids, pixels = _materialize_inputs(
            group,
            seed=seed,
            dtype=dtype,
            device=device,
        )
        native_outputs: list[torch.Tensor] = []
        control_outputs: list[torch.Tensor] = []
        for item, grid, pixel in zip(group, grids, pixels):
            native_bucket = select_vision_bucket(
                int(item["real_vision_tokens"]), buckets
            )
            if native_bucket is None:
                raise AssertionError("like-for-like item escaped compiled ladder")
            native_outputs.append(
                _compiled_single_output(
                    model,
                    runtime,
                    item,
                    grid,
                    pixel,
                    physical=native_bucket,
                    buckets=buckets,
                    device=device,
                )
            )
            control_outputs.append(
                _compiled_single_output(
                    model,
                    runtime,
                    item,
                    grid,
                    pixel,
                    physical=target,
                    buckets=buckets,
                    device=device,
                )
            )
        packed_route = _route(
            strategy="packed",
            lengths=[int(item["real_vision_tokens"]) for item in group],
            execution="torchair",
            buckets=buckets,
            forced_bucket=target,
        )
        packed = _run_group(
            model,
            runtime,
            group,
            strategy="packed",
            route=packed_route,
            grids=grids,
            pixels=pixels,
            device=device,
        )
        timing_s += float(packed["stage_ms"]["vision_tower"]) / 1000.0
        for item, native, control, actual in zip(
            group,
            native_outputs,
            control_outputs,
            packed["projector"],
        ):
            rows.append(
                {
                    "name": item["name"],
                    "grid_thw": list(item["grid_thw"]),
                    "real_vision_tokens": int(item["real_vision_tokens"]),
                    "control_padded_single_vs_native": _tensor_delta(control, native),
                    "packed_vs_native": _tensor_delta(actual.detach().cpu(), native),
                }
            )
        del grids, pixels, native_outputs, control_outputs, packed

    control_abs = [row["control_padded_single_vs_native"]["max_abs"] for row in rows]
    control_rel = [row["control_padded_single_vs_native"]["max_rel"] for row in rows]
    packed_abs = [row["packed_vs_native"]["max_abs"] for row in rows]
    packed_rel = [row["packed_vs_native"]["max_rel"] for row in rows]
    thresholds = {
        "max_abs": max(1e-6, multiplier * _percentile(control_abs, 0.99)),
        "max_rel": max(1e-6, multiplier * _percentile(control_rel, 0.99)),
    }
    for row in rows:
        row["control_ratio"] = {
            metric: row["packed_vs_native"][metric]
            / max(row["control_padded_single_vs_native"][metric], 1e-12)
            for metric in ("max_abs", "max_rel")
        }
        row["within_calibrated_threshold"] = all(
            row["packed_vs_native"][metric] <= thresholds[metric]
            for metric in ("max_abs", "max_rel")
        )
    failures = [row for row in rows if not row["within_calibrated_threshold"]]
    worst = sorted(
        rows,
        key=lambda row: max(
            row["packed_vs_native"][metric] / thresholds[metric]
            for metric in ("max_abs", "max_rel")
        ),
        reverse=True,
    )[:5]
    return {
        "corpus": name,
        "target": target,
        "policy": "best_fit",
        "reference": "single_promptfa_compiled_native_bucket",
        "control": "single_promptfa_compiled_padded_to_target_vs_native_bucket",
        "threshold_calibration": {
            "method": "control p99 multiplied by control_multiplier",
            "control_multiplier": multiplier,
            "thresholds": thresholds,
            "control_distribution": {
                "max_abs": _summary(control_abs),
                "max_rel": _summary(control_rel),
            },
        },
        "packed_distribution": {
            "max_abs": _summary(packed_abs),
            "max_rel": _summary(packed_rel),
            "ratio_to_same_crop_control": {
                "max_abs": _ratio_histogram(
                    row["control_ratio"]["max_abs"] for row in rows
                ),
                "max_rel": _ratio_histogram(
                    row["control_ratio"]["max_rel"] for row in rows
                ),
            },
        },
        "validated_crops": len(rows),
        "skipped_crops": [
            {
                "name": item["name"],
                "grid_thw": list(item["grid_thw"]),
                "real_vision_tokens": int(item["real_vision_tokens"]),
                "reason": "larger_than_validation_target_or_compiled_ladder",
            }
            for item in skipped
        ],
        "groups": len(groups),
        "measured_packed_vision_s": timing_s,
        "failures": len(failures),
        "worst_5": worst,
        "per_crop": rows,
        "verdict": "valid" if not failures else "invalid",
        "valid": not failures,
    }


def _measure_group(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    runtime: VisionPrefillRuntime,
    group: list[dict[str, Any]],
    strategy: str,
    route: dict[str, Any],
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    grids, pixels = _materialize_inputs(group, seed=seed, dtype=dtype, device=device)
    for _ in range(warmup):
        _run_group(
            model,
            runtime,
            group,
            strategy=strategy,
            route=route,
            grids=grids,
            pixels=pixels,
            device=device,
        )
    baseline = _memory_baseline(device)
    samples: list[float] = []
    for _ in range(repeats):
        run = _run_group(
            model,
            runtime,
            group,
            strategy=strategy,
            route=route,
            grids=grids,
            pixels=pixels,
            device=device,
        )
        samples.append(float(run["stage_ms"]["vision_tower"]))
    peak = _peak_memory_delta(device, baseline)
    del grids, pixels
    return {
        "mean_ms": statistics.mean(samples),
        "samples_ms": samples,
        "peak_allocated_bytes_delta": peak,
    }


def _scout_target(
    *,
    items: list[dict[str, Any]],
    target: int,
    model: LocalPaddleOCRVLForConditionalGeneration,
    runtime: VisionPrefillRuntime,
    buckets: tuple[int, ...],
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    eligible = [item for item in items if int(item["real_vision_tokens"]) <= MAX_COMPILED_LADDER]
    groups = _packing_groups(eligible, target=target, policy="best_fit")
    cases: list[dict[str, Any]] = []
    warmed_target = False
    for group in groups:
        total = sum(int(item["real_vision_tokens"]) for item in group)
        kind = _group_kind(group, target)
        if kind == "target_graph":
            route = _route(
                strategy="packed",
                lengths=[int(item["real_vision_tokens"]) for item in group],
                execution="eager",
                buckets=buckets,
                forced_bucket=target,
            )
            strategy = "packed"
        elif kind == "leftover_tail":
            route = _route(
                strategy="packed",
                lengths=[int(item["real_vision_tokens"]) for item in group],
                execution="eager",
                buckets=buckets,
            )
            strategy = "packed"
        else:
            route = _route(
                strategy="single",
                lengths=[int(group[0]["real_vision_tokens"])],
                execution="eager",
                buckets=buckets,
            )
            strategy = "single"
        measured = _measure_group(
            model=model,
            runtime=runtime,
            group=group,
            strategy=strategy,
            route=route,
            seed=seed,
            dtype=dtype,
            device=device,
            warmup=warmup if kind == "target_graph" and not warmed_target else 0,
            repeats=repeats,
        )
        warmed_target = warmed_target or kind == "target_graph"
        cases.append(
            {
                "kind": kind,
                "target": target,
                "items": len(group),
                "real_tokens": total,
                "physical_tokens": int(route["physical_vision_tokens"]),
                **measured,
            }
        )
    target_samples = [case["mean_ms"] for case in cases if case["kind"] == "target_graph"]
    return {
        "target": target,
        "policy": "best_fit",
        "groups": len(cases),
        "group_kinds": {
            kind: sum(case["kind"] == kind for case in cases)
            for kind in ("target_graph", "leftover_tail", "passthrough")
        },
        "measured_eager_vision_s": sum(case["mean_ms"] for case in cases) / 1000.0,
        "target_graph_mean_ms": statistics.mean(target_samples) if target_samples else None,
        "target_graph_real_tokens_per_s": (
            target / (statistics.mean(target_samples) / 1000.0)
            if target_samples
            else None
        ),
        "peak_allocated_bytes_delta": max(
            (case["peak_allocated_bytes_delta"] or 0) for case in cases
        ),
        "padding_audit": _padding_audit(cases),
        "cases": cases,
    }


def _existing_bucket_cost(
    real_tokens: int,
    *,
    bucket_cost_ms: dict[int, float],
    fallback_ms_per_token: float,
) -> float:
    bucket = select_vision_bucket(real_tokens, OPTIMIZED_VISION_BUCKETS)
    if bucket is not None and bucket in bucket_cost_ms:
        return bucket_cost_ms[bucket]
    return real_tokens * fallback_ms_per_token


def _policy_projection(
    *,
    items: list[dict[str, Any]],
    target: int,
    policy: str,
    target_compiled_ms: float,
    bucket_cost_ms: dict[int, float],
    overflow_constant_s: float,
    fallback_ms_per_token: float,
) -> dict[str, Any]:
    eligible = [item for item in items if int(item["real_vision_tokens"]) <= MAX_COMPILED_LADDER]
    groups = _packing_groups(eligible, target=target, policy=policy)
    costs_ms: list[float] = []
    kinds: list[str] = []
    totals: list[int] = []
    physical_totals: list[int] = []
    for group in groups:
        total = sum(int(item["real_vision_tokens"]) for item in group)
        kind = _group_kind(group, target)
        kinds.append(kind)
        totals.append(total)
        if kind == "target_graph":
            costs_ms.append(target_compiled_ms)
            physical_totals.append(target)
        elif kind == "passthrough":
            costs_ms.append(
                sum(
                    _existing_bucket_cost(
                        int(item["real_vision_tokens"]),
                        bucket_cost_ms=bucket_cost_ms,
                        fallback_ms_per_token=fallback_ms_per_token,
                    )
                    for item in group
                )
            )
            physical_totals.append(
                sum(
                    select_vision_bucket(
                        int(item["real_vision_tokens"]), OPTIMIZED_VISION_BUCKETS
                    )
                    or int(item["real_vision_tokens"])
                    for item in group
                )
            )
        else:
            costs_ms.append(
                _existing_bucket_cost(
                    total,
                    bucket_cost_ms=bucket_cost_ms,
                    fallback_ms_per_token=fallback_ms_per_token,
                )
            )
            physical_totals.append(
                select_vision_bucket(total, OPTIMIZED_VISION_BUCKETS) or total
            )
    real_tokens = sum(totals)
    physical_tokens = sum(physical_totals)
    if physical_tokens < real_tokens:
        raise AssertionError("policy projection physical-token underflow")
    projection_s = sum(costs_ms) / 1000.0 + overflow_constant_s
    return {
        "policy": policy,
        "groups": len(groups),
        "target_graph_groups": kinds.count("target_graph"),
        "leftover_tail_groups": kinds.count("leftover_tail"),
        "passthrough_groups": kinds.count("passthrough"),
        "mean_fill_fraction": statistics.mean(
            min(total, target) / target for total in totals
        ),
        "padding_audit": {
            "real_tokens": real_tokens,
            "physical_tokens": physical_tokens,
            "padding_tokens": physical_tokens - real_tokens,
            "real_token_fraction": real_tokens / physical_tokens,
            "padding_fraction": (physical_tokens - real_tokens) / physical_tokens,
            "max_group_padding_tokens": max(
                physical - real
                for physical, real in zip(physical_totals, totals)
            ),
            "passed": True,
        },
        "projected_compiled_vision_s": projection_s,
        "overflow_constant_s": overflow_constant_s,
    }


def _measure_existing_compiled(
    *,
    items: list[dict[str, Any]],
    target: int,
    policy: str,
    model: LocalPaddleOCRVLForConditionalGeneration,
    runtime: VisionPrefillRuntime,
    buckets: tuple[int, ...],
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if target not in buckets:
        raise ValueError(f"exact compiled target {target} is not warm")
    eligible = [item for item in items if int(item["real_vision_tokens"]) <= MAX_COMPILED_LADDER]
    groups = _packing_groups(eligible, target=target, policy=policy)
    cases: list[dict[str, Any]] = []
    warmed_target = False
    for group in groups:
        lengths = [int(item["real_vision_tokens"]) for item in group]
        total = sum(lengths)
        kind = _group_kind(group, target)
        if kind == "target_graph":
            route = _route(
                strategy="packed",
                lengths=lengths,
                execution="torchair",
                buckets=buckets,
                forced_bucket=target,
            )
            strategy = "packed"
        elif kind == "leftover_tail":
            route = _route(
                strategy="packed",
                lengths=lengths,
                execution="torchair",
                buckets=buckets,
            )
            strategy = "packed"
        else:
            route = _route(
                strategy="single",
                lengths=[lengths[0]],
                execution="torchair",
                buckets=buckets,
            )
            strategy = "single"
        measured = _measure_group(
            model=model,
            runtime=runtime,
            group=group,
            strategy=strategy,
            route=route,
            seed=seed,
            dtype=dtype,
            device=device,
            warmup=warmup if kind == "target_graph" and not warmed_target else 0,
            repeats=repeats,
        )
        warmed_target = warmed_target or kind == "target_graph"
        cases.append(
            {
                "kind": kind,
                "target": target,
                "items": len(group),
                "real_tokens": total,
                "physical_tokens": int(route["physical_vision_tokens"]),
                **measured,
            }
        )
    return {
        "target": target,
        "policy": policy,
        "new_graphs": 0,
        "groups": len(cases),
        "projected_vision_s": sum(case["mean_ms"] for case in cases) / 1000.0,
        "peak_allocated_bytes_delta": max(
            (case["peak_allocated_bytes_delta"] or 0) for case in cases
        ),
        "padding_audit": _padding_audit(cases),
        "cases": cases,
    }


def _headline_row(
    *,
    corpus: str,
    strategy: str,
    target: int | str,
    new_graphs: int,
    vision_s: float,
    verdict: str,
    peak_mem: int | None,
    pipeline_vision_s: float,
    pipeline_e2e_s: float,
    occupancy: float,
) -> dict[str, Any]:
    return {
        "corpus": corpus,
        "strategy_length": f"{strategy}@{target}",
        "new_graphs": new_graphs,
        "projected_corpus_vision_s": vision_s,
        "ratio_vs_10.7_pipeline": vision_s / pipeline_vision_s,
        "like_for_like_verdict": verdict,
        "peak_allocated_bytes_delta": peak_mem,
        "projected_e2e_s_at_device_occupancy": (
            pipeline_e2e_s - (pipeline_vision_s - vision_s) / occupancy
        ),
    }


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    import torch_npu

    device = torch.device("npu:0")
    if not torch.npu.is_available():
        raise RuntimeError("Phase 0 vision lab requires an available NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    dtype = torch.float16
    model_dir = args.model.expanduser().resolve()
    corpus_payload, corpora = _load_corpora(args.corpus, args.variant)
    buckets = tuple(int(value) for value in OPTIMIZED_VISION_BUCKETS)
    needed_buckets = {
        int(args.validation_target),
        *(
            int(target)
            for target in args.variant_targets
            if int(target) <= MAX_COMPILED_LADDER
        ),
    }
    for items in corpora.values():
        needed_buckets.update(
            bucket
            for item in items
            if int(item["real_vision_tokens"]) <= MAX_COMPILED_LADDER
            for bucket in [
                select_vision_bucket(int(item["real_vision_tokens"]), buckets)
            ]
            if bucket is not None
        )

    synchronize(device)
    setup_started = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=dtype,
        device=device,
    )
    cache_preflight = _cache_preflight(
        cache_root=args.cache_dir.expanduser().resolve(),
        buckets=needed_buckets,
        model=model,
        model_dir=model_dir,
        device=device,
        dtype=dtype,
    )
    compiled_runtime = VisionPrefillRuntime(
        model,
        backend="torchair",
        buckets=sorted(needed_buckets),
        cache_root=args.cache_dir.expanduser().resolve(),
        device=device,
        dtype=dtype,
        model_dir=model_dir,
        attention_impl="prompt_flash_attention",
        padding="bucket",
    )
    eager_runtime = VisionPrefillRuntime(
        model,
        backend="raw_eager",
        buckets=(buckets[0],),
        cache_root=args.cache_dir.expanduser().resolve() / "eager_unused",
        device=device,
        dtype=dtype,
        model_dir=model_dir,
        attention_impl="prompt_flash_attention",
        padding="none",
    )
    synchronize(device)
    setup_s = time.perf_counter() - setup_started

    validations: dict[str, Any] = {}
    if args.skip_like_for_like:
        validations = {
            name: {
                "verdict": "skipped_speed_only",
                "valid": None,
                "reason": "explicit --skip-like-for-like",
            }
            for name in corpora
        }
    else:
        for name, items in corpora.items():
            validations[name] = _like_for_like_validation(
                name=name,
                items=items,
                target=args.validation_target,
                model=model,
                runtime=compiled_runtime,
                buckets=buckets,
                seed=args.seed,
                dtype=dtype,
                device=device,
                multiplier=args.control_multiplier,
            )
    run_speed_study = args.skip_like_for_like or all(
        item["valid"] for item in validations.values()
    )

    eager_scout: dict[str, list[dict[str, Any]]] = {}
    if run_speed_study:
        for name, items in corpora.items():
            targets = args.default_targets if name == "default" else args.variant_targets
            eager_scout[name] = [
                _scout_target(
                    items=items,
                    target=int(target),
                    model=model,
                    runtime=eager_runtime,
                    buckets=buckets,
                    seed=args.seed,
                    dtype=dtype,
                    device=device,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
                for target in targets
            ]

    bucket_cost_ms, default_overflow_s = _report_bucket_costs(args.baseline_report)
    anchor_compiled_ms = {
        704: bucket_cost_ms[704],
        1920: _packed_anchor_cost(args.packed_report, 1920),
    }
    anchors: dict[str, Any] = {}
    extrapolation_ratio: float | None = None
    if run_speed_study:
        default_by_target = {
            int(row["target"]): row for row in eager_scout["default"]
        }
        for target in (704, 1920):
            eager_ms = default_by_target[target]["target_graph_mean_ms"]
            anchors[str(target)] = {
                "compiled_ms": anchor_compiled_ms[target],
                "eager_ms": eager_ms,
                "compiled_over_eager": anchor_compiled_ms[target] / eager_ms,
            }
        extrapolation_ratio = statistics.mean(
            item["compiled_over_eager"] for item in anchors.values()
        )

    policy_analysis: dict[str, Any] = {}
    exact_variant_compiled: list[dict[str, Any]] = []
    headline_rows: list[dict[str, Any]] = []
    phase1_candidates: list[dict[str, Any]] = []
    if run_speed_study and extrapolation_ratio is not None:
        for name, items in corpora.items():
            rows_by_target = {
                int(row["target"]): row for row in eager_scout[name]
            }
            target_rows: list[dict[str, Any]] = []
            for target, scout in rows_by_target.items():
                eager_target_ms = float(scout["target_graph_mean_ms"])
                compiled_target_ms = (
                    anchor_compiled_ms[target]
                    if target in anchor_compiled_ms
                    else bucket_cost_ms[target]
                    if target in bucket_cost_ms
                    else eager_target_ms * extrapolation_ratio
                )
                real_ms_per_token = eager_target_ms / target * extrapolation_ratio
                overflow_s = default_overflow_s if name == "default" else 0.0
                policies = [
                    _policy_projection(
                        items=items,
                        target=target,
                        policy=policy,
                        target_compiled_ms=compiled_target_ms,
                        bucket_cost_ms=bucket_cost_ms,
                        overflow_constant_s=overflow_s,
                        fallback_ms_per_token=real_ms_per_token,
                    )
                    for policy in POLICIES
                ]
                best = min(policies, key=lambda item: item["projected_compiled_vision_s"])
                worst = max(policies, key=lambda item: item["projected_compiled_vision_s"])
                target_rows.append(
                    {
                        "target": target,
                        "compiled_target_ms": compiled_target_ms,
                        "compiled_cost_source": (
                            "measured_existing"
                            if target in anchor_compiled_ms or target in bucket_cost_ms
                            else "eager_times_mean_anchor_ratio"
                        ),
                        "policies": policies,
                        "best_policy": best["policy"],
                        "best_projected_compiled_vision_s": best[
                            "projected_compiled_vision_s"
                        ],
                        "best_minus_worst_s": best["projected_compiled_vision_s"]
                        - worst["projected_compiled_vision_s"],
                        "worst_minus_best_s": worst["projected_compiled_vision_s"]
                        - best["projected_compiled_vision_s"],
                    }
                )
            policy_analysis[name] = target_rows

        ordered_default = sorted(
            policy_analysis["default"], key=lambda row: int(row["target"])
        )
        previous_s: float | None = None
        for row in ordered_default:
            target = int(row["target"])
            current_s = float(row["best_projected_compiled_vision_s"])
            if previous_s is not None and target > MAX_COMPILED_LADDER:
                improvement = (previous_s - current_s) / previous_s
                candidate = {
                    "target": target,
                    "policy": row["best_policy"],
                    "estimated_vision_s": current_s,
                    "incremental_gain_fraction_vs_previous_point": improvement,
                    "passes_5_percent_compile_gate": improvement >= 0.05,
                    "new_graphs": 1,
                }
                if candidate["passes_5_percent_compile_gate"]:
                    phase1_candidates.append(candidate)
            previous_s = current_s

        variant_policy_by_target = {
            int(row["target"]): str(row["best_policy"])
            for row in policy_analysis[args.variant]
        }
        for target in args.variant_targets:
            exact_variant_compiled.append(
                _measure_existing_compiled(
                    items=corpora[args.variant],
                    target=int(target),
                    policy=variant_policy_by_target[int(target)],
                    model=model,
                    runtime=compiled_runtime,
                    buckets=buckets,
                    seed=args.seed,
                    dtype=dtype,
                    device=device,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            )

        headline_rows.append(
            _headline_row(
                corpus="default",
                strategy="single_current",
                target="ladder",
                new_graphs=0,
                vision_s=10.508171070480346,
                verdict="calibrated_baseline",
                peak_mem=None,
                pipeline_vision_s=args.pipeline_vision_s,
                pipeline_e2e_s=args.pipeline_e2e_s,
                occupancy=args.device_occupancy,
            )
        )
        default_1920 = next(
            row for row in policy_analysis["default"] if int(row["target"]) == 1920
        )
        default_1920_s = float(default_1920["best_projected_compiled_vision_s"])
        default_1920_peak = next(
            row["peak_allocated_bytes_delta"]
            for row in eager_scout["default"]
            if int(row["target"]) == 1920
        )
        headline_rows.append(
            _headline_row(
                corpus="default",
                strategy=f"packed_{default_1920['best_policy']}",
                target=1920,
                new_graphs=0,
                vision_s=default_1920_s,
                verdict=validations["default"]["verdict"],
                peak_mem=default_1920_peak,
                pipeline_vision_s=args.pipeline_vision_s,
                pipeline_e2e_s=args.pipeline_e2e_s,
                occupancy=args.device_occupancy,
            )
        )
        for row in exact_variant_compiled:
            headline_rows.append(
                _headline_row(
                    corpus=args.variant,
                    strategy=f"packed_{row['policy']}",
                    target=int(row["target"]),
                    new_graphs=0,
                    vision_s=float(row["projected_vision_s"]),
                    verdict=validations[args.variant]["verdict"],
                    peak_mem=row["peak_allocated_bytes_delta"],
                    pipeline_vision_s=args.pipeline_vision_s,
                    pipeline_e2e_s=args.pipeline_e2e_s,
                    occupancy=args.device_occupancy,
                )
            )

    payload = {
        "schema_version": 2,
        "created_at_unix_s": time.time(),
        "phase": "phase0_zero_new_graphs",
        "status": (
            "complete_speed_only"
            if args.skip_like_for_like
            else "complete"
            if run_speed_study
            else "stopped_after_invalid_numerics"
        ),
        "config": {
            "corpus": str(args.corpus.expanduser().resolve()),
            "variant": args.variant,
            "default_targets": list(args.default_targets),
            "variant_targets": list(args.variant_targets),
            "validation_target": args.validation_target,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "skip_like_for_like": args.skip_like_for_like,
            "cache_dir": str(args.cache_dir.expanduser().resolve()),
            "pipeline_vision_s": args.pipeline_vision_s,
            "pipeline_e2e_s": args.pipeline_e2e_s,
            "device_occupancy": args.device_occupancy,
        },
        "corpus_regression_self_check": corpus_payload["self_check"],
        "corpora": {
            name: _corpus_summary(items) for name, items in corpora.items()
        },
        "cache_preflight": cache_preflight,
        "setup_s": setup_s,
        "runtime_metadata": compiled_runtime.metadata,
        "environment": _environment(device),
        "like_for_like": validations,
        "eager_scout": eager_scout,
        "eager_to_compiled_anchors": {
            "anchors": anchors,
            "mean_compiled_over_eager": extrapolation_ratio,
            "assumption": (
                "The 704/1920 mean ratio is held constant when extrapolating "
                "uncompiled target lengths."
            ),
        },
        "packing_policy_analysis": policy_analysis,
        "exact_existing_graph_variant_runs": exact_variant_compiled,
        "headline_table": headline_rows,
        "phase1_candidates": sorted(
            phase1_candidates, key=lambda item: item["estimated_vision_s"]
        ),
        "new_graphs_compiled": 0,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output={output}")
    print(f"status={payload['status']} new_graphs_compiled=0")
    for name, validation in validations.items():
        suffix = (
            f" failures={validation['failures']}/{validation['validated_crops']}"
            if "failures" in validation
            else ""
        )
        print(f"like_for_like {name}: {validation['verdict']}{suffix}")
    print("corpus | strategy_length | new_graphs | vision_s | ratio | verdict | peak_mem")
    for row in headline_rows:
        print(
            " | ".join(
                str(row[key])
                for key in (
                    "corpus",
                    "strategy_length",
                    "new_graphs",
                    "projected_corpus_vision_s",
                    "ratio_vs_10.7_pipeline",
                    "like_for_like_verdict",
                    "peak_allocated_bytes_delta",
                )
            )
        )
    if phase1_candidates:
        print("phase1_candidates=" + json.dumps(payload["phase1_candidates"], indent=2))


if __name__ == "__main__":
    main()
