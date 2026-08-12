#!/usr/bin/env python3
"""Alternate stock B1 and pseudo-B2 IncreFA in one full-decoder process."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(HERE.parent))

from text_decode_lab import TextDecodeLab, parse_args as parse_lab_args
from utils.timing import DeviceTimeline, synchronize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-position", type=int, default=1249)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument(
        "--schedule",
        choices=("weight", "kv_then_weight"),
        default="weight",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.schedule == "weight":
        control_name = "combined_apply_prefetch_rope_lut"
        candidate_name = "combined_apply_prefetch_rope_lut_pseudo_b2"
    else:
        control_name = "combined_apply_kv_then_mlp_prefetch_rope_lut"
        candidate_name = (
            "combined_apply_kv_then_mlp_prefetch_rope_lut_pseudo_b2"
        )

    common = [
        "--mode",
        "profile",
        "--batch-size",
        "1",
        "--cache-length",
        "4096",
        "--profile-position",
        str(args.profile_position),
        "--warmup",
        "0",
        "--repeats",
        "1",
        "--synthetic-lm-head-size",
        "16384",
        "--allow-compile",
    ]
    control = TextDecodeLab(
        parse_lab_args([*common, "--decode-optimization", control_name])
    )
    candidate = TextDecodeLab(
        parse_lab_args([*common, "--decode-optimization", candidate_name])
    )
    if control.device != candidate.device:
        raise AssertionError("the two runtimes must share one physical NPU")

    control_arena = control._dummy_arena(
        active_slots=1,
        cache_position=args.profile_position,
    )
    candidate_arena = candidate._dummy_arena(
        active_slots=1,
        cache_position=args.profile_position,
    )
    for iteration in range(args.warmup):
        control_arena.step(control.runtime.fn, iteration=iteration)
        candidate_arena.step(candidate.runtime.fn, iteration=iteration)
    synchronize(control.device)

    timeline = DeviceTimeline(control.device)
    lane_names: list[str] = []
    for round_index in range(args.rounds):
        order = (
            (("control", control), ("candidate", candidate))
            if round_index % 2 == 0
            else (("candidate", candidate), ("control", control))
        )
        for lane_name, lab in order:
            arena = control_arena if lane_name == "control" else candidate_arena
            label = f"round_{round_index:04d}_{lane_name}"
            lane_names.append(label)
            timeline.measure(
                label,
                lambda arena=arena, lab=lab, round_index=round_index: arena.step(
                    lab.runtime.fn,
                    iteration=args.warmup + round_index,
                ),
            )
    spans = timeline.resolve()
    per_lane: dict[str, list[float]] = {"control": [], "candidate": []}
    for label in lane_names:
        lane = "candidate" if label.endswith("candidate") else "control"
        per_lane[lane].append(float(spans[label]) * 1000.0)

    rows = {}
    for lane, values in per_lane.items():
        rows[lane] = {
            "optimization": (
                control_name if lane == "control" else candidate_name
            ),
            "samples": len(values),
            "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "tok_per_s": 1000.0 / statistics.mean(values),
        }
    rows["candidate"]["latency_change_percent"] = (
        (rows["candidate"]["mean_ms"] / rows["control"]["mean_ms"]) - 1.0
    ) * 100.0
    rows["candidate"]["throughput_change_percent"] = (
        (rows["candidate"]["tok_per_s"] / rows["control"]["tok_per_s"])
        - 1.0
    ) * 100.0

    payload = {
        "chip": "Ascend 910B",
        "contract": "alternating same-process full 18-layer decoder",
        "shape": {
            "physical_batch": 1,
            "cache_length": 4096,
            "profile_position": args.profile_position,
            "synthetic_lm_head_size": 16384,
        },
        "warmup_pairs": args.warmup,
        "measured_pairs": args.rounds,
        "schedule": args.schedule,
        "results": rows,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
