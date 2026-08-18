#!/usr/bin/env python3
"""Summarize the minimal UniRec prefill-versus-decode factorization gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def infer_vision_preset(summary: dict[str, Any]) -> str:
    expected = {
        frozenset(
            {
                "960x64_b16",
                "512x256_b16",
                "960x256_b4",
                "512x512_b8",
                "960x512_b4",
            }
        ): "production_v1",
        frozenset(
            {
                "448x192_b2",
                "448x384_b2",
                "512x64_b4",
                "960x64_b4",
                "960x128_b2",
                "960x256_b1",
                "960x448_b1",
                "960x576_b1",
                "960x896_b1",
                "960x1408_b1",
            }
        ): "310p_k10_l4_all",
    }
    worker_graph_sets = {
        frozenset(
            worker["prefix_graph_warmup"]["graphs"]
        )
        for worker in summary["worker_setup_diagnostics"]
    }
    if len(worker_graph_sets) != 1:
        raise ValueError(f"worker graph inventories differ: {worker_graph_sets}")
    graph_set = next(iter(worker_graph_sets))
    try:
        return expected[graph_set]
    except KeyError as error:
        raise ValueError(f"unknown vision graph inventory: {sorted(graph_set)}") from error


def config(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "pages": int(summary["artifact"]["page_count"]),
        "crops": int(summary["artifact"]["crop_count"]),
        "workers": int(summary["workers"]),
        "layout_batch_size": int(summary["layout_batch_size"]),
        "vision_preset": infer_vision_preset(summary),
        "vision_weight_format": str(summary["vision_weight_format"]),
        "vision_depthwise": str(summary["vision_focal_depthwise_rewrite"]),
        "wall_s": float(summary["producer_wall_s"]),
        "pages_per_s": float(summary["throughput"]["pages_per_s"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-summary", type=Path, required=True)
    parser.add_argument("--intermediate-summary", type=Path, required=True)
    parser.add_argument("--optimized-summary", type=Path, required=True)
    parser.add_argument("--optimized-mismatch-report", type=Path, required=True)
    parser.add_argument("--intermediate-replay", type=Path, required=True)
    parser.add_argument("--intermediate-parity", type=Path, required=True)
    parser.add_argument("--intermediate-cross-kv", type=Path, required=True)
    parser.add_argument("--optimized-cross-kv", type=Path, required=True)
    parser.add_argument("--allow-no-optimized-mismatch", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    canonical = config(load(args.canonical_summary))
    intermediate = config(load(args.intermediate_summary))
    optimized = config(load(args.optimized_summary))
    mismatch = load(args.optimized_mismatch_report)
    replay = load(args.intermediate_replay)
    parity = load(args.intermediate_parity)
    intermediate_kv = load(args.intermediate_cross_kv)
    optimized_kv = load(args.optimized_cross_kv)

    for lane in (canonical, intermediate, optimized):
        if lane["pages"] != 128 or lane["crops"] != canonical["crops"]:
            raise ValueError(f"workload mismatch: {lane} versus {canonical}")
        if lane["workers"] != 4 or lane["layout_batch_size"] != 2:
            raise ValueError(f"unexpected worker/layout contract: {lane}")
    if canonical["vision_weight_format"] != "native":
        raise ValueError("canonical lane is not native vision weights")
    if canonical["vision_depthwise"] != "native":
        raise ValueError("canonical lane is not native depthwise")
    for lane in (intermediate, optimized):
        if lane["vision_weight_format"] != "torchair_internal":
            raise ValueError(f"lane is not torchair_internal: {lane}")
        if lane["vision_depthwise"] != "constant_grouped_all":
            raise ValueError(f"lane is not constant_grouped_all: {lane}")
    if intermediate["vision_preset"] != "production_v1":
        raise ValueError(f"intermediate preset mismatch: {intermediate}")
    if optimized["vision_preset"] != "310p_k10_l4_all":
        raise ValueError(f"optimized preset mismatch: {optimized}")

    optimized_mismatch_count = int(mismatch["compared_count"]) - int(
        mismatch["token_exact_count"]
    )
    compared = int(parity["compared_count"])
    intermediate_mismatch_count = compared - int(parity["token_exact_count"])
    if optimized_mismatch_count < 1 and not args.allow_no_optimized_mismatch:
        raise ValueError("optimized prior report has no mismatch to factor")
    if optimized_mismatch_count < 1 and not intermediate_mismatch_count:
        verdict = "NO_MISMATCH_REPRODUCED_ON_VALIDATION_CHIP"
        next_step = (
            "The complete harness passed, but this chip did not reproduce the "
            "310P-only mismatch; use the 310P result for causal attribution."
        )
    elif intermediate_mismatch_count:
        verdict = "VISION_WEIGHT_OR_DEPTHWISE_PATH_IMPLICATED"
        next_step = (
            "Keep production_v1 and split torchair_internal from "
            "constant_grouped_all in two cached five-bucket lanes."
        )
    else:
        verdict = "K10_BUCKET_PADDING_OR_MASK_PATH_IMPLICATED"
        next_step = (
            "Keep torchair_internal plus constant_grouped_all and compare "
            "production_v1 against K10 with native-equivalent bucket masks."
        )

    full_pages = 1651.0
    reported_sequential_pages_per_s = 1.908
    sequential_core_s = full_pages / reported_sequential_pages_per_s
    prefill_s = 535.0
    decode_graph_s = 290.0
    decode_phase_s = sequential_core_s - prefill_s
    report = {
        "schema": "unirec_prefill_decode_factorization_v1",
        "status": "ok",
        "verdict": verdict,
        "next_step": next_step,
        "lanes": {
            "canonical_native": canonical,
            "production_buckets_optimized_weights": intermediate,
            "k10_optimized": optimized,
        },
        "decode_probe": {
            "selected_crops": int(replay["workload"]["selected_crops"]),
            "decode_iterations": int(replay["decode"]["decode_iterations"]),
            "decode_graph_s": float(replay["decode"]["decode_s"]),
            "raw_tok_s": float(replay["decode"]["raw_decode_tokens_per_s"]),
            "effective_tok_s": float(
                replay["decode"]["effective_decode_tokens_per_s"]
            ),
            "intermediate_token_mismatches": intermediate_mismatch_count,
            "optimized_prior_token_mismatches": optimized_mismatch_count,
            "intermediate_long_outputs": parity["long_output_counts"],
            "optimized_prior_long_outputs": mismatch["long_output_counts"],
        },
        "cross_kv": {
            "intermediate_vs_canonical": {
                key: intermediate_kv[key]
                for key in (
                    "compared_rows",
                    "exact_rows",
                    "weighted_mean_abs",
                    "weighted_rmse",
                    "max_abs",
                )
            },
            "optimized_vs_canonical": {
                key: optimized_kv[key]
                for key in (
                    "compared_rows",
                    "exact_rows",
                    "weighted_mean_abs",
                    "weighted_rmse",
                    "max_abs",
                )
            },
        },
        "full_baseline_budget": {
            "pages": int(full_pages),
            "reported_sequential_pages_per_s": reported_sequential_pages_per_s,
            "implied_sequential_core_s": sequential_core_s,
            "prefill_s": prefill_s,
            "prefill_share_of_sequential_core": prefill_s / sequential_core_s,
            "prefill_pages_per_s": full_pages / prefill_s,
            "decode_phase_s_implied_by_sequential_core": decode_phase_s,
            "decode_phase_share_of_sequential_core": (
                decode_phase_s / sequential_core_s
            ),
            "decode_phase_pages_per_s": full_pages / decode_phase_s,
            "decode_graph_s_reported": decode_graph_s,
            "decode_graph_share_of_sequential_core": (
                decode_graph_s / sequential_core_s
            ),
            "decode_graph_pages_per_s": full_pages / decode_graph_s,
            "decode_non_graph_s_implied": decode_phase_s - decode_graph_s,
            "timing_note": (
                "The reported 290 s decode value cannot be the whole decode "
                "phase because 535+290 s implies 2.00 pages/s, not the "
                "reported 1.908. It is retained as decode-graph work; the "
                "whole decode phase is inferred from sequential core."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "UNIREC_PREFILL_DECODE_FACTORIZATION: PASS "
        f"verdict={verdict} intermediate_mismatches={intermediate_mismatch_count} "
        f"optimized_prior_mismatches={optimized_mismatch_count} "
        f"intermediate_prefill_s={intermediate['wall_s']:.3f} "
        f"intermediate_prefill_pages_s={intermediate['pages_per_s']:.3f} "
        f"decode_raw_tok_s={report['decode_probe']['raw_tok_s']:.1f} "
        f"decode_effective_tok_s={report['decode_probe']['effective_tok_s']:.1f} "
        f"output={args.output.resolve()}",
        flush=True,
    )
    print("UNIREC_PREFILL_DECODE_FACTORIZATION_REPORT")
    print(json.dumps(report, separators=(",", ":")))


if __name__ == "__main__":
    main()
