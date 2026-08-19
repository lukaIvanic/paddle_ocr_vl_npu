#!/usr/bin/env python3
"""Run a fast synthetic-state smoke of the production dual decode scheduler."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--a-items", type=int, default=130)
    parser.add_argument("--b-items", type=int, default=8)
    parser.add_argument("--warmup-passes", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["UNIREC_STATIC_CACHE_LEN"] = "2048"
    os.environ["UNIREC_STATIC_CROSS_CACHE_LEN"] = "1320"
    from continuous_unirec import (
        ContinuousReadyItem,
        ContinuousWorkerPrefilledItem,
    )
    from dual_lane_decode_policy import DecodeLaneSpec
    from dual_lane_unirec import (
        DualLaneContinuousUniRecDecoder,
        RankedReadyItem,
    )
    from modeling_optimized_unirec import OptimizedUniRecRunner

    runner = OptimizedUniRecRunner(
        model_path=args.model,
        device=args.device,
        dtype="float16",
        compile_cache_dir=args.cache_dir,
    )
    processor_shape = tuple(int(value) for value in runner.processor.max_side)
    runner._static_cross_cache_len_by_processor_max_side[processor_shape] = 1320
    layers = int(runner.config.decoder_layers)
    heads = int(runner.config.decoder_attention_heads)
    head_dim = int(runner.config.d_model) // heads
    a_cross = np.zeros(
        (2 * layers, 1, heads, 32, head_dim),
        dtype=np.float16,
    )
    b_cross = np.zeros(
        (2 * layers, 1, heads, 385, head_dim),
        dtype=np.float16,
    )
    items = []
    rank = 0
    for lane_name, count, packed in (
        ("a", args.a_items, a_cross),
        ("b", args.b_items, b_cross),
    ):
        for index in range(count):
            request_id = f"synthetic_{lane_name}_{index:04d}"
            prefilled = ContinuousWorkerPrefilledItem(
                packed_cross_kv=packed,
                prep={
                    "image": request_id,
                    "prepare_total_s": 0.0,
                    "processed_image_size": [1, 1],
                    "encoder_seq_len_hint": int(packed.shape[-2]),
                },
                prefill_s=0.0,
                actual_cross_attention_length=int(packed.shape[-2]),
                text_prefill_real_source_tokens=int(packed.shape[-2]),
                text_prefill_physical_source_tokens=int(packed.shape[-2]),
            )
            ready = ContinuousReadyItem(
                request_id=request_id,
                payload=request_id,
                prefilled=prefilled,
            )
            items.append(RankedReadyItem(ready=ready, global_rank=rank))
            rank += 1

    completed = []
    summary = DualLaneContinuousUniRecDecoder(
        runner=runner,
        a_spec=DecodeLaneSpec("a", 128, 1408, 384, 4),
        b_spec=DecodeLaneSpec("b", 128, 2048, 1320, 8),
        quantum_steps=2,
        max_skipped_quanta=2,
        overflow_policy="restart_b",
    ).run(
        items,
        on_complete=completed.append,
        graph_warmup_passes=args.warmup_passes,
    )
    expected = args.a_items + args.b_items
    if len(completed) != expected:
        raise RuntimeError(f"completion mismatch: {len(completed)} != {expected}")
    if len({item.request_id for item in completed}) != expected:
        raise RuntimeError("dual smoke produced duplicate completions")
    if summary["routed_a"] != args.a_items:
        raise RuntimeError("lane-A routing mismatch")
    if summary["routed_b"] != args.b_items:
        raise RuntimeError("lane-B routing mismatch")
    if summary["scheduler_quanta"] < 2 or summary["graph_switches"] < 1:
        raise RuntimeError("dual smoke did not switch between both graphs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "UNIREC_DUAL_DECODE_SMOKE: PASS "
        f"completed={len(completed)} routed_a={summary['routed_a']} "
        f"routed_b={summary['routed_b']} "
        f"promoted={summary['promoted_a_to_b']} "
        f"quanta={summary['scheduler_quanta']} "
        f"switches={summary['graph_switches']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
