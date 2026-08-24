#!/usr/bin/env python3
"""Measure one process owning layout, UniRec prefill graphs, and decode."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TE_PARALLEL_COMPILER", "1")
os.environ.setdefault("CANN_KNOWLEDGE_BANK_PROCESS_NUM", "0")
os.environ.setdefault("UNIREC_DEINIT_TBE_AFTER_WARMUP", "1")
os.environ.setdefault("UNIREC_STATIC_CACHE_LEN", "2048")
os.environ.setdefault("UNIREC_STATIC_CROSS_CACHE_LEN", "1320")

import numpy as np
import torch
import torch_npu

from benchmark_shared_vision_streams import (
    build_inputs,
    run_concurrent,
)
from decode_model_optimizations import (
    apply_decode_model_optimizations,
    decode_cache_variant_root,
)
from host_memory_diagnostics import process_snapshot
from modeling_optimized_unirec import OptimizedUniRecRunner
from opendoc_layout_npu import PPDocLayoutV2NpuAdapter
from run_opendoc_batched_unirec import warmup_configured_graphs
from tbe_compiler_lifecycle import deinitialize_after_warmup
from vision_bucket_presets import resolve_vision_bucket_specs
from vision_full_batch import BucketedFullVisionRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--layout-model", type=Path, required=True)
    parser.add_argument("--vision-cache", type=Path, required=True)
    parser.add_argument("--layout-cache", type=Path, required=True)
    parser.add_argument("--decode-cache-parent", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--bucket-preset", default="310p_k20_l4")
    parser.add_argument("--reset-dynamo-after-warmup", action="store_true")
    return parser.parse_args()


def record(stages: dict[str, object], label: str) -> None:
    stages[label] = process_snapshot()
    print(
        "UNIREC_UNIFIED_OWNER_MEMORY "
        + json.dumps({"label": label, **stages[label]}, sort_keys=True),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    visible = {
        int(value)
        for value in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    }
    if visible.intersection({5, 6}):
        raise RuntimeError("physical NPU 5 and NPU 6 are excluded")
    torch_npu.npu.set_compile_mode(jit_compile=False)
    stages: dict[str, object] = {}
    record(stages, "enter")
    runner = OptimizedUniRecRunner(
        model_path=args.model_path,
        device=args.device,
        dtype="float16",
        compile_cache_dir=args.vision_cache,
    )
    record(stages, "after_unirec")
    decode_optimizations = apply_decode_model_optimizations(
        runner,
        weight_format="nz",
        lm_head_rows=57344,
    )
    record(stages, "after_decode_weights")

    layout = PPDocLayoutV2NpuAdapter(
        model_path=args.layout_model,
        device=args.device,
        dtype="float32",
        reading_order_dtype="float32",
        threshold=0.5,
        execution="torchair",
        compile_cache_dir=args.layout_cache,
        batch_size=2,
        weight_format="native",
        depthwise_rewrite="native",
        input_color_order="rgb",
    )
    layout_stream = torch.npu.Stream()
    layout_inputs = [np.zeros((800, 800, 3), dtype=np.uint8) for _ in range(2)]
    with torch.npu.stream(layout_stream):
        layout(layout_inputs, threshold=0.5)
    layout_stream.synchronize()
    record(stages, "after_layout_warmup")

    vision = BucketedFullVisionRuntime(
        runner,
        specs=resolve_vision_bucket_specs(args.bucket_preset),
        focal_depthwise_rewrite="constant_grouped_all",
        weight_format="torchair_internal",
        preset_name=args.bucket_preset,
    )
    vision_inputs = build_inputs(vision, lanes=args.lanes)
    vision_streams = [torch.npu.Stream() for _ in range(args.lanes)]
    keys_by_lane = [[] for _ in range(args.lanes)]
    lane_pixels = [0 for _ in range(args.lanes)]
    for spec in sorted(
        vision.specs,
        key=lambda row: row.batch_size * row.width * row.height,
        reverse=True,
    ):
        lane = min(range(args.lanes), key=lane_pixels.__getitem__)
        keys_by_lane[lane].append(spec.key)
        lane_pixels[lane] += spec.batch_size * spec.width * spec.height
    run_concurrent(vision, keys_by_lane, vision_inputs, vision_streams)
    record(stages, "after_vision_warmup")

    text_runtime = runner._get_compiled_packed_text_prefill_runtime()
    text_input = torch.zeros(
        (1, text_runtime.bucket, runner.config.d_model),
        dtype=runner.dtype,
        device=torch.device(args.device),
    )
    text_output = text_runtime.compiled(text_input)
    torch.npu.synchronize()
    text_runtime._first_call = False
    del text_output, text_input
    record(stages, "after_text_prefill_warmup")

    runner.compile_cache_dir = decode_cache_variant_root(
        args.decode_cache_parent,
        weight_format="nz",
        lm_head_rows=57344,
    )
    decode_args = SimpleNamespace(
        text_prefill_mode="eager",
        decode_mode="compiled_ifa",
        compile_backend="torchair",
        decode_batch_size=128,
    )
    decode_report = warmup_configured_graphs(
        args=decode_args,
        runner=runner,
        vision_atlas_runtime=None,
        passes=2,
        warmup_decode=True,
    )
    record(stages, "after_decode_warmup")

    deinit_report = deinitialize_after_warmup("unified_owner_warmup_complete")
    dynamo_reset_report = None
    if args.reset_dynamo_after_warmup:
        record(stages, "before_dynamo_reset")
        torch._dynamo.reset()
        collected = gc.collect()
        record(stages, "after_dynamo_reset")

        # Replay every resident graph on the stream used for its first call.
        run_concurrent(vision, keys_by_lane, vision_inputs, vision_streams)
        with torch.npu.stream(layout_stream):
            layout(layout_inputs, threshold=0.5)
        layout_stream.synchronize()
        replay_text_input = torch.zeros(
            (1, text_runtime.bucket, runner.config.d_model),
            dtype=runner.dtype,
            device=torch.device(args.device),
        )
        replay_text_output = text_runtime.compiled(replay_text_input)
        torch.npu.synchronize()
        del replay_text_output, replay_text_input
        replay_decode_report = warmup_configured_graphs(
            args=decode_args,
            runner=runner,
            vision_atlas_runtime=None,
            passes=1,
            warmup_decode=True,
        )
        from te_fusion import parallel_compilation

        dynamo_reset_report = {
            "gc_collected": int(collected),
            "replay_decode": replay_decode_report,
            "compiler_respawned": (
                parallel_compilation.OpCompiler.compiler is not None
            ),
        }
        record(stages, "after_dynamo_reset_replay")
    del vision_inputs, layout_inputs
    torch.npu.empty_cache()
    record(stages, "final")
    report = {
        "status": "pass",
        "chip": torch_npu.npu.get_device_name(0),
        "lanes": args.lanes,
        "keys_by_lane": keys_by_lane,
        "decode_optimizations": decode_optimizations,
        "decode_warmup": decode_report,
        "tbe_deinit": deinit_report,
        "dynamo_reset": dynamo_reset_report,
        "stages": stages,
        "layout_batch_size": 2,
        "vision_bucket_preset": args.bucket_preset,
        "decode_batch_size": 128,
    }
    print("UNIREC_UNIFIED_OWNER_SUMMARY " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
