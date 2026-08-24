#!/usr/bin/env python3
"""Prove one vision GE executor survives graph-definition compaction."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

os.environ.setdefault("TE_PARALLEL_COMPILER", "1")
os.environ.setdefault("CANN_KNOWLEDGE_BANK_PROCESS_NUM", "0")
os.environ.setdefault("UNIREC_PURGE_HOST_AFTER_WARMUP", "1")

import torch
import torch_npu

from host_memory_diagnostics import process_snapshot
from modeling_optimized_unirec import OptimizedUniRecRunner
from post_warmup_host_cleanup import cleanup_after_warmup
from tbe_compiler_lifecycle import deinitialize_after_warmup
from torchair_ge_graph_compaction import compact_loaded_ge_graphs
from vision_bucket_presets import resolve_vision_bucket_specs
from vision_full_batch import BucketedFullVisionRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--bucket-preset", default="310p_k20_l4")
    parser.add_argument("--bucket", default="960x64_b4")
    parser.add_argument("--device", default="npu:0")
    return parser.parse_args()


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
    runner = OptimizedUniRecRunner(
        model_path=args.model_path,
        device=args.device,
        dtype="float16",
        compile_cache_dir=args.cache_dir,
    )
    runtime = BucketedFullVisionRuntime(
        runner,
        specs=resolve_vision_bucket_specs(args.bucket_preset),
        focal_depthwise_rewrite="constant_grouped_all",
        weight_format="torchair_internal",
        preset_name=args.bucket_preset,
    )
    specs = {spec.key: spec for spec in runtime.specs}
    if args.bucket not in specs:
        raise ValueError(f"bucket {args.bucket!r} is not in {args.bucket_preset}")
    spec = specs[args.bucket]
    device = torch.device(args.device)
    pixels = torch.full(
        (spec.batch_size, 3, spec.height, spec.width),
        0.25,
        dtype=runner.dtype,
        device=device,
    )
    masks = tuple(
        torch.ones(
            (
                spec.batch_size,
                1,
                spec.height // factor,
                spec.width // factor,
            ),
            dtype=runner.dtype,
            device=device,
        )
        for factor in (2, 4, 8, 16, 32)
    )
    stream = torch.npu.Stream()
    with torch.inference_mode(), torch.npu.stream(stream):
        started = time.perf_counter()
        reference = runtime.compiled[args.bucket](pixels, *masks)
    stream.synchronize()
    first_call_s = time.perf_counter() - started
    reference_cpu = reference.cpu()
    del reference

    tbe_report = deinitialize_after_warmup("ge_compaction_probe_warm")
    before = process_snapshot()
    compaction = compact_loaded_ge_graphs([runtime.compiled[args.bucket]])
    collected = gc.collect()
    cleanup = cleanup_after_warmup("ge_compaction_probe_compacted")
    after = process_snapshot()

    with torch.inference_mode(), torch.npu.stream(stream):
        started = time.perf_counter()
        candidate = runtime.compiled[args.bucket](pixels, *masks)
    stream.synchronize()
    replay_s = time.perf_counter() - started
    candidate_cpu = candidate.cpu()
    diff = (candidate_cpu.float() - reference_cpu.float()).abs()

    from te_fusion import parallel_compilation

    report = {
        "status": "pass" if torch.equal(reference_cpu, candidate_cpu) else "fail",
        "chip": torch_npu.npu.get_device_name(0),
        "bucket": args.bucket,
        "first_call_s": first_call_s,
        "replay_s": replay_s,
        "exact": torch.equal(reference_cpu, candidate_cpu),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "compiler_respawned": parallel_compilation.OpCompiler.compiler is not None,
        "gc_collected": int(collected),
        "compaction": compaction,
        "cleanup": cleanup,
        "before": before,
        "after": after,
        "pss_reclaimed_bytes": max(
            0,
            int(before["proc_bytes"]["pss"])
            - int(after["proc_bytes"]["pss"]),
        ),
        "tbe_deinit": tbe_report,
    }
    print("UNIREC_GE_GRAPH_COMPACTION " + json.dumps(report, sort_keys=True))
    if report["status"] != "pass" or report["compiler_respawned"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
