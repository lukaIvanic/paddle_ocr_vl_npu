#!/usr/bin/env python3
"""Replay one cached K20 vision graph through several shared-model executors."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable

os.environ.setdefault("TE_PARALLEL_COMPILER", "1")
os.environ.setdefault("CANN_KNOWLEDGE_BANK_PROCESS_NUM", "0")
os.environ.setdefault("UNIREC_DEINIT_TBE_AFTER_WARMUP", "1")

import torch
import torch_npu

from host_memory_diagnostics import process_snapshot
from modeling_optimized_unirec import OptimizedUniRecRunner
from post_warmup_host_cleanup import purge_host_allocator_pages
from tbe_compiler_lifecycle import deinitialize_after_warmup
from vision_bucket_presets import (
    assign_vision_bucket_cache_slots,
    resolve_vision_bucket_specs,
)
from vision_full_batch import (
    EXTENDED_FLAT_GLOBAL_CONTEXT_BUCKET_KEYS,
    FLAT_GLOBAL_CONTEXT_BUCKET_KEYS,
    BucketedFullVisionRuntime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--bucket-preset", default="310p_k20_l4")
    parser.add_argument("--bucket-key", default="960x64_b4")
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--total-calls", type=int, default=148)
    parser.add_argument(
        "--executor-mode",
        choices=("rebased", "shared"),
        default="rebased",
    )
    return parser.parse_args()


def compiled_module_inventory(cache_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(cache_dir)),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(cache_dir.rglob("compiled_module"))
    ]


def om_inventory(cache_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(cache_dir)),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(cache_dir.rglob("*.om"))
    ]


def compile_method_for_key(
    runtime: BucketedFullVisionRuntime,
    key: str,
) -> Callable[..., torch.Tensor]:
    specs = tuple(runtime.specs)
    slots = assign_vision_bucket_cache_slots(
        specs,
        slot_count=max(10, len(specs)),
    )
    slot = dict(zip((spec.key for spec in specs), slots))[key]
    module = runtime.modules[key]
    if key in (
        FLAT_GLOBAL_CONTEXT_BUCKET_KEYS
        | EXTENDED_FLAT_GLOBAL_CONTEXT_BUCKET_KEYS
    ):
        method_name = f"_forward_flat_bucket_slot_{slot}"
    else:
        method_name = f"_forward_bucket_slot_{slot}"
    return getattr(module, method_name)


def run_calls(
    executors: list[Callable[..., torch.Tensor]],
    inputs: list[tuple[torch.Tensor, ...]],
    streams: list[Any],
    call_counts: list[int],
) -> tuple[float, list[torch.Tensor]]:
    barrier = threading.Barrier(len(executors) + 1)

    def lane(index: int) -> torch.Tensor:
        torch_npu.npu.set_device(inputs[index][0].device)
        output = None
        with torch.inference_mode(), torch.npu.stream(streams[index]):
            barrier.wait()
            for _ in range(call_counts[index]):
                output = executors[index](*inputs[index])
        streams[index].synchronize()
        if output is None:
            raise RuntimeError(f"lane {index} had no calls")
        return output.cpu()

    with ThreadPoolExecutor(max_workers=len(executors)) as pool:
        futures = [pool.submit(lane, index) for index in range(len(executors))]
        barrier.wait()
        started = time.perf_counter()
        outputs = [future.result() for future in futures]
    return time.perf_counter() - started, outputs


def main() -> None:
    args = parse_args()
    if args.lanes < 1 or args.total_calls < args.lanes:
        raise ValueError("total calls must provide at least one call per lane")
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
        dtype=args.dtype,
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
    if args.bucket_key not in specs:
        raise ValueError(f"unknown bucket key {args.bucket_key!r}")
    spec = specs[args.bucket_key]
    cache_dir = runtime.cache_dirs[args.bucket_key]
    before = {
        "compiled_modules": compiled_module_inventory(cache_dir),
        "oms": om_inventory(cache_dir),
    }

    device = torch.device(args.device)
    streams = [torch.npu.Stream(device=device) for _ in range(args.lanes)]
    inputs = []
    for lane in range(args.lanes):
        pixels = torch.full(
            (spec.batch_size, 3, spec.height, spec.width),
            float(lane + 1) / float(args.lanes + 1),
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
        inputs.append((pixels, *masks))

    first_call_started = time.perf_counter()
    with torch.inference_mode(), torch.npu.stream(streams[0]):
        runtime.compiled[args.bucket_key](*inputs[0])
    streams[0].synchronize()
    first_call_s = time.perf_counter() - first_call_started
    after_primary = process_snapshot()

    compiled_modules = sorted(cache_dir.rglob("compiled_module"))
    if len(compiled_modules) != 1:
        raise RuntimeError(
            f"expected one compiled_module in {cache_dir}, found {compiled_modules}"
        )
    try:
        from torch_npu.dynamo.torchair.inference._cache_compiler import (
            CompiledModel,
        )
    except ImportError:
        from torchair.inference._cache_compiler import CompiledModel

    executors = [runtime.compiled[args.bucket_key]]
    namespaces: list[dict[str, Any]] = []
    if args.executor_mode == "shared":
        executors *= args.lanes
    else:
        method = compile_method_for_key(runtime, args.bucket_key)
        module = runtime.modules[args.bucket_key]
        for _ in range(1, args.lanes):
            namespace: dict[str, Any] = {}
            executors.append(
                CompiledModel.load(str(compiled_modules[0])).rebase(
                    module,
                    global_vars=namespace,
                    func=method,
                    cache_dir=str(compiled_modules[0].parent),
                )
            )
            namespaces.append(namespace)

    reference_outputs = []
    for lane in range(args.lanes):
        with torch.inference_mode(), torch.npu.stream(streams[0]):
            reference_outputs.append(executors[0](*inputs[lane]).cpu())
        streams[0].synchronize()

    warm_started = time.perf_counter()
    warm_s, candidate_outputs = run_calls(
        executors,
        inputs,
        streams,
        [1] * args.lanes,
    )
    if warm_s <= 0:
        raise RuntimeError("executor warmup produced a non-positive duration")
    warm_wall_s = time.perf_counter() - warm_started
    after_all = process_snapshot()
    deinit = deinitialize_after_warmup("shared_vision_same_graph_warm")

    parity = []
    for reference, candidate in zip(reference_outputs, candidate_outputs):
        diff = (candidate.float() - reference.float()).abs()
        parity.append(
            {
                "exact": torch.equal(reference, candidate),
                "max_abs": float(diff.max()),
                "mean_abs": float(diff.mean()),
            }
        )

    serial_s, _ = run_calls(
        [executors[0]],
        [inputs[0]],
        [streams[0]],
        [args.total_calls],
    )
    base = args.total_calls // args.lanes
    call_counts = [base] * args.lanes
    for lane in range(args.total_calls - base * args.lanes):
        call_counts[lane] += 1
    concurrent_s, _ = run_calls(
        executors,
        inputs,
        streams,
        call_counts,
    )
    del candidate_outputs
    del reference_outputs
    if args.executor_mode == "rebased":
        del executors[1:]
    for namespace in namespaces:
        namespace.clear()
    namespaces.clear()
    gc.collect()
    torch.npu.empty_cache()
    purge_host_allocator_pages()
    after_clone_release = process_snapshot()
    after = {
        "compiled_modules": compiled_module_inventory(cache_dir),
        "oms": om_inventory(cache_dir),
    }
    from te_fusion import parallel_compilation

    report = {
        "status": "pass"
        if all(value["exact"] for value in parity) and before == after
        else "fail",
        "chip": torch_npu.npu.get_device_name(0),
        "bucket_key": args.bucket_key,
        "executor_mode": args.executor_mode,
        "shape": [spec.batch_size, 3, spec.height, spec.width],
        "lanes": args.lanes,
        "total_calls": args.total_calls,
        "calls_per_lane": call_counts,
        "first_call_s": first_call_s,
        "rebase_warm_wall_s": warm_wall_s,
        "serial_s": serial_s,
        "concurrent_s": concurrent_s,
        "speedup": serial_s / concurrent_s,
        "parity": parity,
        "cache_inventory_unchanged": before == after,
        "cache_before": before,
        "cache_after": after,
        "after_primary": after_primary,
        "after_all": after_all,
        "after_clone_release": after_clone_release,
        "additional_executor_pss_bytes": max(
            0,
            int(after_all["proc_bytes"]["pss"])
            - int(after_primary["proc_bytes"]["pss"]),
        ),
        "unreclaimed_executor_pss_bytes": max(
            0,
            int(after_clone_release["proc_bytes"]["pss"])
            - int(after_primary["proc_bytes"]["pss"]),
        ),
        "compiler_respawned": parallel_compilation.OpCompiler.compiler is not None,
        "tbe_deinit": deinit,
    }
    print("UNIREC_SHARED_VISION_SAME_GRAPH " + json.dumps(report, sort_keys=True))
    if report["status"] != "pass" or report["compiler_respawned"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
