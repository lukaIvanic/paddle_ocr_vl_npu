#!/usr/bin/env python3
"""Measure four cached layout executors sharing one FP32 model."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import threading
import time

os.environ.setdefault("TE_PARALLEL_COMPILER", "1")
os.environ.setdefault("CANN_KNOWLEDGE_BANK_PROCESS_NUM", "0")
os.environ.setdefault("UNIREC_DEINIT_TBE_AFTER_WARMUP", "1")

import torch
import torch_npu

from host_memory_diagnostics import process_snapshot
from opendoc_layout_npu import PPDocLayoutV2NpuAdapter
from post_warmup_host_cleanup import cleanup_after_warmup
from tbe_compiler_lifecycle import deinitialize_after_warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--freeze-parameters", action="store_true")
    parser.add_argument("--drop-source-model", action="store_true")
    return parser.parse_args()


def run_concurrent(
    executors: list[object],
    inputs: list[torch.Tensor],
    streams: list[object],
    *,
    repeats: int,
) -> tuple[float, list[tuple[torch.Tensor, ...]]]:
    barrier = threading.Barrier(len(executors) + 1)

    def lane(index: int) -> tuple[torch.Tensor, ...]:
        torch.npu.set_device(inputs[index].device)
        with torch.inference_mode(), torch.npu.stream(streams[index]):
            barrier.wait()
            output = None
            for _ in range(repeats):
                output = executors[index](inputs[index])
        streams[index].synchronize()
        if output is None:
            raise RuntimeError("layout executor produced no output")
        return tuple(value.cpu() for value in output)

    with ThreadPoolExecutor(max_workers=len(executors)) as pool:
        futures = [pool.submit(lane, index) for index in range(len(executors))]
        barrier.wait()
        started = time.perf_counter()
        outputs = [future.result() for future in futures]
    return time.perf_counter() - started, outputs


def main() -> None:
    args = parse_args()
    if args.lanes < 1 or args.repeats < 1:
        raise ValueError("lanes and repeats must be positive")
    visible = {
        int(value)
        for value in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    }
    if visible.intersection({5, 6}):
        raise RuntimeError("physical NPU 5 and NPU 6 are excluded")
    torch_npu.npu.set_compile_mode(jit_compile=False)
    adapter = PPDocLayoutV2NpuAdapter(
        model_path=args.model_path,
        device=args.device,
        dtype="float32",
        reading_order_dtype="float32",
        threshold=0.5,
        execution="torchair",
        compile_cache_dir=args.cache_dir,
        batch_size=2,
        weight_format="native",
        freeze_parameters=args.freeze_parameters,
        depthwise_rewrite="native",
        input_color_order="rgb",
    )
    runtime = adapter.compiled_runtime
    if runtime is None:
        raise RuntimeError("layout adapter did not create a compiled runtime")
    device = torch.device(args.device)
    streams = [torch.npu.Stream() for _ in range(args.lanes)]
    inputs = [
        torch.full(
            (2, 3, 800, 800),
            float(index + 1) / float(args.lanes + 1),
            dtype=torch.float32,
            device=device,
        )
        for index in range(args.lanes)
    ]
    with torch.inference_mode(), torch.npu.stream(streams[0]):
        primary_reference = runtime.compiled(inputs[0])
    streams[0].synchronize()
    del primary_reference
    after_primary = process_snapshot()

    cache_files = sorted(runtime.cache_dir.rglob("compiled_module"))
    if len(cache_files) != 1:
        raise RuntimeError(
            f"expected one layout compiled_module, found {cache_files}"
        )
    try:
        from torch_npu.dynamo.torchair.inference._cache_compiler import (
            CompiledModel,
        )
    except ImportError:
        from torchair.inference._cache_compiler import CompiledModel

    executors = [runtime.compiled]
    for _ in range(1, args.lanes):
        compiled_model = CompiledModel.load(str(cache_files[0]))
        executors.append(
            compiled_model.rebase(
                runtime.stage,
                global_vars=globals(),
                func=runtime.stage.forward,
                cache_dir=str(cache_files[0].parent),
            )
        )
    for index in range(1, args.lanes):
        with torch.inference_mode(), torch.npu.stream(streams[index]):
            executors[index](inputs[index])
        streams[index].synchronize()
    after_all = process_snapshot()
    deinit_report = deinitialize_after_warmup("shared_layout_executors_warm")

    before_drop_outputs = []
    for index, executor in enumerate(executors):
        with torch.inference_mode(), torch.npu.stream(streams[index]):
            output = executor(inputs[index])
        streams[index].synchronize()
        before_drop_outputs.append(tuple(value.cpu() for value in output))

    source_model_release = None
    source_model_parity = None
    if args.drop_source_model:
        if not args.freeze_parameters:
            raise ValueError("dropping the source model requires frozen parameters")
        before_drop = process_snapshot()
        adapter.processor = None
        adapter.model = None
        runtime.stage.model = None
        gc.collect()
        torch.npu.empty_cache()
        cleanup = cleanup_after_warmup("shared_layout_source_model_drop")
        after_drop = process_snapshot()
        after_drop_outputs = []
        for index, executor in enumerate(executors):
            with torch.inference_mode(), torch.npu.stream(streams[index]):
                output = executor(inputs[index])
            streams[index].synchronize()
            after_drop_outputs.append(tuple(value.cpu() for value in output))
        fields = []
        for reference, candidate in zip(before_drop_outputs, after_drop_outputs):
            lane_fields = []
            for left, right in zip(reference, candidate):
                diff = (right.float() - left.float()).abs()
                lane_fields.append(
                    {
                        "exact": torch.equal(left, right),
                        "max_abs": float(diff.max()),
                        "mean_abs": float(diff.mean()),
                    }
                )
            fields.append(lane_fields)
        source_model_parity = fields
        source_model_release = {
            "before": before_drop,
            "after": after_drop,
            "pss_reclaimed_bytes": max(
                0,
                int(before_drop["proc_bytes"]["pss"])
                - int(after_drop["proc_bytes"]["pss"]),
            ),
            "cleanup": cleanup,
        }

    references = []
    serial_started = time.perf_counter()
    for index in range(args.lanes):
        output = None
        with torch.inference_mode(), torch.npu.stream(streams[index]):
            for _ in range(args.repeats):
                output = executors[index](inputs[index])
        streams[index].synchronize()
        if output is None:
            raise RuntimeError("layout executor produced no serial output")
        references.append(tuple(value.cpu() for value in output))
    serial_s = time.perf_counter() - serial_started
    concurrent_s, candidates = run_concurrent(
        executors,
        inputs,
        streams,
        repeats=args.repeats,
    )
    parity = []
    for reference, candidate in zip(references, candidates):
        fields = []
        for left, right in zip(reference, candidate):
            diff = (right.float() - left.float()).abs()
            fields.append(
                {
                    "exact": torch.equal(left, right),
                    "max_abs": float(diff.max()),
                    "mean_abs": float(diff.mean()),
                }
            )
        parity.append(fields)

    from te_fusion import parallel_compilation

    exact = all(field["exact"] for lane in parity for field in lane)
    report = {
        "status": "pass" if exact else "fail",
        "chip": torch_npu.npu.get_device_name(0),
        "lanes": args.lanes,
        "repeats": args.repeats,
        "serial_s": serial_s,
        "concurrent_s": concurrent_s,
        "speedup": serial_s / concurrent_s,
        "parity": parity,
        "after_primary": after_primary,
        "after_all": after_all,
        "additional_executor_pss_bytes": max(
            0,
            int(after_all["proc_bytes"]["pss"])
            - int(after_primary["proc_bytes"]["pss"]),
        ),
        "compiler_respawned": parallel_compilation.OpCompiler.compiler is not None,
        "tbe_deinit": deinit_report,
        "freeze_parameters": bool(args.freeze_parameters),
        "drop_source_model": bool(args.drop_source_model),
        "source_model_release": source_model_release,
        "source_model_parity": source_model_parity,
        "cache_file": str(cache_files[0]),
    }
    print("UNIREC_SHARED_LAYOUT_EXECUTORS " + json.dumps(report, sort_keys=True))
    drop_exact = (
        source_model_parity is None
        or all(field["exact"] for lane in source_model_parity for field in lane)
    )
    if report["status"] != "pass" or report["compiler_respawned"] or not drop_exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
