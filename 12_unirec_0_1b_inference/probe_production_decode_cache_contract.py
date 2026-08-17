#!/usr/bin/env python3
"""Gate a persisted UniRec decode graph against the production tensor contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any


RECOMPILE_WARNING = (
    "Skip cache as LocalUniRecCachedDecodeStepModule.forward recompiled"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--compile-cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--self-cache-length", type=int, default=2048)
    parser.add_argument("--cross-cache-length", type=int, default=1320)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.passes < 2:
        parser.error("batch size must be positive and passes must be >= 2")
    if args.self_cache_length < 1 or args.cross_cache_length < 1:
        parser.error("cache lengths must be positive")
    return args


def inventory(directory: Path) -> dict[str, Any]:
    compiled = sorted(directory.rglob("compiled_module")) if directory.exists() else []
    oms = sorted(directory.rglob("*.om")) if directory.exists() else []
    return {
        "directory": str(directory),
        "compiled_module_count": len(compiled),
        "om_count": len(oms),
        "compiled_module_bytes": sum(path.stat().st_size for path in compiled),
        "om_bytes": sum(path.stat().st_size for path in oms),
        "compiled_module_files": [str(path.relative_to(directory)) for path in compiled],
        "om_files": [str(path.relative_to(directory)) for path in oms],
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def main() -> int:
    args = parse_args()
    os.environ["UNIREC_STATIC_CACHE_LEN"] = str(args.self_cache_length)
    os.environ["UNIREC_STATIC_CROSS_CACHE_LEN"] = str(args.cross_cache_length)

    import torch
    import torch_npu

    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch._logging.set_logs(recompiles=True)

    from continuous_unirec import (
        ContinuousUniRecDecoder,
        production_decode_cache_parent,
    )
    from modeling_optimized_unirec import OptimizedUniRecRunner, synchronize_device

    setup_started = time.perf_counter()
    cache_parent = production_decode_cache_parent(args.compile_cache_dir)
    runner = OptimizedUniRecRunner(
        model_path=args.model_path.expanduser().resolve(),
        device=args.device,
        dtype="float16",
        compile_cache_dir=cache_parent,
    )
    processor_shape = tuple(int(value) for value in runner.processor.max_side)
    runner._static_cross_cache_len_by_processor_max_side[processor_shape] = (
        args.cross_cache_length
    )
    decoder = ContinuousUniRecDecoder(
        runner=runner,
        batch_size=args.batch_size,
        max_length=args.self_cache_length,
        decode_mode="compiled_ifa",
        compile_backend="torchair",
    )
    compiled, metadata = runner._compile_decode_module(
        backend="torchair",
        self_attention_backend="increfa_all",
        compile_dynamic=False,
        cross_cache_len=args.cross_cache_length,
        batch_size=args.batch_size,
    )
    cache_dir = Path(metadata["torchair_cache_dir"])
    before = inventory(cache_dir)
    arena = decoder._allocate_empty_arena()
    if arena.cross_attention_mask is None:
        raise RuntimeError("decode cache probe has no cross-attention mask")
    # Production always admits valid rows before its first decode call. Avoid
    # an artificial all-masked input, which can timeout the 310P attention
    # kernel even though that state is unreachable in the real scheduler.
    decoder_input_ids, cache_position = decoder._allocate_decode_device_inputs(
        args.batch_size,
        args.device,
    )
    with torch.inference_mode():
        arena.cross_attention_mask.zero_()
        decoder_input_ids.fill_(int(runner.config.decoder_start_token_id))
        cache_position.fill_(1)
    inputs = (
        decoder_input_ids,
        cache_position,
        0,
        arena.key_cache,
        arena.value_cache,
        arena.cross_key_cache,
        arena.cross_value_cache,
        arena.cross_attention_mask,
    )
    setup_s = time.perf_counter() - setup_started

    pass_wall_s: list[float] = []
    warning_rows: list[dict[str, str]] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for pass_index in range(args.passes):
            started = time.perf_counter()
            with torch.inference_mode():
                logits = compiled(*inputs)
            synchronize_device(args.device)
            elapsed = time.perf_counter() - started
            pass_wall_s.append(elapsed)
            del logits
            print(
                "UNIREC_DECODE_CACHE_CONTRACT_PASS "
                f"pass={pass_index + 1}/{args.passes} wall_s={elapsed:.6f}",
                flush=True,
            )
        warning_rows = [
            {
                "category": warning.category.__name__,
                "message": str(warning.message),
            }
            for warning in caught
        ]

    after = inventory(cache_dir)
    recompiled = any(
        RECOMPILE_WARNING in row["message"] for row in warning_rows
    )
    had_complete_cache = (
        before["compiled_module_count"] == 1 and before["om_count"] == 1
    )
    has_complete_cache = (
        after["compiled_module_count"] == 1 and after["om_count"] == 1
    )
    if recompiled:
        status = "recompiled_cache_invalidated"
        exit_code = 3
    elif not had_complete_cache:
        status = "cache_built_requires_fresh_process_reload"
        exit_code = 4
    elif not has_complete_cache:
        status = "cache_incomplete"
        exit_code = 5
    else:
        status = "ok"
        exit_code = 0
    result = {
        "status": status,
        "exit_code": exit_code,
        "config": {
            "batch_size": args.batch_size,
            "self_cache_length": args.self_cache_length,
            "cross_cache_length": args.cross_cache_length,
            "device": args.device,
            "passes": args.passes,
            "cache_parent": str(cache_parent),
        },
        "setup_s": setup_s,
        "pass_wall_s": pass_wall_s,
        "production_contract": {
            "arena_allocator": "ContinuousUniRecDecoder._allocate_empty_arena",
            "device_input_allocator": (
                "ContinuousUniRecDecoder._allocate_decode_device_inputs"
            ),
            "device_inputs_are_inference_tensors": bool(
                decoder_input_ids.is_inference()
                and cache_position.is_inference()
            ),
        },
        "cache_before": before,
        "cache_after": after,
        "warnings": warning_rows,
    }
    atomic_write_json(args.output.expanduser().resolve(), result)
    print(
        "UNIREC_DECODE_CACHE_CONTRACT_END "
        f"status={status} exit={exit_code} "
        f"cache_before={before['compiled_module_count']}/{before['om_count']} "
        f"cache_after={after['compiled_module_count']}/{after['om_count']} "
        f"recompiled={str(recompiled).lower()} output={args.output}",
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
