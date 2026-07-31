#!/usr/bin/env python3
"""Probe the production dense decode graph at exact mixed cache positions.

This is a shape/value control for device-specific decode stalls.  It uses the
same ``TextDecodeRuntime`` as Experiment 09, keeps the physical batch and KV
cache static, and changes only one row's runtime ``cache_position``.  Every
step is synchronized separately and recorded before and after the sync so a
device-side hang leaves a precise final progress record.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPO_ROOT / "09_persistent_page_engine"

import sys

if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_paged_fia_full_decoder as bench  # noqa: E402

from paddleocr_vl.model.text_decode import (  # noqa: E402
    TextDecodeRuntime,
    cast_decode_linear_weights_to_nz,
    prepare_decode_optimization_modules,
)
from utils.timing import synchronize  # noqa: E402


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/text_decode_lab"
    / "dense_decode_position_boundary.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=bench.DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=bench.DEFAULT_CACHE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--target-row", type=int, default=3)
    parser.add_argument("--inactive-position", type=int, default=0)
    parser.add_argument("--positions", default="1277,1278,1279,1280,1281")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        args.positions = tuple(
            int(value.strip())
            for value in args.positions.split(",")
            if value.strip()
        )
    except ValueError as exc:
        parser.error(f"invalid --positions: {exc}")
    if args.batch_size <= 0 or args.cache_length <= 0:
        parser.error("--batch-size and --cache-length must be positive")
    if not 0 <= args.target_row < args.batch_size:
        parser.error("--target-row must be inside the physical batch")
    if not args.positions:
        parser.error("--positions must not be empty")
    all_positions = (*args.positions, args.inactive_position)
    if min(all_positions) < 0 or max(all_positions) >= args.cache_length:
        parser.error("all positions must be inside the physical KV cache")
    return args


class ProgressRecorder:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> None:
        row = {
            "event": event,
            "monotonic_s": time.monotonic(),
            **fields,
        }
        line = json.dumps(row, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        print("DENSE_DECODE_BOUNDARY " + line, flush=True)


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.npu.is_available():
        raise RuntimeError("probe requires an available Ascend NPU")
    device = torch.device("npu:0")
    dtype = torch.float16
    torch.npu.set_compile_mode(jit_compile=False)

    output = args.output.expanduser().resolve()
    progress = ProgressRecorder(output.with_suffix(".progress.jsonl"))
    progress.emit(
        "setup_begin",
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        target_row=args.target_row,
        inactive_position=args.inactive_position,
        positions=list(args.positions),
    )

    config = bench.PaddleOCRVLConfig.from_model_dir(args.model)
    model = bench._create_random_model(
        config,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    optimization = prepare_decode_optimization_modules(
        model,
        bench.OPTIMIZATION,
    )
    weight_format = cast_decode_linear_weights_to_nz(model)
    synchronize(device)
    runtime = TextDecodeRuntime(
        model,
        backend="torchair",
        device=device,
        cache_root=args.cache_dir,
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        dtype=dtype,
        model_dir=args.model,
        linear_weight_format=str(weight_format["effective_mode"]),
        optimization=optimization,
    )
    cache = model.allocate_static_cache(
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        device=device,
        dtype=dtype,
        init_mode="zeros",
    )
    input_ids = torch.full(
        (args.batch_size, 1),
        int(config.text_config.eos_token_id),
        device=device,
        dtype=torch.int64,
    )
    input_ids[args.target_row].fill_(17)
    rope_deltas = torch.zeros(
        (args.batch_size, 1),
        device=device,
        dtype=torch.int64,
    )
    progress.emit(
        "setup_end",
        runtime_metadata=runtime.metadata,
        runtime_setup_timing_s=runtime.setup_timing_s,
        weight_format=weight_format,
    )

    steps: list[dict[str, Any]] = []
    for step_index, target_position in enumerate(args.positions):
        cache_positions = torch.full(
            (args.batch_size,),
            args.inactive_position,
            device=device,
            dtype=torch.int64,
        )
        cache_positions[args.target_row].fill_(target_position)
        progress.emit(
            "step_enqueue_begin",
            step=step_index,
            target_position=target_position,
            effective_length=target_position + 1,
            batch_positions=[
                target_position if row == args.target_row else args.inactive_position
                for row in range(args.batch_size)
            ],
        )
        started = time.perf_counter()
        logits = runtime.fn(
            input_ids,
            cache_positions,
            rope_deltas,
            *cache.flat_tensors(),
        )
        progress.emit(
            "step_sync_begin",
            step=step_index,
            target_position=target_position,
            effective_length=target_position + 1,
        )
        synchronize(device)
        elapsed_s = time.perf_counter() - started
        progress.emit(
            "step_sync_end",
            step=step_index,
            target_position=target_position,
            effective_length=target_position + 1,
            elapsed_s=elapsed_s,
        )
        target_logits = logits[args.target_row, -1].float()
        target_token = int(torch.argmax(target_logits).cpu())
        finite = bool(torch.isfinite(target_logits).all().cpu())
        step = {
            "step": step_index,
            "target_position": target_position,
            "effective_length": target_position + 1,
            "elapsed_s": elapsed_s,
            "target_token": target_token,
            "target_logits_finite": finite,
        }
        steps.append(step)
        progress.emit("step_result", **step)

    result = {
        "schema_version": 1,
        "kind": "dense_decode_mixed_position_boundary",
        "passed": all(step["target_logits_finite"] for step in steps),
        "configuration": {
            "model": str(args.model.expanduser().resolve()),
            "batch_size": args.batch_size,
            "cache_length": args.cache_length,
            "target_row": args.target_row,
            "inactive_position": args.inactive_position,
            "positions": list(args.positions),
            "optimization": bench.OPTIMIZATION,
            "cache_initialization": "zeros",
        },
        "runtime": {
            "metadata": runtime.metadata,
            "setup_timing_s": runtime.setup_timing_s,
            "weight_format": weight_format,
        },
        "steps": steps,
        "progress_jsonl": str(progress.path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    progress.emit("probe_end", passed=result["passed"], output=str(output))
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
