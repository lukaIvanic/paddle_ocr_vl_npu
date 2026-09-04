#!/usr/bin/env python3
"""Compile and benchmark text speculative verification by batch and draft length."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.modeling import (  # noqa: E402
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
)
from paddleocr_vl.model.text_decode import (  # noqa: E402
    TextDecodeRuntime,
    cast_decode_linear_weights_to_nz,
    load_decode_vocab_token_ids,
    prepare_decode_compact_lm_head,
    prepare_decode_optimization_modules,
    torchair_cache_dir_for_shape,
)
from paddleocr_vl.model.text_spec_verify import (  # noqa: E402
    TextSpecVerifyRuntime,
    torchair_cache_dir_for_spec_shape,
)
from utils.timing import DeviceTimeline, synchronize  # noqa: E402


DEFAULT_MODEL = Path("/workspace/models/PaddleOCR-VL-1.6")
DEFAULT_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_torchair"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/text_spec_verify_lab/results.json"
)
DEFAULT_DRAFT_LENGTHS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_BATCH_SIZES = (1,)
DECODE_OPTIMIZATION = "combined_apply_pse_sentinel"
SPEC_OPTIMIZATION = "combined_apply"


def _parse_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(
            "value must be a non-empty list of positive integers"
        )
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-length", type=int, default=768)
    parser.add_argument("--profile-position", type=int, default=256)
    parser.add_argument("--decode-vocab-token-ids", type=Path)
    parser.add_argument(
        "--decode-optimization",
        default=DECODE_OPTIMIZATION,
    )
    parser.add_argument(
        "--spec-optimization",
        default=SPEC_OPTIMIZATION,
    )
    parser.add_argument(
        "--draft-lengths",
        type=_parse_ints,
        default=DEFAULT_DRAFT_LENGTHS,
    )
    parser.add_argument(
        "--batch-sizes",
        type=_parse_ints,
        default=DEFAULT_BATCH_SIZES,
        help="Comma-separated static speculative-verification batch sizes.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--allow-compile",
        action="store_true",
        help="Allow any missing static TorchAir graph to compile.",
    )
    args = parser.parse_args(argv)
    if args.cache_length <= 0:
        parser.error("--cache-length must be positive")
    if args.profile_position < 0:
        parser.error("--profile-position must be non-negative")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats positive")
    maximum_query = max(args.draft_lengths) + 1
    if args.profile_position + maximum_query > args.cache_length:
        parser.error(
            "profile position plus the largest D+1 query exceeds the cache: "
            f"{args.profile_position}+{maximum_query}>{args.cache_length}"
        )
    return args


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _write_progress(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _measure(
    device: torch.device,
    fn: Any,
    *,
    warmup: int,
    repeats: int,
) -> tuple[list[float], torch.Tensor, float]:
    output: torch.Tensor | None = None
    for _ in range(warmup):
        output = fn()
    synchronize(device)
    timeline = DeviceTimeline(device)
    wall_started = time.perf_counter()
    for index in range(repeats):
        output = timeline.measure(f"step_{index:04d}", fn)
    durations = list(timeline.resolve().values())
    host_wall_s = time.perf_counter() - wall_started
    if output is None:
        raise AssertionError("measurement produced no output")
    return durations, output, host_wall_s


def _timing_summary(
    durations: list[float],
    *,
    recovered_tokens_per_call: int,
    host_wall_s: float,
) -> dict[str, Any]:
    total_s = sum(durations)
    calls = len(durations)
    return {
        "device_s": total_s,
        "latency_ms": {
            "mean": statistics.mean(durations) * 1000.0,
            "median": statistics.median(durations) * 1000.0,
            "p95": _percentile(durations, 0.95) * 1000.0,
            "min": min(durations) * 1000.0,
            "max": max(durations) * 1000.0,
        },
        "graph_calls_per_s": calls / total_s,
        "host_wall_s": float(host_wall_s),
        "host_graph_calls_per_s": calls / host_wall_s,
        "recovered_tokens_per_call": int(recovered_tokens_per_call),
        "effective_recovered_tok_per_s": (
            calls * int(recovered_tokens_per_call) / total_s
        ),
        "host_effective_recovered_tok_per_s": (
            calls * int(recovered_tokens_per_call) / host_wall_s
        ),
    }


def _zero_cache(cache: Any) -> None:
    for tensor in cache.flat_tensors():
        tensor.zero_()


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = True
    device = torch.device("npu:0")
    if not torch.npu.is_available():
        raise RuntimeError("text speculative-verification lab requires an NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    dtype = torch.float16
    model_dir = _resolve_model_dir(args.model)
    output_path = args.output.expanduser().resolve()

    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "text_spec_verify_draft_sweep",
        "status": "setup",
        "contract": {
            # Keep the historical scalar contract for the default B1 run,
            # while making multi-B sweeps explicit.
            "batch_size": (
                int(args.batch_sizes[0]) if len(args.batch_sizes) == 1 else None
            ),
            "batch_sizes": list(args.batch_sizes),
            "cache_length": int(args.cache_length),
            "profile_position": int(args.profile_position),
            "draft_lengths": list(args.draft_lengths),
            "query_length": "draft_length + 1",
            "fully_accepted_tokens_per_call": "draft_length + 1",
            "decode_optimization": args.decode_optimization,
            "spec_optimization": args.spec_optimization,
            "spec_attention": "PromptFA GQA over persistent KV arena",
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
        },
        "setup": {},
        "decode_b1": None,
        "spec_verify": [],
    }
    _write_progress(output_path, result)

    setup_started = time.perf_counter()
    print("SPEC_VERIFY_PROGRESS stage=model_load status=begin", flush=True)
    started = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=dtype,
        device=device,
    ).eval()
    synchronize(device)
    model_load_s = time.perf_counter() - started
    print(
        f"SPEC_VERIFY_PROGRESS stage=model_load status=end seconds={model_load_s:.3f}",
        flush=True,
    )

    decode_vocab: dict[str, Any] = {"enabled": False}
    if args.decode_vocab_token_ids is not None:
        token_ids, decode_vocab = load_decode_vocab_token_ids(
            args.decode_vocab_token_ids,
            full_vocab_size=int(model.lm_head.weight.shape[0]),
        )
        prepare_decode_compact_lm_head(model, token_ids)
        synchronize(device)

    prepare_decode_optimization_modules(model, args.spec_optimization)
    started = time.perf_counter()
    weight_format = cast_decode_linear_weights_to_nz(model)
    synchronize(device)
    weight_format_s = time.perf_counter() - started
    linear_weight_format = str(weight_format["effective_mode"])
    result["setup"] = {
        "model_dir": str(model_dir),
        "model_load_s": model_load_s,
        "weight_format_s": weight_format_s,
        "weight_format": weight_format,
        "decode_vocab": decode_vocab,
    }
    _write_progress(output_path, result)

    decode_cache_dir = torchair_cache_dir_for_shape(
        args.cache_dir,
        batch_size=1,
        cache_length=args.cache_length,
        dtype=dtype,
        device=device,
        model_dir=model_dir,
        linear_weight_format=linear_weight_format,
        optimization=args.decode_optimization,
    )
    if not decode_cache_dir.is_dir() and not args.allow_compile:
        raise RuntimeError(
            "missing B1 decode graph cache; rerun with --allow-compile:\n"
            f"  - {decode_cache_dir}"
        )
    print("SPEC_VERIFY_PROGRESS lane=decode_b1 status=setup_begin", flush=True)
    decode_runtime = TextDecodeRuntime(
        model,
        backend="torchair",
        device=device,
        cache_root=args.cache_dir,
        batch_size=1,
        cache_length=args.cache_length,
        dtype=dtype,
        model_dir=model_dir,
        linear_weight_format=linear_weight_format,
        optimization=args.decode_optimization,
    )
    decode_input = torch.ones((1, 1), device=device, dtype=torch.int64)
    decode_position = torch.tensor(
        [args.profile_position], device=device, dtype=torch.int64
    )
    rope_deltas = torch.zeros((1, 1), device=device, dtype=torch.int64)
    compact_decode_output = hasattr(model, "decode_token_id_map")

    def decode_target_ids(output: torch.Tensor) -> torch.Tensor:
        if compact_decode_output:
            return output
        return torch.argmax(output, dim=-1)

    def decode_step() -> torch.Tensor:
        output = decode_runtime.fn(
            decode_input,
            decode_position,
            rope_deltas,
            *decode_runtime.warm_cache.flat_tensors(),
        )
        return decode_target_ids(output)

    durations, _decode_output, host_wall_s = _measure(
        device,
        decode_step,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    result["decode_b1"] = {
        **_timing_summary(
            durations,
            recovered_tokens_per_call=1,
            host_wall_s=host_wall_s,
        ),
        "runtime": decode_runtime.metadata,
    }
    print(
        "SPEC_VERIFY_RESULT lane=decode_b1 "
        f"latency_ms={result['decode_b1']['latency_ms']['median']:.3f} "
        f"tok_s={result['decode_b1']['effective_recovered_tok_per_s']:.1f}",
        flush=True,
    )
    _write_progress(output_path, result)

    reference_cache = model.allocate_static_cache(
        batch_size=1,
        cache_length=args.cache_length,
        device=device,
        dtype=dtype,
        init_mode="zeros",
    )

    total_lanes = len(args.batch_sizes) * len(args.draft_lengths)
    lane_index = 0
    vocab_size = int(model.config.text_config.vocab_size)

    for draft_length in args.draft_lengths:
        query_length = int(draft_length + 1)
        # A B1 graph is the exact-token reference for every batched row. If
        # B1 is one of the measured lanes, that lane also owns this runtime.
        b1_reference_runtime: TextSpecVerifyRuntime | None = None

        for batch_size in args.batch_sizes:
            lane_index += 1
            cache_dir = torchair_cache_dir_for_spec_shape(
                args.cache_dir,
                batch_size=batch_size,
                draft_length=draft_length,
                cache_length=args.cache_length,
                dtype=dtype,
                device=device,
                model_dir=model_dir,
                linear_weight_format=linear_weight_format,
                optimization=args.spec_optimization,
            )
            cache_hit = cache_dir.is_dir() and any(cache_dir.iterdir())
            if not cache_hit and not args.allow_compile:
                raise RuntimeError(
                    f"missing B{batch_size} D{draft_length} graph cache; "
                    f"rerun with --allow-compile:\n  - {cache_dir}"
                )
            print(
                "SPEC_VERIFY_PROGRESS "
                f"lane={lane_index}/{total_lanes} "
                f"batch=B{batch_size} draft=D{draft_length} "
                f"query={query_length} "
                f"cache={'hit' if cache_hit else 'compile'} "
                "status=setup_begin",
                flush=True,
            )
            lane_started = time.perf_counter()
            runtime = TextSpecVerifyRuntime(
                model,
                batch_size=batch_size,
                device=device,
                cache_root=args.cache_dir,
                draft_length=draft_length,
                cache_length=args.cache_length,
                dtype=dtype,
                model_dir=model_dir,
                linear_weight_format=linear_weight_format,
                optimization=args.spec_optimization,
            )
            # Give every row a different deterministic sequence. This catches
            # cross-row cache or indexing errors that identical rows can hide.
            input_ids = (
                torch.arange(
                    1,
                    batch_size * query_length + 1,
                    device=device,
                    dtype=torch.int64,
                )
                .remainder(vocab_size - 1)
                .add(1)
                .view(batch_size, query_length)
            )
            cache_position = torch.full(
                (batch_size,),
                args.profile_position,
                device=device,
                dtype=torch.int64,
            )
            spec_rope_deltas = torch.zeros(
                (batch_size, 1), device=device, dtype=torch.int64
            )

            _zero_cache(runtime.warm_cache)

            def spec_step() -> torch.Tensor:
                return runtime.fn(
                    input_ids,
                    cache_position,
                    spec_rope_deltas,
                    *runtime.warm_cache.flat_tensors(),
                )

            durations, spec_targets, host_wall_s = _measure(
                device,
                spec_step,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            # TorchAir can reuse graph-output storage across calls. Preserve
            # the measured call before any B1 reference graph is invoked.
            spec_targets = spec_targets.clone()

            if batch_size == 1 and b1_reference_runtime is None:
                b1_reference_runtime = runtime
            if b1_reference_runtime is None:
                b1_cache_dir = torchair_cache_dir_for_spec_shape(
                    args.cache_dir,
                    batch_size=1,
                    draft_length=draft_length,
                    cache_length=args.cache_length,
                    dtype=dtype,
                    device=device,
                    model_dir=model_dir,
                    linear_weight_format=linear_weight_format,
                    optimization=args.spec_optimization,
                )
                b1_cache_hit = b1_cache_dir.is_dir() and any(
                    b1_cache_dir.iterdir()
                )
                if not b1_cache_hit and not args.allow_compile:
                    raise RuntimeError(
                        "missing B1 correctness-reference graph cache; "
                        f"rerun with --allow-compile:\n  - {b1_cache_dir}"
                    )
                b1_reference_runtime = TextSpecVerifyRuntime(
                    model,
                    batch_size=1,
                    device=device,
                    cache_root=args.cache_dir,
                    draft_length=draft_length,
                    cache_length=args.cache_length,
                    dtype=dtype,
                    model_dir=model_dir,
                    linear_weight_format=linear_weight_format,
                    optimization=args.spec_optimization,
                )

            # Compare every row against the same B1 graph using exact integer
            # target IDs. Each reference call starts from an all-zero cache.
            b1_targets = []
            for row_index in range(batch_size):
                _zero_cache(b1_reference_runtime.warm_cache)
                row_targets = b1_reference_runtime.fn(
                    input_ids[row_index : row_index + 1],
                    cache_position[row_index : row_index + 1],
                    spec_rope_deltas[row_index : row_index + 1],
                    *b1_reference_runtime.warm_cache.flat_tensors(),
                )
                b1_targets.append(row_targets.clone())
            b1_targets_tensor = torch.cat(b1_targets, dim=0)
            synchronize(device)
            spec_cpu = spec_targets.detach().cpu()
            b1_cpu = b1_targets_tensor.detach().cpu()
            agreement = spec_cpu == b1_cpu
            exact_positions = int(agreement.sum().item())
            target_count = int(agreement.numel())
            per_row = []
            mismatch_examples = []
            for row_index in range(batch_size):
                row_agreement = agreement[row_index]
                row_exact = int(row_agreement.sum().item())
                mismatch_positions = torch.nonzero(
                    ~row_agreement, as_tuple=False
                ).reshape(-1)
                first_mismatch = (
                    int(mismatch_positions[0].item())
                    if int(mismatch_positions.numel()) > 0
                    else None
                )
                per_row.append(
                    {
                        "row": int(row_index),
                        "exact_positions": row_exact,
                        "positions": query_length,
                        "first_mismatch_position": first_mismatch,
                    }
                )
                for mismatch_position in mismatch_positions[:4].tolist():
                    if len(mismatch_examples) >= 16:
                        break
                    mismatch_examples.append(
                        {
                            "row": int(row_index),
                            "position": int(mismatch_position),
                            "batch_target_id": int(
                                spec_cpu[row_index, mismatch_position].item()
                            ),
                            "b1_target_id": int(
                                b1_cpu[row_index, mismatch_position].item()
                            ),
                        }
                    )

            # Preserve the historical B1 teacher-forced serial-decode check.
            serial_agreement: dict[str, Any] | None = None
            if batch_size == 1:
                _zero_cache(reference_cache)
                serial_targets = []
                for query_index in range(query_length):
                    position = torch.tensor(
                        [args.profile_position + query_index],
                        device=device,
                        dtype=torch.int64,
                    )
                    decode_output = decode_runtime.fn(
                        input_ids[:, query_index : query_index + 1],
                        position,
                        rope_deltas,
                        *reference_cache.flat_tensors(),
                    )
                    serial_targets.append(decode_target_ids(decode_output))
                serial_targets_tensor = torch.cat(serial_targets, dim=1)
                synchronize(device)
                serial_cpu = serial_targets_tensor.detach().cpu()
                serial_exact = int((spec_cpu == serial_cpu).sum().item())
                serial_count = int(spec_cpu.numel())
                serial_agreement = {
                    "exact_positions": serial_exact,
                    "positions": serial_count,
                    "fraction": serial_exact / serial_count,
                }

            timing = _timing_summary(
                durations,
                recovered_tokens_per_call=query_length,
                host_wall_s=host_wall_s,
            )
            calls = len(durations)
            physical_tokens_per_call = batch_size * query_length
            lane_result = {
                "batch_size": int(batch_size),
                "draft_length": int(draft_length),
                "query_length": query_length,
                "fully_accepted_tokens_per_call": query_length,
                "physical_verified_tokens_per_call": physical_tokens_per_call,
                "physical_verified_tok_per_s": (
                    calls * physical_tokens_per_call / timing["device_s"]
                ),
                "host_physical_verified_tok_per_s": (
                    calls * physical_tokens_per_call / host_wall_s
                ),
                "cache_was_warm": bool(cache_hit),
                **timing,
                "batch_vs_b1_spec_target_agreement": {
                    "comparison": "exact_token_id",
                    "distinct_input_sequence_per_row": True,
                    "exact_positions": exact_positions,
                    "positions": target_count,
                    "fraction": exact_positions / target_count,
                    "exact_rows": sum(
                        row["exact_positions"] == row["positions"]
                        for row in per_row
                    ),
                    "rows": int(batch_size),
                    "per_row": per_row,
                    "mismatch_examples": mismatch_examples,
                },
                "serial_decode_target_agreement": serial_agreement,
                "runtime": runtime.metadata,
                "lane_wall_s": time.perf_counter() - lane_started,
            }
            result["spec_verify"].append(lane_result)
            _write_progress(output_path, result)
            print(
                "SPEC_VERIFY_RESULT "
                f"batch=B{batch_size} draft=D{draft_length} "
                f"query={query_length} "
                f"latency_ms={lane_result['latency_ms']['median']:.3f} "
                f"winner_tok_s={lane_result['effective_recovered_tok_per_s']:.1f} "
                f"physical_tok_s={lane_result['physical_verified_tok_per_s']:.1f} "
                f"b1_target_match={exact_positions}/{target_count} "
                f"lane_wall_s={lane_result['lane_wall_s']:.1f}",
                flush=True,
            )
            del runtime

        if b1_reference_runtime is not None:
            del b1_reference_runtime

    result["status"] = "complete"
    result["setup"]["total_wall_s"] = time.perf_counter() - setup_started
    _write_progress(output_path, result)
    print(
        "SPEC_VERIFY_COMPLETE "
        f"lanes={len(result['spec_verify'])} "
        f"wall_s={result['setup']['total_wall_s']:.1f} "
        f"output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
