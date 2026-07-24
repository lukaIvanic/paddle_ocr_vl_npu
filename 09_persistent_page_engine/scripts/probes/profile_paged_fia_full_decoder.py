#!/usr/bin/env python3
"""Profile matched compiled IncreFA and paged-FIA full decoder steps.

This reuses the exact model construction and compiled graph cache keys from
``benchmark_paged_fia_full_decoder.py``. Each profiler lane contains only the
full one-token decoder step and argmax:

    embedding -> 18 decoder layers -> final norm -> LM head -> argmax

Model construction, cache allocation, graph loading, and warmup are outside the
captured regions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch_npu

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import benchmark_paged_fia_full_decoder as bench

from paddleocr_vl.model.text_decode import (
    TextDecodeRuntime,
    cast_decode_linear_weights_to_nz,
    prepare_decode_optimization_modules,
)
from utils.timing import synchronize


DEFAULT_PROFILE_ROOT = (
    REPO_ROOT
    / ".runtime_cache/09_persistent_page_engine_profiles"
    / "paged_fia_full_decoder"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/text_decode_lab"
    / "paged_fia_full_decoder_profile.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=bench.DEFAULT_MODEL)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=bench.DEFAULT_CACHE_ROOT,
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=DEFAULT_PROFILE_ROOT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache-length", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--paged-cache-update",
        choices=bench.CACHE_UPDATE_MODES,
        default="scatter_nd",
    )
    parser.add_argument(
        "--paged-single-stream",
        action="store_true",
        help=(
            "Compile both comparison lanes with ge.enableSingleStream=true; "
            "TorchAir requires one global stream mode per process."
        ),
    )
    parser.add_argument("--position", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile-iters", type=int, default=5)
    parser.add_argument(
        "--profile-metric",
        choices=("pipe", "memory", "l2", "memory_access"),
        default="pipe",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.cache_length <= 0:
        parser.error("--cache-length must be positive")
    if args.block_size <= 0 or args.cache_length % args.block_size:
        parser.error("--block-size must evenly divide --cache-length")
    if args.position < 0 or args.position >= args.cache_length:
        parser.error("--position must fit in --cache-length")
    if args.warmup < 0 or args.profile_iters <= 0:
        parser.error("--warmup must be non-negative and profile-iters positive")
    return args


def _profiler_config(metric: str) -> Any:
    import torch_npu.profiler as npu_prof

    metrics = {
        "pipe": npu_prof.AiCMetrics.PipeUtilization,
        "memory": npu_prof.AiCMetrics.Memory,
        "l2": npu_prof.AiCMetrics.L2Cache,
        "memory_access": npu_prof.AiCMetrics.MemoryAccess,
    }
    return npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1,
        aic_metrics=metrics[metric],
        l2_cache=metric == "l2",
        export_type=npu_prof.ExportType.Text,
        data_simplification=False,
    )


def _profile_lane(
    *,
    name: str,
    fn: Callable[..., torch.Tensor],
    call_args: tuple[torch.Tensor, ...],
    device: torch.device,
    profile_dir: Path,
    metric: str,
    warmup: int,
    profile_iters: int,
) -> dict[str, Any]:
    import torch_npu.profiler as npu_prof

    shutil.rmtree(profile_dir, ignore_errors=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(warmup):
        logits = fn(*call_args)
        torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
    synchronize(device)

    schedule = npu_prof.schedule(wait=0, warmup=0, active=1, repeat=1)
    wall_started = time.perf_counter()
    with npu_prof.profile(
        activities=[
            npu_prof.ProfilerActivity.CPU,
            npu_prof.ProfilerActivity.NPU,
        ],
        schedule=schedule,
        experimental_config=_profiler_config(metric),
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(profile_dir),
            analyse_flag=True,
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
        with_modules=False,
        with_flops=False,
    ) as profiler:
        with torch.profiler.record_function(
            f"paddleocr_vl.full_decoder.{name}"
        ):
            for _ in range(profile_iters):
                logits = fn(*call_args)
                torch.argmax(
                    logits[:, -1, :].float(),
                    dim=-1,
                    keepdim=True,
                )
        synchronize(device)
        profiler.step()
    synchronize(device)
    return {
        "profile_dir": str(profile_dir),
        "warmup_steps_outside_profiler": warmup,
        "captured_steps": profile_iters,
        "profile_wall_s": time.perf_counter() - wall_started,
        "profile_wall_is_throughput_measurement": False,
    }


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.npu.is_available():
        raise RuntimeError("profiling requires an available Ascend NPU")
    device = torch.device("npu:0")
    dtype = torch.float16
    torch.npu.set_compile_mode(jit_compile=False)
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
    linear_weight_format = str(weight_format["effective_mode"])

    if args.paged_single_stream:
        incre_runtime = bench._single_stream_incre_runtime(
            model,
            device=device,
            cache_root=args.cache_dir,
            batch_size=args.batch_size,
            cache_length=args.cache_length,
            dtype=dtype,
            optimization=optimization,
        )
    else:
        incre_runtime = TextDecodeRuntime(
            model,
            backend="torchair",
            device=device,
            cache_root=args.cache_dir,
            batch_size=args.batch_size,
            cache_length=args.cache_length,
            dtype=dtype,
            model_dir=args.model,
            linear_weight_format=linear_weight_format,
            optimization=optimization,
        )
    paged_stage = bench.PagedFIATextDecodeStage(
        model,
        block_size=args.block_size,
        cache_update_mode=args.paged_cache_update,
        optimization=optimization,
    ).eval()
    paged_fn, paged_compile = bench._compile_paged_stage(
        paged_stage,
        cache_root=args.cache_dir,
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        block_size=args.block_size,
        cache_update_mode=args.paged_cache_update,
        single_stream=args.paged_single_stream,
    )

    dense_cache, paged_cache = bench._allocate_matching_caches(
        config.text_config,
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        block_size=args.block_size,
        device=device,
        dtype=dtype,
        seed=args.seed + 1000 + args.position,
    )
    input_ids = (
        torch.arange(
            args.batch_size,
            device=device,
            dtype=torch.int64,
        ).view(-1, 1)
        + 17
    )
    cache_position = torch.full(
        (args.batch_size,),
        args.position,
        device=device,
        dtype=torch.int64,
    )
    rope_deltas = torch.zeros(
        (args.batch_size, 1),
        device=device,
        dtype=torch.int64,
    )
    incre_args = (
        input_ids,
        cache_position,
        rope_deltas,
        *dense_cache.flat_tensors(),
    )
    paged_args = (
        input_ids,
        cache_position,
        rope_deltas,
        paged_cache.block_table,
        *paged_cache.flat_tensors(),
    )

    # Load or compile the paged graph before either profiler capture.
    paged_logits = paged_fn(*paged_args)
    incre_logits = incre_runtime.fn(*incre_args)
    synchronize(device)
    correctness = bench._delta_stats(paged_logits, incre_logits)
    correctness["argmax_matches"] = int(
        (
            torch.argmax(paged_logits[:, -1, :].float(), dim=-1)
            == torch.argmax(incre_logits[:, -1, :].float(), dim=-1)
        )
        .sum()
        .cpu()
    )
    correctness["argmax_total"] = args.batch_size

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stream_mode = (
        "single_stream"
        if args.paged_single_stream
        else "multi_stream"
    )
    run_root = (
        args.profile_root.expanduser().resolve()
        / (
            f"b{args.batch_size}_k{args.cache_length}_"
            f"p{args.position}_{stream_mode}_"
            f"{args.profile_metric}_{run_stamp}"
        )
    )
    incre_profile = _profile_lane(
        name="increfa",
        fn=incre_runtime.fn,
        call_args=incre_args,
        device=device,
        profile_dir=run_root / "increfa",
        metric=args.profile_metric,
        warmup=args.warmup,
        profile_iters=args.profile_iters,
    )
    paged_profile = _profile_lane(
        name="paged_fia_v2",
        fn=paged_fn,
        call_args=paged_args,
        device=device,
        profile_dir=run_root / "paged_fia_v2",
        metric=args.profile_metric,
        warmup=args.warmup,
        profile_iters=args.profile_iters,
    )

    result = {
        "schema_version": 1,
        "kind": "compiled_full_decoder_attention_profile",
        "configuration": {
            "batch_size": args.batch_size,
            "cache_length": args.cache_length,
            "actual_kv_length": args.position + 1,
            "block_size": args.block_size,
            "paged_cache_update": args.paged_cache_update,
            "paged_single_stream": args.paged_single_stream,
            "paged_metadata_scope": (
                "once_per_decode_step_before_18_layer_loop"
            ),
            "dtype": str(dtype),
            "optimization": bench.OPTIMIZATION,
            "profile_metric": args.profile_metric,
            "profile_iters": args.profile_iters,
            "full_step": (
                "embedding_18_layers_final_norm_lm_head_argmax"
            ),
        },
        "architecture": {
            "decoder_layer_parameters": sum(
                parameter.numel()
                for parameter in model.model.layers.parameters()
            ),
            "num_hidden_layers": config.text_config.num_hidden_layers,
            "hidden_size": config.text_config.hidden_size,
            "intermediate_size": config.text_config.intermediate_size,
            "num_attention_heads": config.text_config.num_attention_heads,
            "num_key_value_heads": (
                config.text_config.num_key_value_heads
            ),
            "head_dim": config.text_config.head_dim,
        },
        "correctness": correctness,
        "compile": {
            "increfa": incre_runtime.metadata,
            "increfa_setup_detail_s": incre_runtime.setup_timing_s,
            "paged_fia_v2": paged_compile,
        },
        "profiles": {
            "increfa": incre_profile,
            "paged_fia_v2": paged_profile,
        },
        "parser": str(
            REPO_ROOT
            / "05_full_recognizer_optimizations/parse_npu_profile.py"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
