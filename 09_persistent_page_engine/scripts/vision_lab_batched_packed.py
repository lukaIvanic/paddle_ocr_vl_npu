#!/usr/bin/env python3
"""Measure compiled PromptFA batching with packed crops inside each batch row."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.modeling import LocalPaddleOCRVLForConditionalGeneration
from paddleocr_vl.model.compile_utils import (
    TORCHAIR_EXECUTION_MODE,
    cache_key_part,
    import_torchair,
    short_file_hash,
    torch_npu_version_label,
    torchair_version_label,
)
from paddleocr_vl.model.vision_prefill import (
    VisionPrefillRuntime,
    VisionPrefillStage,
    get_vision_prompt_fa_layout,
    get_vision_prompt_fa_mask_sparse_mode,
    unique_bucket_forward,
    vision_source_hash,
)
from utils.timing import DeviceTimeline, synchronize
from vision_lab import (
    DEFAULT_MODEL,
    _environment,
    _materialize_inputs,
    _memory_baseline,
    _packed_prepared,
    _peak_memory_delta,
    _route,
    _run_group,
)
from vision_lab_phase0 import (
    DEFAULT_CORPUS,
    MAX_COMPILED_LADDER,
    _load_corpora,
    _packing_groups,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/vision_lab"
    / "batched_packed_eager.json"
)
DEFAULT_CACHE_ROOT = Path("/dev/shm/vision_lab_cache_all")


def _shape(value: str) -> tuple[int, int]:
    try:
        batch, sequence = (int(piece) for piece in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("shape must look like BxS, e.g. 2x1536") from exc
    if batch <= 0 or sequence <= 0:
        raise argparse.ArgumentTypeError("batch and sequence must be positive")
    return batch, sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--variant", default="min_pixels_28224")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--shape",
        type=_shape,
        action="append",
        default=[],
        help="Repeat static compiled shapes; default: 1x3072, 2x1536, 4x768.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.shape = tuple(args.shape or ((1, 3072), (2, 1536), (4, 768)))
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats positive")
    return args


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _chunk_rows(
    rows: list[list[dict[str, Any]]], batch_size: int
) -> list[list[list[dict[str, Any]]]]:
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def _run_batched_packed(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    run_tower: Callable[..., torch.Tensor],
    rows: list[list[dict[str, Any]]],
    batch_size: int,
    sequence_length: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    flat_items = [item for row in rows for item in row]
    grids, pixels = _materialize_inputs(
        flat_items,
        seed=seed,
        dtype=dtype,
        device=device,
    )
    row_prepared = []
    offset = 0
    for row in rows:
        end = offset + len(row)
        hidden = [
            model.visual.vision_model.embeddings(
                pixel.unsqueeze(0),
                image_grid_thw=grid,
            )
            for pixel, grid in zip(pixels[offset:end], grids[offset:end])
        ]
        row_prepared.append(
            _packed_prepared(
                model,
                hidden,
                grids[offset:end],
                physical=sequence_length,
                execution="eager_padded",
            )
        )
        offset = end
    if not row_prepared:
        raise ValueError("batched-packed call requires at least one real row")
    template = row_prepared[0]
    while len(row_prepared) < batch_size:
        row_prepared.append(
            type(template)(
                prefix_hidden_states=torch.zeros_like(template.prefix_hidden_states),
                rope_cos=torch.ones_like(template.rope_cos),
                rope_sin=torch.zeros_like(template.rope_sin),
                attention_mask=torch.zeros_like(template.attention_mask),
                real_seq_len=0,
                physical_seq_len=sequence_length,
                execution="eager_padded_dummy_row",
            )
        )
    prefix = torch.cat([item.prefix_hidden_states for item in row_prepared], dim=0)
    rope_cos = torch.cat([item.rope_cos for item in row_prepared], dim=0)
    rope_sin = torch.cat([item.rope_sin for item in row_prepared], dim=0)
    attention_mask = torch.cat([item.attention_mask for item in row_prepared], dim=0)
    timeline = DeviceTimeline(device)
    output = timeline.measure(
        "vision_tower",
        lambda: run_tower(prefix, rope_cos, rope_sin, attention_mask),
    )
    spans = timeline.resolve_spans()
    if tuple(output.shape[:2]) != (batch_size, sequence_length):
        raise RuntimeError(
            "compiled PromptFA batching returned the wrong static shape: "
            f"expected {(batch_size, sequence_length)}, got {tuple(output.shape)}"
        )
    real_tokens = sum(int(item["real_vision_tokens"]) for item in flat_items)
    physical_tokens = batch_size * sequence_length
    if real_tokens > physical_tokens:
        raise AssertionError("batched-packed real tokens exceed physical capacity")
    return {
        "tower_ms": float(spans["vision_tower"]["seconds"]) * 1000.0,
        "real_tokens": real_tokens,
        "physical_tokens": physical_tokens,
        "real_rows": len(rows),
        "dummy_rows": batch_size - len(rows),
    }


def _single_passthrough_ms(
    *,
    item: dict[str, Any],
    model: LocalPaddleOCRVLForConditionalGeneration,
    runtime: VisionPrefillRuntime,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> float:
    grids, pixels = _materialize_inputs([item], seed=seed, dtype=dtype, device=device)
    route = _route(
        strategy="single",
        lengths=[int(item["real_vision_tokens"])],
        execution="eager",
        buckets=(32,),
    )
    run = _run_group(
        model,
        runtime,
        [item],
        strategy="single",
        route=route,
        grids=grids,
        pixels=pixels,
        device=device,
    )
    return float(run["stage_ms"]["vision_tower"])


def _run_shape(
    *,
    corpus_name: str,
    items: list[dict[str, Any]],
    batch_size: int,
    sequence_length: int,
    model: LocalPaddleOCRVLForConditionalGeneration,
    run_tower: Callable[..., torch.Tensor],
    eager_runtime: VisionPrefillRuntime,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    repeats: int,
    passthrough_cache: dict[tuple[str, int], float],
) -> dict[str, Any]:
    overflow = [item for item in items if int(item["real_vision_tokens"]) > MAX_COMPILED_LADDER]
    in_scope = [item for item in items if int(item["real_vision_tokens"]) <= MAX_COMPILED_LADDER]
    packable = [item for item in in_scope if int(item["real_vision_tokens"]) <= sequence_length]
    passthrough = [item for item in in_scope if int(item["real_vision_tokens"]) > sequence_length]
    rows = _packing_groups(
        packable,
        target=sequence_length,
        policy="first_fit_decreasing",
    )
    batches = _chunk_rows(rows, batch_size)
    cases: list[dict[str, Any]] = []
    for case_index, batch_rows in enumerate(batches):
        for _ in range(warmup if case_index == 0 else 0):
            _run_batched_packed(
                model=model,
                run_tower=run_tower,
                rows=batch_rows,
                batch_size=batch_size,
                sequence_length=sequence_length,
                seed=seed,
                dtype=dtype,
                device=device,
            )
        baseline = _memory_baseline(device)
        runs = [
            _run_batched_packed(
                model=model,
                run_tower=run_tower,
                rows=batch_rows,
                batch_size=batch_size,
                sequence_length=sequence_length,
                seed=seed,
                dtype=dtype,
                device=device,
            )
            for _ in range(repeats)
        ]
        cases.append(
            {
                "case_index": case_index,
                "tower_ms": statistics.mean(run["tower_ms"] for run in runs),
                "tower_ms_samples": [run["tower_ms"] for run in runs],
                "real_tokens": runs[-1]["real_tokens"],
                "physical_tokens": runs[-1]["physical_tokens"],
                "real_rows": runs[-1]["real_rows"],
                "dummy_rows": runs[-1]["dummy_rows"],
                "peak_allocated_bytes_delta": _peak_memory_delta(device, baseline),
            }
        )
    passthrough_ms = 0.0
    for item in passthrough:
        key = (corpus_name, int(item["source_index"]))
        if key not in passthrough_cache:
            passthrough_cache[key] = _single_passthrough_ms(
                item=item,
                model=model,
                runtime=eager_runtime,
                seed=seed,
                dtype=dtype,
                device=device,
            )
        passthrough_ms += passthrough_cache[key]

    target_ms = sum(case["tower_ms"] for case in cases)
    target_real = sum(case["real_tokens"] for case in cases)
    target_physical = sum(case["physical_tokens"] for case in cases)
    total_ms = target_ms + passthrough_ms
    passthrough_tokens = sum(int(item["real_vision_tokens"]) for item in passthrough)
    total_real = target_real + passthrough_tokens
    total_physical = target_physical + passthrough_tokens
    if target_physical < target_real or total_physical < total_real:
        raise AssertionError("padding accounting underflow")
    return {
        "name": f"b{batch_size}_s{sequence_length}",
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "physical_tokens_per_full_call": batch_size * sequence_length,
        "supported": True,
        "target_batches": len(cases),
        "packed_rows": len(rows),
        "packable_crops": len(packable),
        "passthrough_crops": len(passthrough),
        "excluded_overflow_crops": len(overflow),
        "excluded_overflow_tokens": sum(
            int(item["real_vision_tokens"]) for item in overflow
        ),
        "target_batch_metrics": {
            "tower_s": target_ms / 1000.0,
            "real_tokens": target_real,
            "physical_tokens": target_physical,
            "padding_tokens": target_physical - target_real,
            "raw_physical_tokens_per_s": target_physical / (target_ms / 1000.0),
            "effective_real_tokens_per_s": target_real / (target_ms / 1000.0),
            "real_token_fraction": target_real / target_physical,
            "call_ms_mean": statistics.mean(case["tower_ms"] for case in cases),
            "call_ms_p50": _percentile([case["tower_ms"] for case in cases], 0.50),
            "call_ms_p95": _percentile([case["tower_ms"] for case in cases], 0.95),
            "peak_allocated_bytes_delta": max(
                (case["peak_allocated_bytes_delta"] or 0) for case in cases
            ),
        },
        "in_scope_corpus_projection": {
            "tower_s": total_ms / 1000.0,
            "real_tokens": total_real,
            "physical_tokens": total_physical,
            "padding_tokens": total_physical - total_real,
            "raw_physical_tokens_per_s": total_physical / (total_ms / 1000.0),
            "effective_real_tokens_per_s": total_real / (total_ms / 1000.0),
            "real_token_fraction": total_real / total_physical,
            "passthrough_tower_s": passthrough_ms / 1000.0,
        },
        "padding_audit": {
            "passed": True,
            "dummy_rows": sum(case["dummy_rows"] for case in cases),
            "max_case_padding_tokens": max(
                case["physical_tokens"] - case["real_tokens"] for case in cases
            ),
        },
        "cases": cases,
    }


def _compiled_shape(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    batch_size: int,
    sequence_length: int,
    cache_root: Path,
    model_dir: Path,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Callable[..., torch.Tensor], dict[str, Any]]:
    """Load or compile exactly one static BxS PromptFA vision graph."""
    torchair, CompilerConfig = import_torchair()
    hidden_size = int(model.config.vision_config.hidden_size)
    head_dim = hidden_size // int(model.config.vision_config.num_attention_heads)
    key = "_".join(
        (
            "encoder_postln_promptfa",
            f"b{batch_size}",
            f"s{sequence_length}",
            f"dtype{cache_key_part(dtype)}",
            f"layout{cache_key_part(get_vision_prompt_fa_layout())}",
            f"sparse{get_vision_prompt_fa_mask_sparse_mode()}",
            f"mode{cache_key_part(TORCHAIR_EXECUTION_MODE)}",
            f"model{short_file_hash(model_dir / 'config.json')}",
            f"torch{cache_key_part(torch.__version__)}",
            f"torchnpu{torch_npu_version_label(device)}",
            f"torchair{torchair_version_label(device)}",
            f"src{vision_source_hash()}",
        )
    )
    cache_dir = cache_root.expanduser().resolve() / key
    cache_existed = cache_dir.exists() and any(cache_dir.iterdir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    module = VisionPrefillStage(
        model,
        attention_impl="prompt_flash_attention",
    ).eval()
    # The three scout shapes have distinct S values, giving Dynamo a distinct
    # code object for every static graph through the existing helper.
    entrypoint = unique_bucket_forward(module, sequence_length)
    synchronize(device)
    wrapper_started = time.perf_counter()
    compiled = torchair.inference.cache_compile(
        entrypoint,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
    )
    synchronize(device)
    wrapper_s = time.perf_counter() - wrapper_started
    warm_prefix = torch.zeros(
        (batch_size, sequence_length, hidden_size),
        device=device,
        dtype=dtype,
    )
    warm_cos = torch.ones(
        (batch_size, sequence_length, head_dim),
        device=device,
        dtype=torch.float32,
    )
    warm_sin = torch.zeros_like(warm_cos)
    warm_mask = torch.zeros(
        (batch_size, 1, sequence_length, sequence_length),
        device=device,
        dtype=torch.bool,
    )
    synchronize(device)
    first_call_started = time.perf_counter()
    warm_output = compiled(warm_prefix, warm_cos, warm_sin, warm_mask)
    synchronize(device)
    first_call_s = time.perf_counter() - first_call_started
    if tuple(warm_output.shape[:2]) != (batch_size, sequence_length):
        raise RuntimeError(
            "compiled PromptFA graph returned the wrong static shape: "
            f"expected {(batch_size, sequence_length)}, got {tuple(warm_output.shape)}"
        )
    del warm_output, warm_prefix, warm_cos, warm_sin, warm_mask
    return compiled, {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "cache_dir": str(cache_dir),
        "cache_existed_before_run": cache_existed,
        "compile_wrapper_s": wrapper_s,
        "compile_first_call_s": first_call_s,
    }


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    import torch_npu

    device = torch.device("npu:0")
    if not torch.npu.is_available():
        raise RuntimeError("batched-packed vision lab requires an NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    dtype = torch.float16
    _, corpora = _load_corpora(args.corpus, args.variant)
    model_dir = args.model.expanduser().resolve()
    synchronize(device)
    setup_started = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=dtype,
        device=device,
    )
    eager_runtime = VisionPrefillRuntime(
        model,
        backend="raw_eager",
        buckets=(32,),
        cache_root=REPO_ROOT / ".runtime_cache/09_vision_batching_eager_unused",
        device=device,
        dtype=dtype,
        model_dir=model_dir,
        attention_impl="prompt_flash_attention",
        padding="none",
    )
    synchronize(device)
    setup_s = time.perf_counter() - setup_started
    compiled: dict[tuple[int, int], Callable[..., torch.Tensor]] = {}
    compile_metadata: dict[str, Any] = {}
    for batch_size, sequence_length in args.shape:
        print(
            f"load_or_compile=b{batch_size}_s{sequence_length} "
            f"physical_tokens={batch_size * sequence_length}",
            flush=True,
        )
        run_tower, metadata = _compiled_shape(
            model=model,
            batch_size=batch_size,
            sequence_length=sequence_length,
            cache_root=args.cache_dir,
            model_dir=model_dir,
            dtype=dtype,
            device=device,
        )
        compiled[(batch_size, sequence_length)] = run_tower
        compile_metadata[f"b{batch_size}_s{sequence_length}"] = metadata
        print(
            f"compiled=b{batch_size}_s{sequence_length} "
            f"cache_was_warm={metadata['cache_existed_before_run']} "
            f"first_call_s={metadata['compile_first_call_s']:.3f}",
            flush=True,
        )
    results: dict[str, list[dict[str, Any]]] = {}
    passthrough_cache: dict[tuple[str, int], float] = {}
    for corpus_name, items in corpora.items():
        results[corpus_name] = []
        for batch_size, sequence_length in args.shape:
            print(
                f"benchmark={corpus_name}:b{batch_size}_s{sequence_length}",
                flush=True,
            )
            try:
                results[corpus_name].append(
                    _run_shape(
                        corpus_name=corpus_name,
                        items=items,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        model=model,
                        run_tower=compiled[(batch_size, sequence_length)],
                        eager_runtime=eager_runtime,
                        seed=args.seed,
                        dtype=dtype,
                        device=device,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        passthrough_cache=passthrough_cache,
                    )
                )
            except Exception as exc:
                results[corpus_name].append(
                    {
                        "name": f"b{batch_size}_s{sequence_length}",
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "supported": False,
                        "failure": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc().splitlines()[-20:],
                        },
                    }
                )
    payload = {
        "schema_version": 1,
        "created_at_unix_s": time.time(),
        "execution": "torchair_static",
        "attention": "prompt_flash_attention",
        "packing_policy": "first_fit_decreasing_within_each_batch_row",
        "corpus": str(args.corpus.expanduser().resolve()),
        "variant": args.variant,
        "shapes": [list(shape) for shape in args.shape],
        "warmup": args.warmup,
        "repeats": args.repeats,
        "setup_s": setup_s,
        "compile_metadata": compile_metadata,
        "new_graphs_compiled": sum(
            not row["cache_existed_before_run"] for row in compile_metadata.values()
        ),
        "environment": _environment(device),
        "results": results,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output={output}")
    print("corpus | shape | supported | raw_tok_s | effective_tok_s | useful | tower_s")
    for corpus_name, rows in results.items():
        for row in rows:
            metrics = row.get("target_batch_metrics", {})
            print(
                " | ".join(
                    str(value)
                    for value in (
                        corpus_name,
                        row["name"],
                        row["supported"],
                        metrics.get("raw_physical_tokens_per_s"),
                        metrics.get("effective_real_tokens_per_s"),
                        metrics.get("real_token_fraction"),
                        metrics.get("tower_s"),
                    )
                )
            )


if __name__ == "__main__":
    main()
