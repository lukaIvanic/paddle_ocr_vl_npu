#!/usr/bin/env python3
"""Benchmark PaddleOCR-VL's vision-transformer-only boundary on real small crops.

The benchmark deliberately prepares these inputs before the timed region:

* crop decoding and PaddleOCR-VL image preprocessing
* patch embedding and interpolated absolute position embeddings
* vision RoPE tensors and the fixed-bucket padding mask

The measured boundary is only the complete vision encoder stack plus its final
LayerNorm over ``[B, S_physical, hidden]``.  Each process benchmarks one
attention implementation, one eager/compiled backend, and one fixed shape.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from tokenizers import Tokenizer

from validate_static_visual_batched_encoder import (
    BatchedStaticVisualEncoderModule,
    build_prefix_batch,
    compile_encoder_forward,
)
from vision_prefill_bench import (
    DEFAULT_VISION_TORCHAIR_CACHE_DIR,
    STATIC_VISUAL_LN_IMPL_CHOICES,
    STATIC_VISUAL_LN_LINEAR_MODE_CHOICES,
    VISION_COMPILE_BACKEND_CHOICES,
    LayoutCrop,
    add_common_args,
    add_torchair_diagnostic_args,
    apply_runtime_env,
    build_prefill_inputs,
    clean_json,
    diff_stats,
    input_row,
    json_default,
    load_model_for_args,
    load_preprocessor_config,
    maybe_sync,
    sha256_file,
    stats,
    vision_tokens,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CROP_RUN_JSON = (
    REPO_ROOT
    / "tmp/08_offline_e2e_b1/five_pages_uniform/promptfa_pair/manual_default/run.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, timing_default="standard")
    parser.set_defaults(model=os.environ.get("MODEL", "/workspace/models/PaddleOCR-VL-1.6"))
    parser.add_argument(
        "--crop-run-json",
        default=str(DEFAULT_CROP_RUN_JSON),
        help="Experiment08 run.json whose real layout boxes define the crop population.",
    )
    parser.add_argument("--preprocessor-min-pixels", type=int, default=112896)
    parser.add_argument("--fixed-physical-seq-len", type=int, required=True)
    parser.add_argument(
        "--bucket-min-exclusive",
        type=int,
        default=0,
        help="Only select crops with real vision tokens greater than this lower bucket edge.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-forwards", type=int, default=5)
    parser.add_argument("--measurement-blocks", type=int, default=3)
    parser.add_argument("--forwards-per-block", type=int, default=20)
    parser.add_argument("--vision-compile-backend", default="none", choices=VISION_COMPILE_BACKEND_CHOICES)
    parser.add_argument("--vision-use-torchair-cache-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vision-torchair-cache-dir", default=str(DEFAULT_VISION_TORCHAIR_CACHE_DIR))
    parser.add_argument("--static-visual-ln-impl", default="module", choices=STATIC_VISUAL_LN_IMPL_CHOICES)
    parser.add_argument(
        "--static-visual-ln-linear-mode",
        default="normal",
        choices=STATIC_VISUAL_LN_LINEAR_MODE_CHOICES,
    )
    parser.add_argument(
        "--static-visual-promptfa-pad-head-dim-to",
        type=int,
        default=0,
        help="0 preserves PaddleOCR-VL's native D=72 PromptFA call; use 80 only as an explicit workaround trial.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "tmp/07_vision_prefill_optimization/small_visual_encoder_case.json",
    )
    add_torchair_diagnostic_args(parser)
    return parser.parse_args()


def resolve_page_image(raw_path: str, *, dataset_dir: str | None) -> Path:
    original = Path(str(raw_path)).expanduser()
    candidates = [original]
    if dataset_dir:
        candidates.append(Path(dataset_dir).expanduser() / "images" / original.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"run.json page image is unavailable: {raw_path}; tried "
        + ", ".join(str(path) for path in candidates)
    )


def crop_box(region: dict[str, Any], *, width: int, height: int) -> tuple[int, int, int, int]:
    box = region.get("box") or {}
    if not isinstance(box, dict):
        raise ValueError(f"recognized region box must be an object, got {type(box).__name__}")
    left = max(0, int(math.floor(float(box["x0"]))))
    top = max(0, int(math.floor(float(box["y0"]))))
    right = min(int(width), int(math.ceil(float(box["x1"]))))
    bottom = min(int(height), int(math.ceil(float(box["y1"]))))
    if right <= left or bottom <= top:
        raise ValueError(f"invalid recognized region box after clamping: {(left, top, right, bottom)}")
    return left, top, right, bottom


def build_real_crop_inputs(
    *,
    run_json_path: Path,
    model_dir: Path,
    tokenizer: Tokenizer,
    dataset_dir: str | None,
    min_pixels: int,
) -> tuple[list[Any], dict[str, Any]]:
    run_json_path = run_json_path.expanduser().resolve()
    payload = json.loads(run_json_path.read_text(encoding="utf-8"))
    crops: list[LayoutCrop] = []
    page_rows: list[dict[str, Any]] = []
    for page_index, page in enumerate(payload.get("pages") or []):
        image_path = resolve_page_image(str(page["image_path"]), dataset_dir=dataset_dir)
        page_count = 0
        with Image.open(image_path).convert("RGB") as image:
            width, height = image.size
            for region_index, region in enumerate(page.get("recognized_regions") or []):
                bbox = crop_box(region, width=width, height=height)
                crop = image.crop(bbox).copy()
                request_id = str(region.get("request_id") or f"page{page_index:04d}_region{region_index:04d}")
                label = str(region.get("label") or "unknown")
                entry = {
                    "id": request_id,
                    "source_image": str(image_path),
                    "image_rel": image_path.name,
                    "page_index": int(page_index),
                    "dataset_index": int(page_index),
                    "layout_label": label,
                    "category_type": label,
                    "bbox_xyxy": [int(value) for value in bbox],
                    "crop_size": [int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])],
                    "suggested_prompt": str(region.get("prompt") or "OCR:"),
                    "ground_truth": str(region.get("text") or ""),
                    "ground_truth_source": "experiment08_previous_output",
                }
                crops.append(LayoutCrop(entry=entry, image=crop))
                page_count += 1
        page_rows.append(
            {
                "page_index": int(page_index),
                "page_id": str(page.get("page_id") or ""),
                "image_path": str(image_path),
                "recognized_region_count": int(page_count),
            }
        )
    if not crops:
        raise ValueError(f"no recognized regions found in {run_json_path}")
    preprocessor = load_preprocessor_config(model_dir)
    original_min_pixels = int(preprocessor["min_pixels"])
    preprocessor["min_pixels"] = int(min_pixels)
    inputs, input_summary = build_prefill_inputs(
        crops=crops,
        tokenizer=tokenizer,
        pre_cfg=preprocessor,
        prompt_override=None,
    )
    return inputs, {
        "run_json": str(run_json_path),
        "run_json_sha256": sha256_file(run_json_path),
        "page_count": int(len(page_rows)),
        "crop_count": int(len(crops)),
        "pages": page_rows,
        "preprocessor_original_min_pixels": int(original_min_pixels),
        "preprocessor_effective_min_pixels": int(preprocessor["min_pixels"]),
        "preprocessor": clean_json(preprocessor),
        "input_build": clean_json(input_summary),
        "vision_tokens": stats([float(vision_tokens(item)) for item in inputs]),
    }


def select_bucket_batch(
    inputs: list[Any],
    *,
    lower_exclusive: int,
    upper_inclusive: int,
    batch_size: int,
) -> tuple[list[Any], dict[str, Any]]:
    eligible = [
        item
        for item in inputs
        if int(lower_exclusive) < int(vision_tokens(item)) <= int(upper_inclusive)
    ]
    eligible.sort(key=lambda item: (-int(vision_tokens(item)), str(item.entry.get("id") or "")))
    if len(eligible) < int(batch_size):
        available = sorted(int(vision_tokens(item)) for item in inputs)
        raise ValueError(
            f"bucket ({lower_exclusive}, {upper_inclusive}] has {len(eligible)} crops, needs B={batch_size}; "
            f"population token range={available[0] if available else None}..{available[-1] if available else None}"
        )
    selected = eligible[: int(batch_size)]
    effective = int(sum(vision_tokens(item) for item in selected))
    physical = int(batch_size) * int(upper_inclusive)
    return selected, {
        "strategy": "highest_fill_real_crops_within_bucket",
        "lower_exclusive": int(lower_exclusive),
        "upper_inclusive": int(upper_inclusive),
        "population_count": int(len(inputs)),
        "eligible_count": int(len(eligible)),
        "batch_size": int(batch_size),
        "selected_effective_tokens": int(effective),
        "selected_physical_tokens": int(physical),
        "selected_useful_token_fraction": float(effective / physical),
        "eligible_vision_tokens": stats([float(vision_tokens(item)) for item in eligible]),
        "selected": [
            {
                "id": str(item.entry.get("id") or ""),
                "vision_tokens": int(vision_tokens(item)),
                "layout_label": str(item.entry.get("layout_label") or ""),
                "crop_size": clean_json(item.entry.get("crop_size") or []),
                "image_grid_thw": [int(value) for value in item.image_grid_thw.flatten().tolist()],
            }
            for item in selected
        ],
    }


def tensor_nonfinite_count(tensor: torch.Tensor) -> int:
    return int((~torch.isfinite(tensor.float())).sum().item())


def real_row_diffs(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    real_lengths: list[int],
) -> list[dict[str, Any]]:
    return [
        diff_stats(candidate[row_idx, :real_len].cpu(), reference[row_idx, :real_len].cpu())
        for row_idx, real_len in enumerate(real_lengths)
    ]


def aggregate_real_diff(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": int(len(rows)),
        "max_abs_diff": max(
            (float(row["max_abs_diff"]) for row in rows if row.get("max_abs_diff") is not None),
            default=None,
        ),
        "mean_abs_diff": stats(
            [float(row["mean_abs_diff"]) for row in rows if row.get("mean_abs_diff") is not None]
        ),
        "allclose_5e_2_count": int(sum(bool(row.get("allclose_atol_5e_2_rtol_5e_2")) for row in rows)),
        "allclose_1e_1_count": int(sum(bool(row.get("allclose_atol_1e_1_rtol_1e_1")) for row in rows)),
        "rows": clean_json(rows),
    }


def timed_forward_blocks(
    forward: Callable[..., torch.Tensor],
    forward_args: tuple[torch.Tensor, ...],
    *,
    device: torch.device,
    warmup_forwards: int,
    measurement_blocks: int,
    forwards_per_block: int,
    effective_tokens_per_forward: int,
    physical_tokens_per_forward: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    warmup_start = time.perf_counter()
    output: torch.Tensor | None = None
    for _ in range(int(warmup_forwards)):
        output = forward(*forward_args)
    maybe_sync(device)
    warmup_s = float(time.perf_counter() - warmup_start)

    blocks: list[dict[str, Any]] = []
    for block_idx in range(int(measurement_blocks)):
        maybe_sync(device)
        start = time.perf_counter()
        for _ in range(int(forwards_per_block)):
            output = forward(*forward_args)
        maybe_sync(device)
        elapsed = float(time.perf_counter() - start)
        blocks.append(
            {
                "block_index": int(block_idx),
                "forwards": int(forwards_per_block),
                "elapsed_s": elapsed,
                "mean_forward_s": float(elapsed / int(forwards_per_block)),
                "effective_tokens_per_s": float(
                    int(effective_tokens_per_forward) * int(forwards_per_block) / elapsed
                ),
                "physical_tokens_per_s": float(
                    int(physical_tokens_per_forward) * int(forwards_per_block) / elapsed
                ),
            }
        )
    if output is None:
        raise RuntimeError("benchmark produced no output")
    total_s = float(sum(row["elapsed_s"] for row in blocks))
    total_forwards = int(measurement_blocks) * int(forwards_per_block)
    return output, {
        "warmup_forwards": int(warmup_forwards),
        "warmup_s": warmup_s,
        "measurement_blocks": int(measurement_blocks),
        "forwards_per_block": int(forwards_per_block),
        "measured_forwards": int(total_forwards),
        "measured_s": total_s,
        "effective_tokens_per_forward": int(effective_tokens_per_forward),
        "physical_tokens_per_forward": int(physical_tokens_per_forward),
        "effective_tokens_per_s": float(effective_tokens_per_forward * total_forwards / total_s),
        "physical_tokens_per_s": float(physical_tokens_per_forward * total_forwards / total_s),
        "mean_forward_s": float(total_s / total_forwards),
        "block_mean_forward_s": stats([float(row["mean_forward_s"]) for row in blocks]),
        "block_effective_tokens_per_s": stats([float(row["effective_tokens_per_s"]) for row in blocks]),
        "block_physical_tokens_per_s": stats([float(row["physical_tokens_per_s"]) for row in blocks]),
        "blocks": blocks,
    }


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(device),
        "git_commit": git_commit(),
        "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    }
    if device.type == "npu":
        import torch_npu

        result["torch_npu"] = str(torch_npu.__version__)
        try:
            result["device_name"] = str(torch_npu.npu.get_device_name(device))
        except Exception as exc:  # pragma: no cover - runtime-version-specific metadata only
            result["device_name_error"] = f"{type(exc).__name__}: {exc}"
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    apply_runtime_env(args)
    if int(args.preprocessor_min_pixels) <= 0:
        raise ValueError("--preprocessor-min-pixels must be positive")
    if int(args.fixed_physical_seq_len) <= 0:
        raise ValueError("--fixed-physical-seq-len must be positive")
    if int(args.bucket_min_exclusive) < 0 or int(args.bucket_min_exclusive) >= int(args.fixed_physical_seq_len):
        raise ValueError("--bucket-min-exclusive must satisfy 0 <= lower < fixed S")
    for name in ("batch_size", "measurement_blocks", "forwards_per_block"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.warmup_forwards) < 0:
        raise ValueError("--warmup-forwards must be non-negative")

    model, model_dir, device, dtype = load_model_for_args(args)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    inputs, source_summary = build_real_crop_inputs(
        run_json_path=Path(args.crop_run_json),
        model_dir=model_dir,
        tokenizer=tokenizer,
        dataset_dir=args.dataset_dir,
        min_pixels=int(args.preprocessor_min_pixels),
    )
    selected, selection = select_bucket_batch(
        inputs,
        lower_exclusive=int(args.bucket_min_exclusive),
        upper_inclusive=int(args.fixed_physical_seq_len),
        batch_size=int(args.batch_size),
    )
    merge_size = int(source_summary["preprocessor"]["merge_size"])
    prefix, rope_cos, rope_sin, attention_mask, prefix_meta = build_prefix_batch(
        model=model,
        batch_items=selected,
        device=device,
        fixed_physical_seq_len=int(args.fixed_physical_seq_len),
        ln_impl=str(args.static_visual_ln_impl),
        ln_linear_mode=str(args.static_visual_ln_linear_mode),
        promptfa_pad_head_dim_to=int(args.static_visual_promptfa_pad_head_dim_to),
        debug_no_padding=False,
        debug_min_pad_tokens=0,
        debug_pad_to_multiple=0,
    )
    forward_args = (prefix, rope_cos, rope_sin, attention_mask)
    real_lengths = [int(vision_tokens(item)) for item in selected]

    candidate_module = BatchedStaticVisualEncoderModule(
        model,
        fixed_physical_seq_len=int(args.fixed_physical_seq_len),
        ln_impl=str(args.static_visual_ln_impl),
        ln_linear_mode=str(args.static_visual_ln_linear_mode),
        promptfa_pad_head_dim_to=int(args.static_visual_promptfa_pad_head_dim_to),
    ).eval()
    maybe_sync(device)
    same_path_eager = candidate_module(*forward_args)
    maybe_sync(device)

    native_promptfa_eager: torch.Tensor | None = None
    if str(args.vision_attention) == "prompt_flash_attention" and int(args.static_visual_promptfa_pad_head_dim_to) > 0:
        native_module = BatchedStaticVisualEncoderModule(
            model,
            fixed_physical_seq_len=int(args.fixed_physical_seq_len),
            ln_impl=str(args.static_visual_ln_impl),
            ln_linear_mode=str(args.static_visual_ln_linear_mode),
            promptfa_pad_head_dim_to=0,
        ).eval()
        native_promptfa_eager = native_module(*forward_args)
        maybe_sync(device)

    candidate_forward, compile_meta = compile_encoder_forward(
        candidate_module,
        backend_name=str(args.vision_compile_backend),
        device=device,
        use_cache_compile=bool(args.vision_use_torchair_cache_compile),
        cache_root=Path(args.vision_torchair_cache_dir).expanduser().resolve(),
        batch_size=int(args.batch_size),
        fixed_physical_seq_len=int(args.fixed_physical_seq_len),
        dtype=dtype,
        torchair_mode=str(args.torchair_mode),
        torchair_run_eagerly=bool(args.torchair_run_eagerly),
        torchair_graph_dump_type=str(args.torchair_graph_dump_type),
        torchair_graph_dump_dir=str(args.torchair_graph_dump_dir or ""),
        torchair_msit_dump_kind=str(args.torchair_msit_dump_kind),
        torchair_msit_dump_dir=str(args.torchair_msit_dump_dir or ""),
        torchair_msit_dump_mode=str(args.torchair_msit_dump_mode),
        torchair_msit_dump_token=str(args.torchair_msit_dump_token or ""),
        torchair_msit_dump_layer=str(args.torchair_msit_dump_layer or ""),
        torchair_msit_fusion_switch_file=str(args.torchair_msit_fusion_switch_file or ""),
        promptfa_mask_sparse_mode=int(args.vision_prompt_fa_mask_sparse_mode),
    )
    maybe_sync(device)
    first_start = time.perf_counter()
    first_output = candidate_forward(*forward_args)
    maybe_sync(device)
    first_call_s = float(time.perf_counter() - first_start)
    if "capture_scalar_outputs_previous" in compile_meta:
        import torch._dynamo as torch_dynamo

        torch_dynamo.config.capture_scalar_outputs = bool(compile_meta["capture_scalar_outputs_previous"])
        compile_meta["capture_scalar_outputs_restored_after_first_call"] = True

    effective_tokens = int(sum(real_lengths))
    physical_tokens = int(args.batch_size) * int(args.fixed_physical_seq_len)
    final_output, timing = timed_forward_blocks(
        candidate_forward,
        forward_args,
        device=device,
        warmup_forwards=int(args.warmup_forwards),
        measurement_blocks=int(args.measurement_blocks),
        forwards_per_block=int(args.forwards_per_block),
        effective_tokens_per_forward=effective_tokens,
        physical_tokens_per_forward=physical_tokens,
    )

    first_real_rows = real_row_diffs(first_output, same_path_eager, real_lengths)
    final_real_rows = real_row_diffs(final_output, same_path_eager, real_lengths)
    same_path_correctness = {
        "first_call_physical": diff_stats(first_output.cpu(), same_path_eager.cpu()),
        "first_call_real_rows": aggregate_real_diff(first_real_rows),
        "final_call_physical": diff_stats(final_output.cpu(), same_path_eager.cpu()),
        "final_call_real_rows": aggregate_real_diff(final_real_rows),
    }
    native_head_correctness: dict[str, Any] | None = None
    if native_promptfa_eager is not None:
        native_head_correctness = {
            "same_path_eager_vs_native_head_physical": diff_stats(
                same_path_eager.cpu(), native_promptfa_eager.cpu()
            ),
            "same_path_eager_vs_native_head_real_rows": aggregate_real_diff(
                real_row_diffs(same_path_eager, native_promptfa_eager, real_lengths)
            ),
        }

    final_nonfinite = tensor_nonfinite_count(final_output)
    same_path_real_pass = bool(
        same_path_correctness["final_call_real_rows"]["allclose_1e_1_count"] == int(args.batch_size)
    )
    native_head_real_pass = bool(
        native_head_correctness is None
        or native_head_correctness["same_path_eager_vs_native_head_real_rows"]["allclose_1e_1_count"]
        == int(args.batch_size)
    )
    correctness_passed = bool(same_path_real_pass and native_head_real_pass and final_nonfinite == 0)

    output = {
        "schema_version": 1,
        "experiment": "07_vision_prefill_optimization",
        "kind": "small_visual_encoder_case",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_metadata(device),
        "measurement_contract": {
            "timed_boundary": "vision_encoder_layers_plus_post_layernorm_only",
            "inputs_already_on_device": True,
            "prefix_build_timed": False,
            "patch_embedding_timed": False,
            "absolute_position_and_rope_build_timed": False,
            "projector_text_prefill_decode_timed": False,
            "timing_method": "one_device_sync_before_and_after_each_repeated_forward_block",
            "profiler_enabled": False,
        },
        "case": {
            "model": str(model_dir),
            "device": str(device),
            "dtype": str(dtype),
            "attention": str(args.vision_attention),
            "promptfa_layout": str(args.vision_prompt_fa_layout),
            "promptfa_mask_sparse_mode": int(args.vision_prompt_fa_mask_sparse_mode),
            "compile_backend": str(args.vision_compile_backend),
            "compile_api": compile_meta.get("compile_api"),
            "uses_torchair_cache_compile": bool(compile_meta.get("uses_torchair_cache_compile", False)),
            "batch_size": int(args.batch_size),
            "fixed_physical_seq_len": int(args.fixed_physical_seq_len),
            "bucket_min_exclusive": int(args.bucket_min_exclusive),
            "preprocessor_min_pixels": int(args.preprocessor_min_pixels),
            "ln_impl": str(args.static_visual_ln_impl),
            "ln_linear_mode": str(args.static_visual_ln_linear_mode),
            "promptfa_pad_head_dim_to": int(args.static_visual_promptfa_pad_head_dim_to),
        },
        "source": source_summary,
        "selection": selection,
        "selected_items": [input_row(item, merge_size=merge_size) for item in selected],
        "prefix": clean_json(prefix_meta),
        "compile": {
            **clean_json(compile_meta),
            "first_call_s": first_call_s,
            "first_output_nonfinite_count": tensor_nonfinite_count(first_output),
        },
        "timing": timing,
        "correctness": {
            "passed": correctness_passed,
            "gate": "all real rows allclose at atol=rtol=0.1 and final physical output has no nonfinite values",
            "same_path_real_passed": same_path_real_pass,
            "native_promptfa_head_real_passed": native_head_real_pass,
            "final_output_nonfinite_count": final_nonfinite,
            "candidate_vs_same_path_eager": same_path_correctness,
            "padded_promptfa_eager_vs_native_head": native_head_correctness,
        },
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "case": output["case"],
                "selection": {
                    "eligible_count": selection["eligible_count"],
                    "selected_effective_tokens": selection["selected_effective_tokens"],
                    "selected_physical_tokens": selection["selected_physical_tokens"],
                    "selected_useful_token_fraction": selection["selected_useful_token_fraction"],
                },
                "first_call_s": first_call_s,
                "timing": {
                    "mean_forward_s": timing["mean_forward_s"],
                    "effective_tokens_per_s": timing["effective_tokens_per_s"],
                    "physical_tokens_per_s": timing["physical_tokens_per_s"],
                },
                "correctness_passed": correctness_passed,
            },
            indent=2,
            default=json_default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
