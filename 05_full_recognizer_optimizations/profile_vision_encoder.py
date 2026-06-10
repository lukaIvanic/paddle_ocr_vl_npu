#!/usr/bin/env python3
"""Profile the native-resolution vision encoder with torch_npu.profiler.

This script intentionally profiles only the ViT encoder stack. Crop
preprocessing, device transfer, patch/position embeddings, post layernorm, the
adaptive MLP projector, text prefill, and decode stay outside the profiler
window so the resulting CANN CSVs are attributable to the current vision
encoder implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from bench_stage_timing import build_cohort_inputs, load_manifest, select_manifest_entries
from local_modeling_paddleocr_vl import (
    VISION_ATTENTION_CHOICES,
    VISION_ATTENTION_ENV,
    VISION_PROMPT_FA_LAYOUT_CHOICES,
    VISION_PROMPT_FA_LAYOUT_ENV,
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
    get_vision_attention_impl,
    get_vision_prompt_fa_layout,
)
from probe_static_compile import maybe_sync
from run_local_recognition import (
    NPU_JIT_COMPILE_CHOICES,
    configure_npu_jit_compile,
    load_preprocessor_config,
    parse_dtype,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_METRIC_CHOICES = ("pipe", "memory", "l2", "memory_access")
DEFAULT_CROP_ID = "hotswap_002_code_txt_p1474_11"
DEFAULT_VISION_ATTENTION = os.environ.get(VISION_ATTENTION_ENV, "prompt_flash_attention").strip() or "prompt_flash_attention"
DEFAULT_VISION_PROMPT_FA_LAYOUT = os.environ.get(VISION_PROMPT_FA_LAYOUT_ENV, "bnsd").strip().lower() or "bnsd"


def npu_profiler_config(metric: str):
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
    )


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def resolve_repo_path(path: Path) -> Path:
    path = path.expanduser()
    if path.exists():
        return path
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate
    return path


def default_profile_run_dir(root: Path, *, crop_id: str, metric: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_crop_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in crop_id)
    return root.expanduser().resolve() / f"vision_encoder_{timestamp}_{safe_crop_id}_{metric}"


@torch.inference_mode()
def prepare_encoder_inputs(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    del input_ids, attention_mask
    pixel_values = pixel_values.to(device=device, dtype=model.visual.dtype).unsqueeze(0)
    cu_seqlens = F.pad(
        torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2],
            image_grid_thw[:, 0],
        ).cumsum(dim=0, dtype=torch.int32),
        (1, 0),
        value=0,
    )
    vision_model = model.visual.vision_model
    hidden_states = vision_model.embeddings(pixel_values, image_grid_thw=image_grid_thw)
    return {
        "hidden_states": hidden_states,
        "cu_seqlens": cu_seqlens,
        "image_grid_thw": image_grid_thw,
    }


@torch.inference_mode()
def run_vision_encoder(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    encoder_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    vision_model = model.visual.vision_model
    return vision_model.encoder(
        encoder_inputs["hidden_states"],
        cu_seqlens=encoder_inputs["cu_seqlens"],
        image_grid_thw=encoder_inputs["image_grid_thw"],
    )


def timed(device: torch.device, fn):
    maybe_sync(device)
    start = time.perf_counter()
    result = fn()
    maybe_sync(device)
    return result, time.perf_counter() - start


def diff_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float | bool]:
    lhs_cpu = lhs.detach().to(dtype=torch.float32).cpu()
    rhs_cpu = rhs.detach().to(dtype=torch.float32).cpu()
    diff = (lhs_cpu - rhs_cpu).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "allclose_atol_2e_2_rtol_2e_2": bool(torch.allclose(lhs_cpu, rhs_cpu, atol=2e-2, rtol=2e-2)),
        "allclose_atol_5e_2_rtol_5e_2": bool(torch.allclose(lhs_cpu, rhs_cpu, atol=5e-2, rtol=5e-2)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "crops" / "hotswap_100_manifest.json")
    parser.add_argument("--crop-id", default=DEFAULT_CROP_ID)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--profile-root", type=Path, default=Path("outputs/vision_encoder_profiles"))
    parser.add_argument("--profile-run-dir", type=Path, default=None)
    parser.add_argument("--profile-metric", default="pipe", choices=PROFILE_METRIC_CHOICES)
    parser.add_argument("--vision-attention", default=DEFAULT_VISION_ATTENTION, choices=VISION_ATTENTION_CHOICES)
    parser.add_argument("--vision-prompt-fa-layout", default=DEFAULT_VISION_PROMPT_FA_LAYOUT, choices=VISION_PROMPT_FA_LAYOUT_CHOICES)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--profile-iters", type=int, default=1)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type != "npu":
        raise ValueError("vision encoder profiling requires --device npu:0")
    if int(args.warmup_iters) < 0:
        raise ValueError("--warmup-iters must be non-negative")
    if int(args.profile_iters) <= 0:
        raise ValueError("--profile-iters must be positive")

    import torch_npu.profiler as npu_prof

    os.environ[VISION_ATTENTION_ENV] = str(args.vision_attention)
    os.environ[VISION_PROMPT_FA_LAYOUT_ENV] = str(args.vision_prompt_fa_layout)
    configure_npu_jit_compile(args.npu_jit_compile, device)
    dtype = parse_dtype(args.dtype, device)
    model_dir = _resolve_model_dir(args.model)
    pre_cfg = load_preprocessor_config(model_dir)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    manifest = load_manifest(args.manifest)
    entries = select_manifest_entries(manifest, num_items=1, crop_ids=[str(args.crop_id)])
    cohort = build_cohort_inputs(
        entries=entries,
        manifest_path=args.manifest,
        tokenizer=tokenizer,
        pre_cfg=pre_cfg,
        prompt_override=args.prompt,
    )
    item = cohort[0]

    model_load_start = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device=device)
    maybe_sync(device)
    model_load_s = time.perf_counter() - model_load_start

    moved = (
        item.input_ids.to(device),
        item.attention_mask.to(device),
        item.pixel_values.to(device=device, dtype=model.visual.dtype),
        item.image_grid_thw,
    )
    input_ids, attention_mask, pixel_values, image_grid_thw = moved

    manual_reference = None
    selected_reference = None
    validation: dict[str, Any] = {
        "reference": "manual_vision_encoder",
        "tested": str(args.vision_attention),
        "prompt_fa_layout": str(args.vision_prompt_fa_layout),
        "enabled": bool(str(args.vision_attention) != "manual"),
    }
    if str(args.vision_attention) != "manual":
        os.environ[VISION_ATTENTION_ENV] = "manual"
        ref_inputs = prepare_encoder_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            device=device,
        )
        manual_reference, manual_s = timed(device, lambda: run_vision_encoder(model=model, encoder_inputs=ref_inputs))
        os.environ[VISION_ATTENTION_ENV] = str(args.vision_attention)
        os.environ[VISION_PROMPT_FA_LAYOUT_ENV] = str(args.vision_prompt_fa_layout)
        selected_inputs = prepare_encoder_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            device=device,
        )
        selected_reference, selected_s = timed(device, lambda: run_vision_encoder(model=model, encoder_inputs=selected_inputs))
        validation.update(
            {
                "manual_time_s": float(manual_s),
                "selected_time_s": float(selected_s),
                "manual_shape": [int(value) for value in manual_reference.shape],
                "selected_shape": [int(value) for value in selected_reference.shape],
                **diff_stats(selected_reference, manual_reference),
            }
        )

    os.environ[VISION_ATTENTION_ENV] = str(args.vision_attention)
    os.environ[VISION_PROMPT_FA_LAYOUT_ENV] = str(args.vision_prompt_fa_layout)
    warmup_times_s = []
    warmup_output_shape: list[int] | None = None
    for _ in range(int(args.warmup_iters)):
        os.environ[VISION_ATTENTION_ENV] = str(args.vision_attention)
        os.environ[VISION_PROMPT_FA_LAYOUT_ENV] = str(args.vision_prompt_fa_layout)
        warm_inputs = prepare_encoder_inputs(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            device=device,
        )
        warm_output, warmup_s = timed(device, lambda: run_vision_encoder(model=model, encoder_inputs=warm_inputs))
        warmup_times_s.append(float(warmup_s))
        warmup_output_shape = [int(value) for value in warm_output.shape]

    profile_run_dir = (
        args.profile_run_dir.expanduser().resolve()
        if args.profile_run_dir is not None
        else default_profile_run_dir(args.profile_root, crop_id=str(args.crop_id), metric=str(args.profile_metric))
    )
    shutil.rmtree(profile_run_dir, ignore_errors=True)
    profile_run_dir.mkdir(parents=True, exist_ok=True)

    prof_inputs = prepare_encoder_inputs(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        device=device,
    )
    schedule = npu_prof.schedule(wait=0, warmup=0, active=1, repeat=1)
    outputs = []
    maybe_sync(device)
    profile_start = time.perf_counter()
    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        schedule=schedule,
        experimental_config=npu_profiler_config(str(args.profile_metric)),
        on_trace_ready=npu_prof.tensorboard_trace_handler(str(profile_run_dir), analyse_flag=True),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    ) as profiler:
        with torch.profiler.record_function("paddle_ocr_vl.vision_encoder_profile"):
            for _ in range(int(args.profile_iters)):
                os.environ[VISION_ATTENTION_ENV] = str(args.vision_attention)
                os.environ[VISION_PROMPT_FA_LAYOUT_ENV] = str(args.vision_prompt_fa_layout)
                outputs.append(run_vision_encoder(model=model, encoder_inputs=prof_inputs))
        maybe_sync(device)
        profiler.step()
    profile_wall_s = time.perf_counter() - profile_start
    maybe_sync(device)

    output_shape = [int(value) for value in outputs[-1].shape]
    summary = {
        "profile_kind": "vision_encoder_only",
        "profile_dir": str(profile_run_dir),
        "profile_metric": str(args.profile_metric),
        "vision_attention": get_vision_attention_impl(),
        "vision_prompt_fa_layout": get_vision_prompt_fa_layout(),
        "validation": validation,
        "with_stack": True,
        "record_shapes": True,
        "profile_memory": False,
        "model": str(model_dir),
        "device": str(device),
        "dtype": str(dtype),
        "npu_jit_compile": str(args.npu_jit_compile),
        "crop_id": str(item.entry.get("id")),
        "crop_file": str(item.crop_path),
        "category_type": item.entry.get("category_type"),
        "crop_size": item.entry.get("crop_size"),
        "input_tokens": int(input_ids.shape[1]),
        "image_grid_thw": [int(value) for value in image_grid_thw.reshape(-1).detach().cpu().tolist()],
        "vision_tokens": int(prof_inputs["hidden_states"].shape[0]),
        "hidden_size": int(prof_inputs["hidden_states"].shape[-1]),
        "warmup_iters": int(args.warmup_iters),
        "warmup_times_s": warmup_times_s,
        "warmup_output_shape": warmup_output_shape,
        "profile_iters": int(args.profile_iters),
        "profile_wall_s": float(profile_wall_s),
        "output_shape": output_shape,
        "model_load_s": float(model_load_s),
        "profile_scope": "vision_model.encoder only; embeddings and device transfer are outside profiler window",
    }
    summary_path = profile_run_dir / "vision_profile_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    print(json.dumps({"profile_dir": str(profile_run_dir), "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
