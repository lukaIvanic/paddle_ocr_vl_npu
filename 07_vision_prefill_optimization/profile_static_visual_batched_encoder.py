#!/usr/bin/env python3
"""Profile only the compiled static visual encoder-layer batch boundary.

The profiled region is exactly:

    encoder_forward(prefix_hidden_states, rope_cos, rope_sin, attention_mask)

Prefix construction, TorchAir cache_compile first-call behavior, validation,
projector, text prefill, and decode are outside the torch_npu profiler window.
One profiler.step() is called after exactly one encoder batch forward.
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
from tokenizers import Tokenizer

from vision_prefill_bench import (
    DEFAULT_VISION_TORCHAIR_CACHE_DIR,
    STATIC_VISUAL_LN_IMPL_CHOICES,
    STATIC_VISUAL_LN_LINEAR_MODE_CHOICES,
    add_common_args,
    add_torchair_diagnostic_args,
    apply_runtime_env,
    build_inputs_from_manifest,
    clean_json,
    diff_stats,
    json_default,
    load_baseline_manifest,
    load_model_for_args,
    maybe_sync,
    resolve_dataset_dir,
    sha256_file,
    stats,
    vision_tokens,
)
from validate_static_visual_batched_encoder import (
    BatchedStaticVisualEncoderModule,
    build_prefix_batch,
    compile_encoder_forward,
)


PROFILE_METRIC_CHOICES = ("pipe", "memory", "l2", "memory_access")


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


def visual_tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    """Small shape/finite summary for visual features, not a logits top-k."""
    data = tensor.detach().float().cpu()
    finite = torch.isfinite(data)
    finite_count = int(finite.sum().item())
    nonfinite_count = int(data.numel() - finite_count)
    summary: dict[str, Any] = {
        "shape": [int(dim) for dim in data.shape],
        "dtype": str(tensor.dtype),
        "numel": int(data.numel()),
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
    }
    if finite_count == 0:
        return summary
    finite_values = data[finite]
    summary.update(
        {
            "min": float(finite_values.min().item()),
            "max": float(finite_values.max().item()),
            "abs_max": float(finite_values.abs().max().item()),
            "mean": float(finite_values.mean().item()),
            "mean_abs": float(finite_values.abs().mean().item()),
            "std": float(finite_values.std(unbiased=False).item()),
            "l2_norm": float(torch.linalg.vector_norm(finite_values).item()),
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, timing_default="standard")
    parser.add_argument("--baseline", default=str(Path(__file__).resolve().parent / "baselines" / "promptfa_fp16_eager_64"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "outputs" / "static_visual_batched_encoder_profile.json"))
    parser.add_argument("--profile-root", type=Path, default=Path(__file__).resolve().parent / "outputs" / "profiles_static_visual_batched_encoder")
    parser.add_argument("--profile-metric", default="pipe", choices=PROFILE_METRIC_CHOICES)
    parser.add_argument("--profile-warmup-steps", type=int, default=2)
    parser.add_argument("--profile-active-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--profile-batch-index", type=int, default=0)
    parser.add_argument("--vision-use-torchair-cache-compile", action="store_true", default=True)
    parser.add_argument("--vision-torchair-cache-dir", default=str(DEFAULT_VISION_TORCHAIR_CACHE_DIR))
    parser.add_argument("--static-visual-fixed-physical-seq-len", type=int, default=1024)
    parser.add_argument("--static-visual-ln-impl", default="manual_fp32", choices=STATIC_VISUAL_LN_IMPL_CHOICES)
    parser.add_argument("--static-visual-ln-linear-mode", default="grouped_qkv_mlp_fc1", choices=STATIC_VISUAL_LN_LINEAR_MODE_CHOICES)
    parser.add_argument("--static-visual-promptfa-pad-head-dim-to", type=int, default=80)
    parser.add_argument("--debug-static-visual-no-padding", action="store_true")
    parser.add_argument("--debug-static-visual-min-pad-tokens", type=int, default=0)
    parser.add_argument("--debug-static-visual-pad-to-multiple", type=int, default=0)
    parser.add_argument("--parse-profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parser-topn", type=int, default=20)
    add_torchair_diagnostic_args(parser)
    return parser.parse_args()


def make_profile_run_dir(root: Path, *, batch_size: int, fixed_s: int, metric: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return root.expanduser().resolve() / f"static_visual_encoder_{timestamp}_B{batch_size}_S{fixed_s}_{metric}"


def build_selected_batches(args: argparse.Namespace, inputs: list[Any]) -> tuple[list[list[tuple[int, Any]]], list[dict[str, Any]]]:
    fixed_s = int(args.static_visual_fixed_physical_seq_len)
    batch_size = int(args.batch_size)
    excluded_rows: list[dict[str, Any]] = []
    eligible_pairs: list[tuple[int, Any]] = []
    for manifest_index, item in enumerate(inputs):
        real_tokens = int(vision_tokens(item))
        if real_tokens > fixed_s:
            excluded_rows.append(
                {
                    "manifest_index": int(manifest_index),
                    "id": str(item.entry.get("id")),
                    "vision_tokens": int(real_tokens),
                    "reason": "real_visual_tokens_exceed_fixed_physical_seq_len",
                }
            )
        else:
            eligible_pairs.append((manifest_index, item))
    if int(args.max_items) > 0:
        eligible_pairs = eligible_pairs[: int(args.max_items)]
    batchable_count = (len(eligible_pairs) // batch_size) * batch_size
    for manifest_index, item in eligible_pairs[batchable_count:]:
        excluded_rows.append(
            {
                "manifest_index": int(manifest_index),
                "id": str(item.entry.get("id")),
                "vision_tokens": int(vision_tokens(item)),
                "reason": "not_enough_items_for_full_transformer_batch",
            }
        )
    selected_pairs = eligible_pairs[:batchable_count]
    if not selected_pairs:
        raise ValueError(f"no full batches available for batch_size={batch_size}, eligible={len(eligible_pairs)}")
    return [selected_pairs[idx : idx + batch_size] for idx in range(0, len(selected_pairs), batch_size)], excluded_rows


def run_parser(profile_dir: Path, *, topn: int) -> dict[str, Any]:
    import subprocess
    import sys

    parser_path = Path(__file__).resolve().parent / "parse_static_visual_encoder_profile.py"
    command = [
        sys.executable,
        str(parser_path),
        "--profile-dir",
        str(profile_dir),
        "--topn",
        str(topn),
        "--skip-trace",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"stdout": completed.stdout, "stderr": completed.stderr}


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    apply_runtime_env(args)
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if int(args.profile_warmup_steps) < 0:
        raise ValueError("--profile-warmup-steps must be non-negative")
    if int(args.profile_active_steps) <= 0:
        raise ValueError("--profile-active-steps must be positive")

    model, model_dir, device, dtype = load_model_for_args(args)
    if device.type != "npu":
        raise ValueError("torch_npu profiler requires --device npu:0")
    if str(args.torchair_run_eagerly).lower() == "true":
        raise ValueError("profile_static_visual_batched_encoder.py profiles GE/CANN execution, not run-eagerly FX")

    import torch_npu.profiler as npu_prof

    baseline_path = Path(args.baseline).expanduser().resolve()
    manifest = load_baseline_manifest(baseline_path)
    baseline_dir = baseline_path if baseline_path.is_dir() else baseline_path.parent
    tensor_dir = baseline_dir / str(manifest.get("tensor_dir", "tensors"))
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    dataset_dir = resolve_dataset_dir(args.dataset_dir or manifest["build_summary"]["page"]["dataset_dir"])
    inputs = build_inputs_from_manifest(manifest=manifest, model_dir=model_dir, tokenizer=tokenizer, dataset_dir=dataset_dir)
    batches, excluded_rows = build_selected_batches(args, inputs)
    if int(args.profile_batch_index) < 0 or int(args.profile_batch_index) >= len(batches):
        raise ValueError(f"--profile-batch-index {args.profile_batch_index} out of range for {len(batches)} batches")
    batch_pairs = batches[int(args.profile_batch_index)]
    batch_items = [item for _manifest_index, item in batch_pairs]
    fixed_s = int(args.static_visual_fixed_physical_seq_len)

    encoder_module = BatchedStaticVisualEncoderModule(
        model,
        fixed_physical_seq_len=fixed_s,
        ln_impl=str(args.static_visual_ln_impl),
        ln_linear_mode=str(args.static_visual_ln_linear_mode),
        promptfa_pad_head_dim_to=int(args.static_visual_promptfa_pad_head_dim_to),
    ).eval()
    encoder_forward, compile_meta = compile_encoder_forward(
        encoder_module,
        backend_name="torchair",
        device=device,
        use_cache_compile=bool(args.vision_use_torchair_cache_compile),
        cache_root=Path(args.vision_torchair_cache_dir).expanduser().resolve(),
        batch_size=int(args.batch_size),
        fixed_physical_seq_len=fixed_s,
        dtype=dtype,
        torchair_mode=str(args.torchair_mode),
        torchair_run_eagerly=False,
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
    prefix_start = time.perf_counter()
    prefix, rope_cos, rope_sin, mask, prefix_meta = build_prefix_batch(
        model=model,
        batch_items=batch_items,
        device=device,
        fixed_physical_seq_len=fixed_s,
        ln_impl=str(args.static_visual_ln_impl),
        ln_linear_mode=str(args.static_visual_ln_linear_mode),
        promptfa_pad_head_dim_to=int(args.static_visual_promptfa_pad_head_dim_to),
        debug_no_padding=bool(args.debug_static_visual_no_padding),
        debug_min_pad_tokens=int(args.debug_static_visual_min_pad_tokens),
        debug_pad_to_multiple=int(args.debug_static_visual_pad_to_multiple),
    )
    maybe_sync(device)
    prefix_build_s = time.perf_counter() - prefix_start

    maybe_sync(device)
    first_start = time.perf_counter()
    first_output = encoder_forward(prefix, rope_cos, rope_sin, mask)
    maybe_sync(device)
    compiled_first_call_s = time.perf_counter() - first_start

    warmup_times_s = []
    for _ in range(int(args.profile_warmup_steps)):
        maybe_sync(device)
        start = time.perf_counter()
        encoder_forward(prefix, rope_cos, rope_sin, mask)
        maybe_sync(device)
        warmup_times_s.append(float(time.perf_counter() - start))

    profile_dir = make_profile_run_dir(
        args.profile_root,
        batch_size=int(args.batch_size),
        fixed_s=fixed_s,
        metric=str(args.profile_metric),
    )
    shutil.rmtree(profile_dir, ignore_errors=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    active_steps = int(args.profile_active_steps)
    schedule = npu_prof.schedule(wait=0, warmup=0, active=active_steps, repeat=1)
    forward_sync_times_s: list[float] = []
    profiler_step_times_s: list[float] = []
    maybe_sync(device)
    context_start = time.perf_counter()
    active_start: float | None = None
    active_end: float | None = None
    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        schedule=schedule,
        experimental_config=npu_profiler_config(str(args.profile_metric)),
        on_trace_ready=npu_prof.tensorboard_trace_handler(str(profile_dir), analyse_flag=True),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    ) as profiler:
        active_start = time.perf_counter()
        for step in range(active_steps):
            with torch.profiler.record_function(f"paddle_ocr_vl.static_visual_batched_encoder.B{args.batch_size}.step{step}"):
                maybe_sync(device)
                forward_start = time.perf_counter()
                profiled_output = encoder_forward(prefix, rope_cos, rope_sin, mask)
                maybe_sync(device)
                forward_sync_times_s.append(float(time.perf_counter() - forward_start))
            step_start = time.perf_counter()
            profiler.step()
            profiler_step_times_s.append(float(time.perf_counter() - step_start))
        active_end = time.perf_counter()
    maybe_sync(device)
    context_wall_s = time.perf_counter() - context_start

    effective_tokens = int(sum(vision_tokens(item) for item in batch_items))
    physical_tokens = int(int(args.batch_size) * fixed_s)
    forward_sync_sum_s = float(sum(forward_sync_times_s))
    profiler_step_sum_s = float(sum(profiler_step_times_s))
    active_loop_wall_s = float((active_end or context_start) - (active_start or context_start))
    context_non_active_s = float(max(0.0, context_wall_s - active_loop_wall_s))
    active_unattributed_s = float(max(0.0, active_loop_wall_s - forward_sync_sum_s - profiler_step_sum_s))

    validation_rows = []
    for local_idx, (manifest_index, item) in enumerate(batch_pairs):
        baseline_item = manifest["items"][manifest_index]
        tensor_path = tensor_dir / str(baseline_item["tensor_file"])
        if sha256_file(tensor_path) != str(baseline_item["tensor_sha256"]):
            raise RuntimeError(f"baseline tensor sha256 mismatch: {tensor_path}")
        baseline_payload = torch.load(tensor_path, map_location="cpu")
        baseline_tensors = baseline_payload["tensors"]
        real_tokens = int(vision_tokens(item))
        candidate_visual = profiled_output[local_idx, :real_tokens, :].detach()
        reference_visual = baseline_tensors["visual_features"].to(device=device, dtype=model.visual.dtype)
        validation_rows.append(
            {
                "manifest_index": int(manifest_index),
                "id": str(item.entry.get("id")),
                "vision_tokens": int(real_tokens),
                "visual_features": diff_stats(candidate_visual.cpu(), baseline_tensors["visual_features"]),
                "candidate_visual_nonfinite_count": int((~torch.isfinite(candidate_visual.float())).sum().item()),
                "reference_visual_summary": visual_tensor_summary(reference_visual),
                "candidate_visual_summary": visual_tensor_summary(candidate_visual),
            }
        )

    summary = {
        "schema_version": 1,
        "experiment": "07_vision_prefill_optimization",
        "kind": "static_visual_batched_encoder_profile",
        "profile_dir": str(profile_dir),
        "profile_metric": str(args.profile_metric),
        "profiler_step_contract": "one profiler.step() is called after exactly one compiled encoder batch forward",
        "profile_scope": "compiled encoder layers + post LayerNorm only; prefix, projector, text prefill, and decode are outside profiler window",
        "batch_size": int(args.batch_size),
        "fixed_physical_seq_len": int(fixed_s),
        "effective_tokens_per_forward": int(effective_tokens),
        "physical_tokens_per_forward": int(physical_tokens),
        "encoder_physical_tokens_per_s_forward_sync": float((physical_tokens * active_steps) / forward_sync_sum_s)
        if forward_sync_sum_s > 0
        else None,
        "encoder_effective_tokens_per_s_forward_sync": float((effective_tokens * active_steps) / forward_sync_sum_s)
        if forward_sync_sum_s > 0
        else None,
        "profile_active_steps": int(active_steps),
        "profile_warmup_steps": int(args.profile_warmup_steps),
        "profile_context_wall_s": float(context_wall_s),
        "profile_active_loop_wall_s": float(active_loop_wall_s),
        "profile_forward_sync_sum_s": float(forward_sync_sum_s),
        "profile_profiler_step_sum_s": float(profiler_step_sum_s),
        "profile_context_non_active_s": float(context_non_active_s),
        "profile_active_loop_unattributed_s": float(active_unattributed_s),
        "profile_forward_sync_s": stats(forward_sync_times_s),
        "profile_profiler_step_s": stats(profiler_step_times_s),
        "prefix_build_s": float(prefix_build_s),
        "compiled_first_call_s": float(compiled_first_call_s),
        "warmup_forward_sync_s": stats(warmup_times_s),
        "compile": clean_json(compile_meta),
        "candidate": {
            "device": str(device),
            "dtype": str(dtype),
            "vision_attention": os.environ.get("VISION_ATTENTION_IMPL", ""),
            "batch_size": int(args.batch_size),
            "static_visual_fixed_physical_seq_len": int(fixed_s),
            "static_visual_ln_impl": str(args.static_visual_ln_impl),
            "static_visual_ln_linear_mode": str(args.static_visual_ln_linear_mode),
            "static_visual_promptfa_pad_head_dim_to": int(args.static_visual_promptfa_pad_head_dim_to),
            "batched_boundary": "encoder_layers_plus_post_layernorm_only",
            "prefix_boundary": "per_crop_patch_embedding_plus_abs_pos_plus_padding_outside_compile",
        },
        "batch": {
            "profile_batch_index": int(args.profile_batch_index),
            "ids": [str(item.entry.get("id")) for item in batch_items],
            "manifest_indices": [int(manifest_index) for manifest_index, _item in batch_pairs],
            "prefix_meta": clean_json(prefix_meta),
            "first_output_nonfinite_count": int((~torch.isfinite(first_output.float())).sum().item()),
            "profiled_output_nonfinite_count": int((~torch.isfinite(profiled_output.float())).sum().item()),
        },
        "validation": {
            "items": clean_json(validation_rows),
            "visual_nonfinite_item_count": int(sum(int(row["candidate_visual_nonfinite_count"]) > 0 for row in validation_rows)),
            "visual_max_abs_diff": stats(
                [
                    float(row["visual_features"]["max_abs_diff"])
                    for row in validation_rows
                    if row["visual_features"].get("max_abs_diff") is not None
                ]
            ),
        },
        "excluded": clean_json(excluded_rows[:16]),
    }
    summary_path = profile_dir / "profile_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    parsed = run_parser(profile_dir, topn=int(args.parser_topn)) if bool(args.parse_profile) else {"skipped": True}
    summary["parsed_profile"] = parsed

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "profile_dir": str(profile_dir), "parsed_profile": parsed}, indent=2, default=json_default))


if __name__ == "__main__":
    main()
