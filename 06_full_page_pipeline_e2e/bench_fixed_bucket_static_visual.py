#!/usr/bin/env python3
"""Benchmark fixed-size compiled static_visual buckets on real OCR crops.

This experiment asks a service-shaped question:

    If crops are preprocessed with a chosen min_pixels value, can we route all
    crops with real vision sequence length <= S_cap through one compiled
    PromptFA static_visual graph of fixed physical length S_cap?

The compiled boundary covers the native-resolution visual encoder only:
patch embedding, absolute position add, 27 vision encoder layers, and post
layernorm. It deliberately does not include the adaptive MLP projector, text
prefill, or decode in the speed timing. Correctness checks can feed a small
sample through projector/prefill/decode to verify that padding/compile does not
change generated OCR text.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from tokenizers import Tokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EXP5_DIR = REPO_ROOT / "05_full_recognizer_optimizations"
if str(EXP5_DIR) not in sys.path:
    sys.path.insert(0, str(EXP5_DIR))

from bench_page_pipeline_e2e import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    build_detected_crops,
    build_omnidocbench_gt_layout_pages,
    build_queue_inputs_from_crops,
    clean_json,
    load_pages_result,
    page_load_summary,
    resolve_dataset_dir,
    rough_ground_truth_accuracy,
    tok_per_s,
)
from bench_recognizer_queue import QueueInput, json_default  # noqa: E402
from local_modeling_paddleocr_vl import (  # noqa: E402
    VISION_ATTENTION_CHOICES,
    VISION_ATTENTION_ENV,
    VISION_PROMPT_FA_LAYOUT_CHOICES,
    VISION_PROMPT_FA_LAYOUT_ENV,
    VISION_PROMPT_FA_MASK_SPARSE_MODE_ENV,
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
    apply_rotary_pos_emb_vision,
    attention_softmax,
    get_vision_attention_impl,
    get_vision_prompt_fa_layout,
    get_vision_prompt_fa_mask_sparse_mode,
    get_vision_softmax_dtype_mode,
    vision_prompt_flash_attention_bnsd,
)
from probe_static_compile import maybe_sync  # noqa: E402
from run_local_recognition import (  # noqa: E402
    NPU_JIT_COMPILE_CHOICES,
    configure_npu_jit_compile,
    load_preprocessor_config,
    resolve_device,
)

from bench_vision_prefill_only import (  # noqa: E402
    VISION_COMPILE_BACKEND_CHOICES,
    build_single_crop_vision_cu_seqlens,
    build_static_abs_pos_embed,
    build_static_vision_rope,
    parse_vision_dtype,
    tensor_grid,
    vision_compile_backend,
    vision_tokens,
)
from compare_compiled_visual_downstream import (  # noqa: E402
    decoded_proxy,
    diff_stats,
    generate_from_prefill,
    prefill_from_visual_features,
    trim_after_eos,
)


DEFAULT_BUCKET_CONFIGS = "28224:256,28224:384,50176:512,112896:768"


@dataclass
class BucketConfig:
    min_pixels: int
    cap_tokens: int

    @property
    def name(self) -> str:
        return f"min{self.min_pixels}_cap{self.cap_tokens}"


@dataclass
class FixedBucketPreparedItem:
    item: QueueInput
    pixel_values: torch.Tensor
    abs_pos_embed: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    attention_mask: torch.Tensor
    row_keep: torch.Tensor
    real_seq_len: int
    cap_tokens: int


def parse_bucket_configs(raw: str) -> list[BucketConfig]:
    configs: list[BucketConfig] = []
    for part in str(raw or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"bucket config must be MIN_PIXELS:CAP_TOKENS, got {part!r}")
        min_s, cap_s = part.split(":", 1)
        config = BucketConfig(min_pixels=int(min_s), cap_tokens=int(cap_s))
        if config.min_pixels <= 0 or config.cap_tokens <= 0:
            raise ValueError(f"bucket values must be positive, got {part!r}")
        configs.append(config)
    if not configs:
        raise ValueError("--bucket-configs selected zero buckets")
    return configs


def percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * float(q)))))
    return float(sorted_values[idx])


def timing_stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "sum": 0.0,
            "avg": None,
            "std": None,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(len(ordered)),
        "sum": float(sum(ordered)),
        "avg": float(sum(ordered) / len(ordered)),
        "std": float(statistics.pstdev(ordered)) if len(ordered) > 1 else 0.0,
        "min": float(ordered[0]),
        "p50": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "max": float(ordered[-1]),
    }


def select_representative_items(items: list[QueueInput], count: int) -> list[QueueInput]:
    if int(count) <= 0 or len(items) <= int(count):
        return list(items)
    ordered = sorted(items, key=lambda item: (vision_tokens(item), str(item.entry.get("id"))))
    if int(count) == 1:
        return [ordered[(len(ordered) - 1) // 2]]
    positions = [
        int(round(index * (len(ordered) - 1) / max(1, int(count) - 1)))
        for index in range(int(count))
    ]
    selected: list[QueueInput] = []
    seen: set[int] = set()
    for pos in positions:
        pos = min(len(ordered) - 1, max(0, pos))
        if pos in seen:
            continue
        selected.append(ordered[pos])
        seen.add(pos)
    cursor = 0
    while len(selected) < int(count) and cursor < len(ordered):
        if cursor not in seen:
            selected.append(ordered[cursor])
            seen.add(cursor)
        cursor += 1
    return selected


def build_fixed_bucket_attention_mask(real_seq_len: int, cap_tokens: int, *, device: torch.device) -> torch.Tensor:
    real_seq_len = int(real_seq_len)
    cap_tokens = int(cap_tokens)
    if real_seq_len > cap_tokens:
        raise ValueError(f"real_seq_len {real_seq_len} exceeds cap_tokens {cap_tokens}")
    mask = torch.zeros((1, 1, cap_tokens, cap_tokens), device=device, dtype=torch.bool)
    if real_seq_len < cap_tokens:
        mask[..., :real_seq_len, real_seq_len:cap_tokens] = True
        mask[..., real_seq_len:cap_tokens, :real_seq_len] = True
    return mask.contiguous()


def pad_rows(
    tensor: torch.Tensor,
    cap_tokens: int,
    *,
    fill_value: float = 0.0,
) -> torch.Tensor:
    real_seq_len = int(tensor.shape[0])
    cap_tokens = int(cap_tokens)
    if real_seq_len > cap_tokens:
        raise ValueError(f"cannot pad tensor with first dim {real_seq_len} to cap {cap_tokens}")
    if real_seq_len == cap_tokens:
        return tensor.contiguous()
    pad_shape = (cap_tokens - real_seq_len, *tuple(int(dim) for dim in tensor.shape[1:]))
    pad = torch.full(pad_shape, fill_value, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=0).contiguous()


def prepare_fixed_bucket_item(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    item: QueueInput,
    cap_tokens: int,
    device: torch.device,
) -> FixedBucketPreparedItem:
    real_seq_len = int(vision_tokens(item))
    if real_seq_len > int(cap_tokens):
        raise ValueError(f"item {item.entry.get('id')} real seq {real_seq_len} exceeds cap {cap_tokens}")
    pixel_values = item.pixel_values.to(device=device, dtype=model.visual.dtype)
    pixel_values = pad_rows(pixel_values, int(cap_tokens), fill_value=0.0)
    abs_pos = build_static_abs_pos_embed(model, item.image_grid_thw, device=device)
    abs_pos = pad_rows(abs_pos, int(cap_tokens), fill_value=0.0)
    rope_cos, rope_sin = build_static_vision_rope(model, item.image_grid_thw, device=device)
    rope_cos = pad_rows(rope_cos, int(cap_tokens), fill_value=1.0)
    rope_sin = pad_rows(rope_sin, int(cap_tokens), fill_value=0.0)
    attention_mask = build_fixed_bucket_attention_mask(real_seq_len, int(cap_tokens), device=device)
    row_keep = torch.zeros((int(cap_tokens), 1), device=device, dtype=model.visual.dtype)
    row_keep[:real_seq_len, :] = 1
    return FixedBucketPreparedItem(
        item=item,
        pixel_values=pixel_values.contiguous(),
        abs_pos_embed=abs_pos.contiguous(),
        rope_cos=rope_cos.contiguous(),
        rope_sin=rope_sin.contiguous(),
        attention_mask=attention_mask.contiguous(),
        row_keep=row_keep.contiguous(),
        real_seq_len=real_seq_len,
        cap_tokens=int(cap_tokens),
    )


class FixedBucketStaticVisualModule(torch.nn.Module):
    """Fixed-shape static visual encoder for one service bucket."""

    def __init__(self, model: LocalPaddleOCRVLForConditionalGeneration, cap_tokens: int):
        super().__init__()
        self.model = model
        self.cap_tokens = int(cap_tokens)

    def _zero_pad_rows(self, hidden_states: torch.Tensor, row_keep: torch.Tensor) -> torch.Tensor:
        return hidden_states * row_keep.to(dtype=hidden_states.dtype)

    def _attention(
        self,
        attention: torch.nn.Module,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states = attention.q_proj(hidden_states).view(seq_length, attention.num_heads, attention.head_dim)
        key_states = attention.k_proj(hidden_states).view(seq_length, attention.num_heads, attention.head_dim)
        value_states = attention.v_proj(hidden_states).view(seq_length, attention.num_heads, attention.head_dim)
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, rope_cos, rope_sin)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        attention_impl = get_vision_attention_impl()
        if attention_impl == "prompt_flash_attention":
            if get_vision_prompt_fa_layout() != "bnsd":
                raise ValueError("fixed bucket static_visual currently supports PromptFA layout bnsd only")
            attn_output = vision_prompt_flash_attention_bnsd(
                query_states,
                key_states,
                value_states,
                num_heads=int(attention.num_heads),
                scale=float(attention.scaling),
                atten_mask=attention_mask,
            )
        elif attention_impl == "manual":
            scores = torch.matmul(query_states, key_states.transpose(2, 3)) * attention.scaling
            scores = scores.masked_fill(attention_mask, torch.finfo(scores.dtype).min)
            probs = attention_softmax(
                scores,
                dim=-1,
                output_dtype=query_states.dtype,
                mode=get_vision_softmax_dtype_mode(),
            )
            attn_output = torch.matmul(probs, value_states)
        else:
            raise ValueError(f"unknown vision attention implementation: {attention_impl!r}")
        attn_output = attn_output.transpose(1, 2).contiguous().view(seq_length, -1)
        return attention.out_proj(attn_output)

    def forward(
        self,
        pixel_values: torch.Tensor,
        abs_pos_embed: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: torch.Tensor,
        row_keep: torch.Tensor,
    ) -> torch.Tensor:
        transformer = self.model.visual.vision_model
        embeddings_module = transformer.embeddings
        pixel_values = pixel_values.to(dtype=embeddings_module.patch_embedding.weight.dtype)
        hidden_states = embeddings_module.patch_embedding(pixel_values).flatten(-2).squeeze(-1)
        hidden_states = self._zero_pad_rows(hidden_states + abs_pos_embed, row_keep)
        for encoder_layer in transformer.encoder.layers:
            attn_input = encoder_layer.layer_norm1(hidden_states)
            hidden_states = hidden_states + self._attention(
                encoder_layer.self_attn,
                attn_input,
                rope_cos,
                rope_sin,
                attention_mask,
            )
            hidden_states = self._zero_pad_rows(hidden_states, row_keep)
            hidden_states = hidden_states + encoder_layer.mlp(encoder_layer.layer_norm2(hidden_states))
            hidden_states = self._zero_pad_rows(hidden_states, row_keep)
        hidden_states = transformer.post_layernorm(hidden_states)
        return self._zero_pad_rows(hidden_states, row_keep)


def compile_fixed_bucket_forward(
    *,
    wrapper: FixedBucketStaticVisualModule,
    backend_name: str,
    device: torch.device,
) -> tuple[Callable[..., torch.Tensor], dict[str, Any]]:
    if str(backend_name) == "none":
        return wrapper, {
            "enabled": False,
            "backend": "none",
            "compile_api": None,
            "fullgraph": False,
            "dynamic": False,
        }
    backend = vision_compile_backend(str(backend_name), device)
    compile_kwargs: dict[str, Any] = {"fullgraph": True, "dynamic": False}
    if backend is not None:
        compile_kwargs["backend"] = backend
    maybe_sync(device)
    start = time.perf_counter()
    compiled = torch.compile(wrapper, **compile_kwargs)
    maybe_sync(device)
    return compiled, {
        "enabled": True,
        "backend": str(backend_name),
        "compile_api": "torch.compile",
        "fullgraph": True,
        "dynamic": False,
        "compile_wrapper_s": float(time.perf_counter() - start),
    }


def call_fixed_bucket_forward(
    fn: Callable[..., torch.Tensor],
    prepared: FixedBucketPreparedItem,
) -> torch.Tensor:
    return fn(
        prepared.pixel_values,
        prepared.abs_pos_embed,
        prepared.rope_cos,
        prepared.rope_sin,
        prepared.attention_mask,
        prepared.row_keep,
    )


def call_fixed_bucket_real_rows(
    fn: Callable[..., torch.Tensor],
    prepared: FixedBucketPreparedItem,
) -> torch.Tensor:
    """Run a fixed physical bucket and immediately discard padded visual rows."""
    return call_fixed_bucket_forward(fn, prepared)[: prepared.real_seq_len]


def original_visual_forward(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    item: QueueInput,
    device: torch.device,
) -> torch.Tensor:
    cu_seqlens = build_single_crop_vision_cu_seqlens(item.image_grid_thw, device=device)
    return model.visual(
        pixel_values=item.pixel_values.to(device=device, dtype=model.visual.dtype).unsqueeze(0),
        image_grid_thw=item.image_grid_thw,
        cu_seqlens=cu_seqlens,
    )


def item_row(item: QueueInput) -> dict[str, Any]:
    return {
        "id": str(item.entry.get("id")),
        "page_index": int(item.entry.get("page_index", 0)),
        "layout_label": str(item.entry.get("layout_label", "")),
        "crop_size": clean_json(item.entry.get("crop_size", [0, 0])),
        "image_grid_thw": tensor_grid(item),
        "vision_tokens": int(vision_tokens(item)),
        "projected_image_tokens": int(vision_tokens(item) // 4),
        "input_tokens": int(item.input_ids.shape[1]),
    }


@torch.inference_mode()
def run_correctness_checks(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    tokenizer: Tokenizer,
    compiled_forward: Callable[..., torch.Tensor],
    eager_wrapper: FixedBucketStaticVisualModule,
    prepared_items: list[FixedBucketPreparedItem],
    device: torch.device,
    cache_length: int,
    max_new_tokens: int,
    run_downstream: bool,
    rough_gt_min_iou: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    eager_decoded = []
    compiled_decoded = []
    eos_token_id = int(model.config.eos_token_id)
    for idx, prepared in enumerate(prepared_items):
        item = prepared.item
        maybe_sync(device)
        original_visual = original_visual_forward(model=model, item=item, device=device)
        maybe_sync(device)
        fixed_eager = call_fixed_bucket_real_rows(eager_wrapper, prepared)
        maybe_sync(device)
        fixed_compiled = call_fixed_bucket_real_rows(compiled_forward, prepared)
        maybe_sync(device)
        row: dict[str, Any] = {
            "idx": int(idx),
            **item_row(item),
            "real_seq_len": int(prepared.real_seq_len),
            "cap_tokens": int(prepared.cap_tokens),
            "pad_tokens": int(prepared.cap_tokens - prepared.real_seq_len),
            "diffs": {
                "fixed_eager_vs_original_visual": diff_stats(fixed_eager, original_visual),
                "compiled_vs_fixed_eager": diff_stats(fixed_compiled, fixed_eager),
            },
        }
        if run_downstream:
            eager_prefill = prefill_from_visual_features(
                model=model,
                item=item,
                image_features=fixed_eager,
                device=device,
                cache_length=int(cache_length),
            )
            compiled_prefill = prefill_from_visual_features(
                model=model,
                item=item,
                image_features=fixed_compiled,
                device=device,
                cache_length=int(cache_length),
            )
            eager_ids = generate_from_prefill(
                model=model,
                prefill=eager_prefill,
                max_new_tokens=int(max_new_tokens),
                eos_token_id=eos_token_id,
            )
            compiled_ids = generate_from_prefill(
                model=model,
                prefill=compiled_prefill,
                max_new_tokens=int(max_new_tokens),
                eos_token_id=eos_token_id,
            )
            eager_tokens = [int(value) for value in eager_ids[0].detach().cpu().tolist()]
            compiled_tokens = [int(value) for value in compiled_ids[0].detach().cpu().tolist()]
            eager_trimmed = trim_after_eos(eager_tokens, eos_token_id)
            compiled_trimmed = trim_after_eos(compiled_tokens, eos_token_id)
            eager_text = tokenizer.decode(eager_trimmed, skip_special_tokens=True)
            compiled_text = tokenizer.decode(compiled_trimmed, skip_special_tokens=True)
            row["diffs"]["projected_image_embeddings"] = diff_stats(
                compiled_prefill["image_embeds"],
                eager_prefill["image_embeds"],
            )
            row["diffs"]["prefill_logits"] = diff_stats(compiled_prefill["logits"], eager_prefill["logits"])
            row["downstream"] = {
                "eager_tokens": eager_trimmed,
                "compiled_tokens": compiled_trimmed,
                "token_match": bool(eager_trimmed == compiled_trimmed),
                "eager_text": eager_text,
                "compiled_text": compiled_text,
                "text_match": bool(eager_text == compiled_text),
            }
            eager_decoded.append(
                decoded_proxy(
                    item=item,
                    trimmed_ids=eager_trimmed,
                    text=eager_text,
                    eos_token_id=eos_token_id,
                    max_new_tokens=int(max_new_tokens),
                )
            )
            compiled_decoded.append(
                decoded_proxy(
                    item=item,
                    trimmed_ids=compiled_trimmed,
                    text=compiled_text,
                    eos_token_id=eos_token_id,
                    max_new_tokens=int(max_new_tokens),
                )
            )
        rows.append(row)

    fixed_eager_bad = [
        row for row in rows if not row["diffs"]["fixed_eager_vs_original_visual"]["allclose_atol_5e_2_rtol_5e_2"]
    ]
    compiled_bad = [
        row for row in rows if not row["diffs"]["compiled_vs_fixed_eager"]["allclose_atol_5e_2_rtol_5e_2"]
    ]
    compiled_nonfinite = [
        row
        for row in rows
        if int(row["diffs"]["compiled_vs_fixed_eager"].get("lhs_nonfinite_count", 0) or 0) > 0
    ]
    downstream_rows = [row for row in rows if "downstream" in row]
    token_mismatches = [row for row in downstream_rows if not row["downstream"]["token_match"]]
    text_mismatches = [row for row in downstream_rows if not row["downstream"]["text_match"]]
    summary = {
        "checked_count": int(len(rows)),
        "run_downstream": bool(run_downstream),
        "fixed_eager_vs_original_allclose_fail_count": int(len(fixed_eager_bad)),
        "compiled_vs_fixed_eager_allclose_fail_count": int(len(compiled_bad)),
        "compiled_real_output_nonfinite_item_count": int(len(compiled_nonfinite)),
        "downstream_token_mismatch_count": int(len(token_mismatches)),
        "downstream_text_mismatch_count": int(len(text_mismatches)),
        "all_required_checks_passed": bool(
            not fixed_eager_bad
            and not compiled_nonfinite
            and (not run_downstream or (not token_mismatches and not text_mismatches))
        ),
        "all_required_checks_note": (
            "Compiled-vs-fixed-eager allclose failures are diagnostic only because TorchAir/CANN can introduce "
            "finite visual drift that may still preserve OCR output. Required checks are: fixed padded eager "
            "matches original visual, compiled real rows are finite, and downstream generated tokens/text match "
            "when downstream checks are enabled."
        ),
    }
    if run_downstream:
        summary["eager_rough_ground_truth_accuracy"] = rough_ground_truth_accuracy(
            eager_decoded,
            min_iou=float(rough_gt_min_iou),
        )
        summary["compiled_rough_ground_truth_accuracy"] = rough_ground_truth_accuracy(
            compiled_decoded,
            min_iou=float(rough_gt_min_iou),
        )
    return {"summary": summary, "items": rows}


@torch.inference_mode()
def run_bucket(
    *,
    config: BucketConfig,
    queue_inputs: list[QueueInput],
    model: LocalPaddleOCRVLForConditionalGeneration,
    tokenizer: Tokenizer,
    device: torch.device,
    compile_backend: str,
    benchmark_repeats: int,
    warmup_forwards: int,
    correctness_items: int,
    cache_length: int,
    max_new_tokens: int,
    run_downstream_check: bool,
    rough_gt_min_iou: float,
    max_benchmark_items: int,
) -> dict[str, Any]:
    eligible = [item for item in queue_inputs if int(vision_tokens(item)) <= int(config.cap_tokens)]
    excluded = [item for item in queue_inputs if int(vision_tokens(item)) > int(config.cap_tokens)]
    if int(max_benchmark_items) > 0:
        benchmark_items = eligible[: int(max_benchmark_items)]
    else:
        benchmark_items = eligible
    if not benchmark_items:
        return {
            "bucket": config.name,
            "min_pixels": int(config.min_pixels),
            "cap_tokens": int(config.cap_tokens),
            "eligible_count": int(len(eligible)),
            "excluded_count": int(len(excluded)),
            "skipped": True,
            "skip_reason": "zero eligible benchmark items",
        }

    maybe_sync(device)
    prepare_start = time.perf_counter()
    prepared_items = [
        prepare_fixed_bucket_item(model=model, item=item, cap_tokens=int(config.cap_tokens), device=device)
        for item in benchmark_items
    ]
    maybe_sync(device)
    prepare_s = time.perf_counter() - prepare_start

    wrapper = FixedBucketStaticVisualModule(model, int(config.cap_tokens)).eval()
    compiled_forward, compile_meta = compile_fixed_bucket_forward(
        wrapper=wrapper,
        backend_name=str(compile_backend),
        device=device,
    )
    first = prepared_items[0]
    maybe_sync(device)
    first_call_start = time.perf_counter()
    first_output = call_fixed_bucket_forward(compiled_forward, first)
    maybe_sync(device)
    compile_meta["compiled_first_call_s"] = float(time.perf_counter() - first_call_start)
    compile_meta["first_output_shape"] = [int(dim) for dim in first_output.shape]

    warmup_count = min(max(0, int(warmup_forwards)), len(prepared_items))
    maybe_sync(device)
    warmup_start = time.perf_counter()
    for idx in range(warmup_count):
        call_fixed_bucket_forward(compiled_forward, prepared_items[idx])
    maybe_sync(device)
    warmup_s = time.perf_counter() - warmup_start

    forward_rows: list[dict[str, Any]] = []
    benchmark_repeats = max(1, int(benchmark_repeats))
    total_start = time.perf_counter()
    for repeat_idx in range(benchmark_repeats):
        for item_idx, prepared in enumerate(prepared_items):
            maybe_sync(device)
            start = time.perf_counter()
            out = call_fixed_bucket_forward(compiled_forward, prepared)
            real_out = out[: prepared.real_seq_len]
            maybe_sync(device)
            elapsed = time.perf_counter() - start
            forward_rows.append(
                {
                    "repeat_idx": int(repeat_idx),
                    "item_idx": int(item_idx),
                    "id": str(prepared.item.entry.get("id")),
                    "layout_label": str(prepared.item.entry.get("layout_label", "")),
                    "real_seq_len": int(prepared.real_seq_len),
                    "cap_tokens": int(config.cap_tokens),
                    "pad_tokens": int(config.cap_tokens - prepared.real_seq_len),
                    "elapsed_s": float(elapsed),
                    "output_shape": [int(dim) for dim in out.shape],
                    "real_output_shape": [int(dim) for dim in real_out.shape],
                    "padded_rows_discarded_after_encoder": True,
                }
            )
    total_s = time.perf_counter() - total_start

    real_seq_sum_per_repeat = int(sum(prepared.real_seq_len for prepared in prepared_items))
    physical_seq_sum_per_repeat = int(len(prepared_items) * int(config.cap_tokens))
    elapsed_values = [float(row["elapsed_s"]) for row in forward_rows]

    correctness_sample = [
        prepare_fixed_bucket_item(model=model, item=item, cap_tokens=int(config.cap_tokens), device=device)
        for item in select_representative_items(eligible, int(correctness_items))
    ]
    correctness = run_correctness_checks(
        model=model,
        tokenizer=tokenizer,
        compiled_forward=compiled_forward,
        eager_wrapper=wrapper,
        prepared_items=correctness_sample,
        device=device,
        cache_length=int(cache_length),
        max_new_tokens=int(max_new_tokens),
        run_downstream=bool(run_downstream_check),
        rough_gt_min_iou=float(rough_gt_min_iou),
    )

    eligible_real_seqs = [int(vision_tokens(item)) for item in eligible]
    excluded_real_seqs = [int(vision_tokens(item)) for item in excluded]
    return {
        "bucket": config.name,
        "min_pixels": int(config.min_pixels),
        "cap_tokens": int(config.cap_tokens),
        "skipped": False,
        "input_count": int(len(queue_inputs)),
        "eligible_count": int(len(eligible)),
        "eligible_pct": float(100.0 * len(eligible) / max(1, len(queue_inputs))),
        "excluded_count": int(len(excluded)),
        "excluded_pct": float(100.0 * len(excluded) / max(1, len(queue_inputs))),
        "benchmark_item_count": int(len(prepared_items)),
        "benchmark_repeats": int(benchmark_repeats),
        "forward_count": int(len(forward_rows)),
        "label_counts": {
            "eligible": dict(sorted(Counter(str(item.entry.get("layout_label", "")) for item in eligible).items())),
            "excluded": dict(sorted(Counter(str(item.entry.get("layout_label", "")) for item in excluded).items())),
            "benchmark": dict(sorted(Counter(str(item.entry.get("layout_label", "")) for item in benchmark_items).items())),
        },
        "real_seq_len": {
            "eligible": timing_stats([float(value) for value in eligible_real_seqs]),
            "excluded": timing_stats([float(value) for value in excluded_real_seqs]),
        },
        "padding": {
            "effective_tokens_per_repeat": int(real_seq_sum_per_repeat),
            "physical_tokens_per_repeat": int(physical_seq_sum_per_repeat),
            "padding_tokens_per_repeat": int(physical_seq_sum_per_repeat - real_seq_sum_per_repeat),
            "padding_waste_pct": float(
                100.0 * (physical_seq_sum_per_repeat - real_seq_sum_per_repeat) / max(1, physical_seq_sum_per_repeat)
            ),
        },
        "timing_s": {
            "prepare_device_inputs_s": float(prepare_s),
            "warmup_s": float(warmup_s),
            "total_measured_s": float(total_s),
            "per_forward_s": timing_stats(elapsed_values),
        },
        "throughput": {
            "items_per_s": tok_per_s(len(forward_rows), total_s),
            "physical_vision_tokens_per_s": tok_per_s(
                int(physical_seq_sum_per_repeat * benchmark_repeats),
                total_s,
            ),
            "effective_vision_tokens_per_s": tok_per_s(
                int(real_seq_sum_per_repeat * benchmark_repeats),
                total_s,
            ),
        },
        "compile": clean_json(compile_meta),
        "correctness": clean_json(correctness),
        "forward_timings": forward_rows,
        "sample_items": [item_row(item) for item in benchmark_items[:16]],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--page-start", type=int, default=0)
    parser.add_argument("--num-pages", type=int, default=64)
    parser.add_argument("--layout-source", default="omnidocbench_gt", choices=["omnidocbench_gt"])
    parser.add_argument("--crop-padding", type=int, default=0)
    parser.add_argument("--min-crop-side", type=int, default=4)
    parser.add_argument("--skip-labels", default="")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "fp32", "float32", "bf16", "bfloat16"])
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--vision-attention", default=os.environ.get(VISION_ATTENTION_ENV, "prompt_flash_attention"), choices=VISION_ATTENTION_CHOICES)
    parser.add_argument(
        "--vision-prompt-fa-layout",
        default=os.environ.get(VISION_PROMPT_FA_LAYOUT_ENV, "bnsd"),
        choices=VISION_PROMPT_FA_LAYOUT_CHOICES,
    )
    parser.add_argument(
        "--vision-prompt-fa-mask-sparse-mode",
        type=int,
        default=int(os.environ.get(VISION_PROMPT_FA_MASK_SPARSE_MODE_ENV, "0")),
        choices=[0, 1],
    )
    parser.add_argument("--vision-compile-backend", default="torchair", choices=VISION_COMPILE_BACKEND_CHOICES)
    parser.add_argument("--bucket-configs", default=DEFAULT_BUCKET_CONFIGS)
    parser.add_argument("--benchmark-repeats", type=int, default=1)
    parser.add_argument("--warmup-forwards", type=int, default=4)
    parser.add_argument("--correctness-items", type=int, default=8)
    parser.add_argument("--cache-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--rough-gt-min-iou", type=float, default=0.5)
    parser.add_argument("--max-crops", type=int, default=0)
    parser.add_argument("--max-benchmark-items", type=int, default=0)
    parser.add_argument("--run-downstream-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    configs = parse_bucket_configs(args.bucket_configs)
    os.environ[VISION_ATTENTION_ENV] = str(args.vision_attention)
    os.environ[VISION_PROMPT_FA_LAYOUT_ENV] = str(args.vision_prompt_fa_layout)
    os.environ[VISION_PROMPT_FA_MASK_SPARSE_MODE_ENV] = str(args.vision_prompt_fa_mask_sparse_mode)

    device = resolve_device(args.device)
    dtype = parse_vision_dtype(args.dtype)
    configure_npu_jit_compile(args.npu_jit_compile, device)
    model_dir = _resolve_model_dir(args.model)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    page_load = load_pages_result(resolve_dataset_dir(args.dataset_dir), page_start=int(args.page_start), num_pages=int(args.num_pages))
    layout_pages, layout_timing = build_omnidocbench_gt_layout_pages(page_load.pages, include_ignored=False, include_empty_gt=False)
    crops, crop_summary, crop_timing = build_detected_crops(pages=page_load.pages, layout_pages=layout_pages, args=args)
    raw_crop_count = int(len(crops))
    if int(args.max_crops) > 0:
        crops = crops[: int(args.max_crops)]
    if not crops:
        raise RuntimeError("zero crops selected")

    original_pre_cfg = load_preprocessor_config(model_dir)
    queue_inputs_by_min_pixels: dict[int, list[QueueInput]] = {}
    input_build_summaries: dict[str, Any] = {}
    for min_pixels in sorted({int(config.min_pixels) for config in configs}):
        pre_cfg = dict(original_pre_cfg)
        pre_cfg["min_pixels"] = int(min_pixels)
        queue_inputs, input_summary = build_queue_inputs_from_crops(
            crops=crops,
            tokenizer=tokenizer,
            pre_cfg=pre_cfg,
            prompt_override=args.prompt,
        )
        queue_inputs_by_min_pixels[int(min_pixels)] = queue_inputs
        input_build_summaries[str(min_pixels)] = input_summary

    maybe_sync(device)
    model_load_start = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device=device)
    maybe_sync(device)
    model_load_s = time.perf_counter() - model_load_start

    bucket_rows: list[dict[str, Any]] = []
    for config in configs:
        print(
            f"FIXED_BUCKET_PROGRESS bucket={config.name} attention={get_vision_attention_impl()} "
            f"backend={args.vision_compile_backend}",
            file=sys.stderr,
            flush=True,
        )
        row = run_bucket(
            config=config,
            queue_inputs=queue_inputs_by_min_pixels[int(config.min_pixels)],
            model=model,
            tokenizer=tokenizer,
            device=device,
            compile_backend=str(args.vision_compile_backend),
            benchmark_repeats=int(args.benchmark_repeats),
            warmup_forwards=int(args.warmup_forwards),
            correctness_items=int(args.correctness_items),
            cache_length=int(args.cache_length),
            max_new_tokens=int(args.max_new_tokens),
            run_downstream_check=bool(args.run_downstream_check),
            rough_gt_min_iou=float(args.rough_gt_min_iou),
            max_benchmark_items=int(args.max_benchmark_items),
        )
        bucket_rows.append(row)

    output = {
        "experiment": "06_fixed_bucket_static_visual",
        "scope": (
            "OmniDocBench GT OCR crops -> PaddleOCR-VL crop preprocessing per min_pixels -> fixed-size padded "
            "static_visual service buckets. Timed region is the compiled/eager native-resolution visual encoder "
            "forward only, with per-forward device synchronization."
        ),
        "model": str(model_dir),
        "dataset_dir": str(resolve_dataset_dir(args.dataset_dir)),
        "device": str(device),
        "dtype": str(dtype),
        "npu_jit_compile": str(args.npu_jit_compile),
        "vision_attention": get_vision_attention_impl(),
        "vision_prompt_fa_layout": get_vision_prompt_fa_layout(),
        "vision_prompt_fa_mask_sparse_mode": int(get_vision_prompt_fa_mask_sparse_mode()),
        "vision_compile_backend": str(args.vision_compile_backend),
        "bucket_configs": [{"min_pixels": c.min_pixels, "cap_tokens": c.cap_tokens, "name": c.name} for c in configs],
        "measurement_contract": {
            "forward_boundary": "static_visual only: patch embedding + abs position + visual encoder + post layernorm",
            "fixed_shape_contract": (
                "Compiled callable shape is fixed by cap_tokens. Per-crop pixel_values, abs_pos_embed, RoPE, "
                "BOOL attention mask, and row_keep tensors have the same physical shape and are inputs."
            ),
            "padding_discard_contract": (
                "The compiled static_visual callable computes cap_tokens physical rows, but callers slice "
                "[:real_seq_len] immediately after the encoder output. Padded rows are never fed into the "
                "adaptive MLP projector, text prefill, or decode correctness path."
            ),
            "timing": "Each measured forward has one device sync before and one after the call.",
            "tokens": {
                "physical_vision_tokens_per_s": "counts cap_tokens for every forward, including padded rows",
                "effective_vision_tokens_per_s": "counts only real pre-padding vision tokens",
            },
        },
        "page_start": int(args.page_start),
        "page_count": int(len(page_load.pages)),
        "page_load": page_load_summary(page_load),
        "raw_extracted_crop_count": int(raw_crop_count),
        "selected_crop_count": int(len(crops)),
        "crop_summary": clean_json(crop_summary),
        "layout_timing_s": clean_json(layout_timing),
        "crop_timing_s": clean_json(crop_timing),
        "input_build_summary_by_min_pixels": clean_json(input_build_summaries),
        "setup_timing_s": {"model_load_s": float(model_load_s)},
        "preprocessor_base": {
            "original_min_pixels": int(original_pre_cfg["min_pixels"]),
            "original_max_pixels": int(original_pre_cfg["max_pixels"]),
            "patch_size": int(original_pre_cfg["patch_size"]),
            "merge_size": int(original_pre_cfg["merge_size"]),
            "temporal_patch_size": int(original_pre_cfg["temporal_patch_size"]),
        },
        "run_options": {
            "benchmark_repeats": int(args.benchmark_repeats),
            "warmup_forwards": int(args.warmup_forwards),
            "correctness_items": int(args.correctness_items),
            "run_downstream_check": bool(args.run_downstream_check),
            "cache_length": int(args.cache_length),
            "max_new_tokens": int(args.max_new_tokens),
            "rough_gt_min_iou": float(args.rough_gt_min_iou),
            "max_benchmark_items": int(args.max_benchmark_items),
        },
        "buckets": bucket_rows,
    }
    print(json.dumps(output, indent=2 if args.json else None, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
