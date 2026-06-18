#!/usr/bin/env python3
"""Benchmark only the PaddleOCR-VL vision prefill path on real page crops.

This is an experiment-6 side harness. It intentionally reuses the same
OmniDocBench page loading, GT layout crop extraction, crop preprocessing, and
prompt/input construction as the full page pipeline, then stops after the
native-resolution vision encoder plus adaptive MLP projector.

Measured vision call per crop:

    CPU preprocessed crop tensor -> device transfer -> model.get_image_features()

No text-token embedding, image embedding scatter, KV prefill, LM head, decode,
or output validation is run here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
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
    aggregate_timing_dicts,
    build_detected_crops,
    build_omnidocbench_gt_layout_pages,
    build_queue_inputs_from_crops,
    clean_json,
    load_pages_result,
    page_load_summary,
    resolve_dataset_dir,
    tok_per_s,
)
from bench_recognizer_queue import QueueInput, json_default, stats  # noqa: E402
from local_modeling_paddleocr_vl import (  # noqa: E402
    VISION_ATTENTION_CHOICES,
    VISION_ATTENTION_ENV,
    VISION_PROMPT_FA_LAYOUT_CHOICES,
    VISION_PROMPT_FA_LAYOUT_ENV,
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
    apply_rotary_pos_emb_vision,
    attention_softmax,
    get_vision_attention_impl,
    get_vision_prompt_fa_layout,
    get_vision_softmax_dtype_mode,
    vision_prompt_flash_attention_bnsd,
)
from probe_static_compile import import_torchair, maybe_sync  # noqa: E402
from run_local_recognition import (  # noqa: E402
    NPU_JIT_COMPILE_CHOICES,
    configure_npu_jit_compile,
    load_preprocessor_config,
    resolve_device,
)


MODE_CHOICES = ("sync_per_crop", "unsynced_loop")
PROFILE_METRIC_CHOICES = ("pipe", "memory", "l2", "memory_access")
CROP_SAMPLE_CHOICES = ("all", "small_medium_large", "small_only")
VISION_COMPILE_BACKEND_CHOICES = ("none", "default", "aot_eager", "inductor", "torchair")
VISION_FORWARD_BOUNDARY_CHOICES = ("get_image_features", "visual", "static_visual")
STATIC_VISUAL_PAD_MODE_CHOICES = ("none", "mask_pad_one", "mask_pad_to_128")


def parse_vision_dtype(name: str) -> torch.dtype:
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def parse_modes(raw: str) -> list[str]:
    modes = [item.strip() for item in str(raw).replace(",", " ").split() if item.strip()]
    if not modes:
        raise ValueError("--modes must select at least one mode")
    bad = [mode for mode in modes if mode not in MODE_CHOICES]
    if bad:
        raise ValueError(f"unsupported mode(s): {bad}; choices={MODE_CHOICES}")
    deduped: list[str] = []
    for mode in modes:
        if mode not in deduped:
            deduped.append(mode)
    return deduped


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


def make_profile_run_dir(
    root: Path,
    *,
    mode: str,
    attention: str,
    dtype: torch.dtype,
    crop_count: int,
    metric: str,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dtype_name = str(dtype).replace("torch.", "")
    safe_attention = str(attention).replace("/", "_")
    return root / f"vision_prefill_{timestamp}_{mode}_{safe_attention}_{dtype_name}_{crop_count}crops_{metric}"


def tensor_grid(item: QueueInput) -> list[int]:
    return [int(value) for value in item.image_grid_thw.reshape(-1).tolist()]


def vision_tokens(item: QueueInput) -> int:
    return int(item.image_grid_thw.prod().item())


def projected_tokens(item: QueueInput, *, merge_size: int) -> int:
    return int(vision_tokens(item) // int(merge_size) // int(merge_size))


def summarize_inputs(inputs: list[QueueInput], *, merge_size: int) -> dict[str, Any]:
    vision_counts = [vision_tokens(item) for item in inputs]
    projected_counts = [projected_tokens(item, merge_size=merge_size) for item in inputs]
    input_counts = [int(item.input_ids.shape[1]) for item in inputs]
    grids = [tuple(tensor_grid(item)) for item in inputs]
    grid_counts = Counter(grids)
    label_counts = Counter(str(item.entry.get("layout_label", "unknown")) for item in inputs)
    prompt_counts = Counter(str(item.prompt) for item in inputs)
    crop_sizes = [
        [int(value) for value in clean_json(item.entry.get("crop_size", [0, 0]))]
        for item in inputs
    ]
    return {
        "count": int(len(inputs)),
        "total_vision_tokens": int(sum(vision_counts)),
        "total_projected_image_tokens": int(sum(projected_counts)),
        "total_input_tokens": int(sum(input_counts)),
        "vision_tokens": {
            "stats": stats([float(value) for value in vision_counts]),
            "unique_count": int(len(set(vision_counts))),
        },
        "projected_image_tokens": {
            "stats": stats([float(value) for value in projected_counts]),
            "unique_count": int(len(set(projected_counts))),
        },
        "input_tokens": {
            "stats": stats([float(value) for value in input_counts]),
            "unique_count": int(len(set(input_counts))),
        },
        "image_grid_thw": {
            "unique_count": int(len(grid_counts)),
            "top_buckets": [
                {"grid": [int(value) for value in grid], "count": int(count)}
                for grid, count in sorted(grid_counts.items(), key=lambda item: (-item[1], item[0]))[:16]
            ],
        },
        "label_counts": dict(sorted(label_counts.items())),
        "prompt_counts": dict(sorted(prompt_counts.items())),
        "crop_size_samples": crop_sizes[:16],
    }


def clone_input_with_sample_meta(
    item: QueueInput,
    *,
    bucket: str,
    selected_rank: int,
    original_idx: int,
) -> QueueInput:
    entry = dict(item.entry)
    entry["crop_sample_bucket"] = bucket
    entry["crop_sample_selected_rank"] = int(selected_rank)
    entry["crop_sample_original_idx"] = int(original_idx)
    return QueueInput(
        entry=entry,
        crop_path=item.crop_path,
        prompt=item.prompt,
        input_ids=item.input_ids,
        attention_mask=item.attention_mask,
        pixel_values=item.pixel_values,
        image_grid_thw=item.image_grid_thw,
        timing_s=item.timing_s,
    )


def parse_crop_ids(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").replace(",", " ").split() if item.strip()]


def filter_crops_by_crop_ids(crops: list[Any], crop_ids: list[str]) -> tuple[list[Any], dict[str, Any]]:
    if not crop_ids:
        return crops, {
            "enabled": False,
            "requested_ids": [],
            "crop_count_before_filter": int(len(crops)),
            "selected_count": int(len(crops)),
            "missing_ids": [],
            "selected": [],
        }
    by_id = {str(item.entry.get("id")): item for item in crops}
    missing = [crop_id for crop_id in crop_ids if crop_id not in by_id]
    if missing:
        raise ValueError(f"--crop-ids requested missing crop ids: {missing[:16]}")
    selected = [by_id[crop_id] for crop_id in crop_ids]
    return selected, {
        "enabled": True,
        "requested_ids": list(crop_ids),
        "crop_count_before_filter": int(len(crops)),
        "selected_count": int(len(selected)),
        "missing_ids": [],
        "selected": [
            {
                "id": str(item.entry.get("id")),
                "page_index": int(item.entry.get("page_index", 0)),
                "layout_label": str(item.entry.get("layout_label", "")),
                "crop_size": clean_json(item.entry.get("crop_size", [0, 0])),
            }
            for item in selected
        ],
    }


def add_queue_input_details_to_crop_id_filter(summary: dict[str, Any], inputs: list[QueueInput]) -> dict[str, Any]:
    if not bool(summary.get("enabled")):
        return summary
    by_id = {str(item.entry.get("id")): item for item in inputs}
    enriched = []
    for row in list(summary.get("selected") or []):
        item = by_id.get(str(row.get("id")))
        if item is None:
            enriched.append(row)
            continue
        enriched.append(
            {
                **row,
                "vision_tokens": int(vision_tokens(item)),
                "projected_image_tokens": int(projected_tokens(item, merge_size=2)),
                "input_tokens": int(item.input_ids.shape[1]),
                "image_grid_thw": tensor_grid(item),
            }
        )
    summary = dict(summary)
    summary["selected"] = enriched
    return summary


def select_profile_crop_sample(inputs: list[QueueInput], *, strategy: str) -> tuple[list[QueueInput], dict[str, Any]]:
    if strategy not in CROP_SAMPLE_CHOICES:
        raise ValueError(f"unsupported crop sample strategy: {strategy}; choices={CROP_SAMPLE_CHOICES}")
    if strategy == "all":
        return inputs, {
            "strategy": "all",
            "input_count_before_sample": int(len(inputs)),
            "selected_count": int(len(inputs)),
            "note": "No crop-size sampling was applied.",
        }
    if not inputs:
        raise ValueError("cannot sample representative crops from an empty input list")

    sorted_pairs = sorted(
        enumerate(inputs),
        key=lambda pair: (
            vision_tokens(pair[1]),
            int(pair[1].input_ids.shape[1]),
            str(pair[1].entry.get("id", "")),
        ),
    )
    if strategy == "small_only":
        original_idx, item = sorted_pairs[0]
        cloned = clone_input_with_sample_meta(
            item,
            bucket="small",
            selected_rank=0,
            original_idx=original_idx,
        )
        return [cloned], {
            "strategy": strategy,
            "input_count_before_sample": int(len(inputs)),
            "selected_count": 1,
            "selection_basis": "lowest post-preprocessing vision token count over real extracted/preprocessed OCR crops",
            "selected": [
                {
                    "bucket": "small",
                    "selected_rank": 0,
                    "original_idx": int(original_idx),
                    "sorted_position": 0,
                    "id": str(item.entry.get("id")),
                    "page_index": int(item.entry.get("page_index", 0)),
                    "layout_label": str(item.entry.get("layout_label", "")),
                    "vision_tokens": int(vision_tokens(item)),
                    "input_tokens": int(item.input_ids.shape[1]),
                    "image_grid_thw": tensor_grid(item),
                    "crop_size": clean_json(item.entry.get("crop_size", [0, 0])),
                }
            ],
        }
    n = len(sorted_pairs)
    targets = [
        ("small", 0),
        ("medium", (n - 1) // 2),
        ("large", n - 1),
    ]
    used_positions: set[int] = set()
    selected: list[QueueInput] = []
    selected_rows: list[dict[str, Any]] = []

    for rank, (bucket, target_pos) in enumerate(targets):
        if len(used_positions) >= n:
            break
        best_pos = min(
            (pos for pos in range(n) if pos not in used_positions),
            key=lambda pos: (abs(pos - target_pos), pos),
        )
        used_positions.add(best_pos)
        original_idx, item = sorted_pairs[best_pos]
        cloned = clone_input_with_sample_meta(
            item,
            bucket=bucket,
            selected_rank=rank,
            original_idx=original_idx,
        )
        selected.append(cloned)
        selected_rows.append(
            {
                "bucket": bucket,
                "selected_rank": int(rank),
                "original_idx": int(original_idx),
                "sorted_position": int(best_pos),
                "id": str(item.entry.get("id")),
                "page_index": int(item.entry.get("page_index", 0)),
                "layout_label": str(item.entry.get("layout_label", "")),
                "vision_tokens": int(vision_tokens(item)),
                "input_tokens": int(item.input_ids.shape[1]),
                "image_grid_thw": tensor_grid(item),
                "crop_size": clean_json(item.entry.get("crop_size", [0, 0])),
            }
        )

    return selected, {
        "strategy": strategy,
        "input_count_before_sample": int(len(inputs)),
        "selected_count": int(len(selected)),
        "selection_basis": "vision_tokens sorted ascending over real extracted/preprocessed OCR crops",
        "selected": selected_rows,
    }


def build_single_crop_vision_cu_seqlens(image_grid_thw: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    grid = image_grid_thw.detach().cpu().reshape(-1, 3)
    lengths: list[int] = []
    for t, h, w in grid.tolist():
        lengths.extend([int(h) * int(w)] * int(t))
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + int(length))
    return torch.tensor(cu, device=device, dtype=torch.int32)


def single_crop_grid_ints(image_grid_thw: torch.Tensor) -> tuple[int, int, int]:
    grid = image_grid_thw.detach().cpu().reshape(-1, 3)
    if int(grid.shape[0]) != 1:
        raise ValueError(f"static single-crop vision expects exactly one image grid, got {tuple(grid.shape)}")
    t, h, w = grid[0].tolist()
    return int(t), int(h), int(w)


def build_static_abs_pos_embed(
    model: LocalPaddleOCRVLForConditionalGeneration,
    image_grid_thw: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    t, h, w = single_crop_grid_ints(image_grid_thw)
    embeddings_module = model.visual.vision_model.embeddings
    dtype = embeddings_module.patch_embedding.weight.dtype
    dummy = torch.empty((int(t) * int(h) * int(w), embeddings_module.embed_dim), device=device, dtype=dtype)
    with torch.inference_mode():
        pos = embeddings_module.interpolate_pos_encoding(dummy, int(h), int(w)).squeeze(0).repeat(int(t), 1)
    return pos.contiguous()


def build_static_vision_rope(
    model: LocalPaddleOCRVLForConditionalGeneration,
    image_grid_thw: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    t, h, w = single_crop_grid_ints(image_grid_thw)
    del t
    encoder = model.visual.vision_model.encoder
    image_pids = torch.arange(int(image_grid_thw.prod().item()), device=device, dtype=torch.int64) % int(h * w)
    pids = torch.stack((image_pids // int(w), image_pids % int(w)), dim=-1)
    rotary_max = encoder.rotary_pos_emb(max(int(h), int(w)))
    rotary_embeddings = rotary_max[pids].flatten(1).repeat(1, 2)
    return rotary_embeddings.cos().contiguous(), rotary_embeddings.sin().contiguous()


def build_static_pad_attention_mask(real_seq_len: int, pad_tokens: int, *, device: torch.device) -> torch.Tensor:
    if int(pad_tokens) <= 0:
        return torch.zeros((1, 1, 1, 1), device=device, dtype=torch.bool)
    physical_seq_len = int(real_seq_len) + int(pad_tokens)
    mask = torch.zeros((1, 1, physical_seq_len, physical_seq_len), device=device, dtype=torch.bool)
    real = int(real_seq_len)
    mask[..., :real, real:physical_seq_len] = True
    mask[..., real:physical_seq_len, :real] = True
    return mask.contiguous()


def static_visual_pad_tokens(real_seq_len: int, mode: str) -> int:
    if mode == "none":
        return 0
    if mode == "mask_pad_one":
        return 1 if int(real_seq_len) % 16 == 0 else 0
    if mode != "mask_pad_to_128":
        raise ValueError(f"unsupported static visual pad mode: {mode!r}; choices={STATIC_VISUAL_PAD_MODE_CHOICES}")
    real_seq_len = int(real_seq_len)
    if real_seq_len % 16 != 0:
        return 0
    min_physical_seq_len = real_seq_len + 1
    if min_physical_seq_len <= 128:
        return 1
    physical_seq_len = ((min_physical_seq_len + 127) // 128) * 128
    return int(physical_seq_len - real_seq_len)


class SingleCropVisionFeatureModule(torch.nn.Module):
    """Shape-specialized wrapper for compiling one real crop's vision path."""

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        image_grid_thw: torch.Tensor,
        *,
        boundary: str,
        device: torch.device,
        static_visual_pad_mode: str = "none",
    ):
        super().__init__()
        self.model = model
        self.boundary = str(boundary)
        self.static_visual_pad_mode = str(static_visual_pad_mode)
        if self.static_visual_pad_mode not in STATIC_VISUAL_PAD_MODE_CHOICES:
            raise ValueError(
                f"unsupported static visual pad mode: {self.static_visual_pad_mode!r}; "
                f"choices={STATIC_VISUAL_PAD_MODE_CHOICES}"
            )
        if self.static_visual_pad_mode != "none" and self.boundary != "static_visual":
            raise ValueError("--static-visual-pad-mode is only valid for boundary=static_visual")
        self.register_buffer("image_grid_thw_const", image_grid_thw.detach().clone(), persistent=False)
        self.register_buffer(
            "cu_seqlens_const",
            build_single_crop_vision_cu_seqlens(image_grid_thw, device=device),
            persistent=False,
        )
        self.static_real_seq_len = int(image_grid_thw.prod().item())
        self.static_pad_tokens = 0
        if self.boundary == "static_visual" and self.static_visual_pad_mode != "none":
            grid_t, _grid_h, _grid_w = single_crop_grid_ints(image_grid_thw)
            if int(grid_t) != 1:
                raise ValueError(
                    f"static_visual {self.static_visual_pad_mode} currently supports single-image crop grids "
                    "with T=1 only"
                )
            self.static_pad_tokens = static_visual_pad_tokens(self.static_real_seq_len, self.static_visual_pad_mode)
        self.static_physical_seq_len = self.static_real_seq_len + self.static_pad_tokens
        if self.boundary == "static_visual":
            abs_pos_embed = build_static_abs_pos_embed(model, image_grid_thw, device=device)
            rope_cos, rope_sin = build_static_vision_rope(model, image_grid_thw, device=device)
            if self.static_pad_tokens:
                abs_pos_embed = torch.cat(
                    [
                        abs_pos_embed,
                        torch.zeros(
                            self.static_pad_tokens,
                            abs_pos_embed.shape[-1],
                            device=device,
                            dtype=abs_pos_embed.dtype,
                        ),
                    ],
                    dim=0,
                ).contiguous()
                rope_cos = torch.cat(
                    [
                        rope_cos,
                        torch.ones(self.static_pad_tokens, rope_cos.shape[-1], device=device, dtype=rope_cos.dtype),
                    ],
                    dim=0,
                ).contiguous()
                rope_sin = torch.cat(
                    [
                        rope_sin,
                        torch.zeros(self.static_pad_tokens, rope_sin.shape[-1], device=device, dtype=rope_sin.dtype),
                    ],
                    dim=0,
                ).contiguous()
            self.register_buffer(
                "abs_pos_embed_const",
                abs_pos_embed,
                persistent=False,
            )
            self.register_buffer("vision_rope_cos_const", rope_cos, persistent=False)
            self.register_buffer("vision_rope_sin_const", rope_sin, persistent=False)
            self.register_buffer(
                "static_pad_attention_mask",
                build_static_pad_attention_mask(self.static_real_seq_len, self.static_pad_tokens, device=device),
                persistent=False,
            )

    def _zero_static_pad_rows(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.static_pad_tokens <= 0:
            return hidden_states
        return torch.cat(
            [
                hidden_states[: self.static_real_seq_len],
                torch.zeros_like(hidden_states[self.static_real_seq_len : self.static_physical_seq_len]),
            ],
            dim=0,
        )

    def _static_mask_padded_attention(self, attention: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states = attention.q_proj(hidden_states).view(seq_length, attention.num_heads, attention.head_dim)
        key_states = attention.k_proj(hidden_states).view(seq_length, attention.num_heads, attention.head_dim)
        value_states = attention.v_proj(hidden_states).view(seq_length, attention.num_heads, attention.head_dim)
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states,
            key_states,
            self.vision_rope_cos_const,
            self.vision_rope_sin_const,
        )
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        attention_impl = get_vision_attention_impl()
        if attention_impl == "prompt_flash_attention":
            if get_vision_prompt_fa_layout() != "bnsd":
                raise ValueError(
                    f"static_visual {self.static_visual_pad_mode} currently supports PromptFA layout bnsd only"
                )
            attn_output = vision_prompt_flash_attention_bnsd(
                query_states,
                key_states,
                value_states,
                num_heads=int(attention.num_heads),
                scale=float(attention.scaling),
                atten_mask=self.static_pad_attention_mask,
            )
        elif attention_impl == "manual":
            scores = torch.matmul(query_states, key_states.transpose(2, 3)) * attention.scaling
            scores = scores.masked_fill(self.static_pad_attention_mask, torch.finfo(scores.dtype).min)
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

    def _static_mask_padded_encoder_layer(self, encoder_layer: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_input = encoder_layer.layer_norm1(hidden_states)
        hidden_states = hidden_states + self._static_mask_padded_attention(encoder_layer.self_attn, attn_input)
        hidden_states = self._zero_static_pad_rows(hidden_states)
        hidden_states = hidden_states + encoder_layer.mlp(encoder_layer.layer_norm2(hidden_states))
        return self._zero_static_pad_rows(hidden_states)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.boundary == "static_visual":
            transformer = self.model.visual.vision_model
            embeddings_module = transformer.embeddings
            pixel_values = pixel_values.to(dtype=embeddings_module.patch_embedding.weight.dtype)
            patch_embeds = embeddings_module.patch_embedding(pixel_values)
            hidden_states = patch_embeds.flatten(-2).squeeze(-1)
            if self.static_pad_tokens:
                hidden_states = torch.cat(
                    [
                        hidden_states,
                        torch.zeros(
                            self.static_pad_tokens,
                            hidden_states.shape[-1],
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        ),
                    ],
                    dim=0,
                )
            hidden_states = hidden_states + self.abs_pos_embed_const
            position_embeddings = (self.vision_rope_cos_const, self.vision_rope_sin_const)
            for encoder_layer in transformer.encoder.layers:
                if self.static_pad_tokens:
                    hidden_states = self._static_mask_padded_encoder_layer(encoder_layer, hidden_states)
                else:
                    hidden_states = encoder_layer(hidden_states, self.cu_seqlens_const, position_embeddings)
            hidden_states = transformer.post_layernorm(hidden_states)
            return hidden_states[: self.static_real_seq_len]
        if self.boundary == "visual":
            pixel_values = pixel_values.type(self.model.visual.dtype).unsqueeze(0)
            return self.model.visual(
                pixel_values=pixel_values,
                image_grid_thw=self.image_grid_thw_const,
                cu_seqlens=self.cu_seqlens_const,
            )
        if self.boundary != "get_image_features":
            raise RuntimeError(f"unsupported vision compile boundary: {self.boundary!r}")
        return self.model.get_image_features(pixel_values, self.image_grid_thw_const)


def vision_compile_backend(name: str, device: torch.device):
    if name == "default":
        return None
    if name == "torchair":
        if device.type != "npu":
            raise ValueError("--vision-compile-backend torchair requires --device npu:0")
        torchair, CompilerConfig = import_torchair()
        config = CompilerConfig()
        return torchair.get_npu_backend(compiler_config=config)
    return name


def compile_single_crop_vision_forward(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    item: QueueInput,
    device: torch.device,
    backend_name: str,
    boundary: str,
    wrapper: SingleCropVisionFeatureModule | None = None,
    static_visual_pad_mode: str = "none",
) -> tuple[Callable[[torch.Tensor], torch.Tensor] | None, dict[str, Any]]:
    backend_name = str(backend_name)
    boundary = str(boundary)
    static_visual_pad_mode = str(static_visual_pad_mode)
    if static_visual_pad_mode not in STATIC_VISUAL_PAD_MODE_CHOICES:
        raise ValueError(f"unsupported --static-visual-pad-mode {static_visual_pad_mode!r}; choices={STATIC_VISUAL_PAD_MODE_CHOICES}")
    if boundary not in VISION_FORWARD_BOUNDARY_CHOICES:
        raise ValueError(f"unsupported --vision-forward-boundary {boundary!r}; choices={VISION_FORWARD_BOUNDARY_CHOICES}")
    if backend_name == "none":
        if boundary == "static_visual":
            if wrapper is None:
                wrapper = SingleCropVisionFeatureModule(
                    model,
                    item.image_grid_thw,
                    boundary=boundary,
                    device=device,
                    static_visual_pad_mode=static_visual_pad_mode,
                ).eval()
            return wrapper, {
                "enabled": False,
                "backend": backend_name,
                "compile_api": None,
                "boundary": boundary,
                "static_visual_pad_mode": static_visual_pad_mode,
                "static_visual_pad_tokens": int(wrapper.static_pad_tokens),
                "static_visual_real_seq_len": int(wrapper.static_real_seq_len),
                "static_visual_real_seq_mod16": int(wrapper.static_real_seq_len % 16),
                "static_visual_physical_seq_len": int(wrapper.static_physical_seq_len),
                "static_visual_physical_seq_mod16": int(wrapper.static_physical_seq_len % 16),
                "static_visual_physical_seq_mod128": int(wrapper.static_physical_seq_len % 128),
                "crop_id": str(item.entry.get("id")),
                "crop_sample_bucket": item.entry.get("crop_sample_bucket"),
                "vision_tokens": int(vision_tokens(item)),
                "image_grid_thw": tensor_grid(item),
                "cu_seqlens": [int(value) for value in wrapper.cu_seqlens_const.detach().cpu().reshape(-1).tolist()],
                "static_abs_pos_embed_shape": [int(dim) for dim in wrapper.abs_pos_embed_const.shape],
                "static_vision_rope_shape": [int(dim) for dim in wrapper.vision_rope_cos_const.shape],
                "note": "Uncompiled static_visual wrapper. Shape metadata is still hoisted out of forward.",
            }
        return None, {
            "enabled": False,
            "backend": backend_name,
            "compile_api": None,
            "boundary": boundary,
        }

    if wrapper is None:
        wrapper = SingleCropVisionFeatureModule(
            model,
            item.image_grid_thw,
            boundary=boundary,
            device=device,
            static_visual_pad_mode=static_visual_pad_mode,
        ).eval()
    compile_kwargs: dict[str, Any] = {
        "fullgraph": True,
        "dynamic": False,
    }
    backend = vision_compile_backend(backend_name, device)
    if backend is not None:
        compile_kwargs["backend"] = backend

    import torch._dynamo

    old_capture_scalar_outputs = bool(torch._dynamo.config.capture_scalar_outputs)
    torch._dynamo.config.capture_scalar_outputs = True
    maybe_sync(device)
    compile_start = time.perf_counter()
    compiled = torch.compile(wrapper, **compile_kwargs)
    maybe_sync(device)
    compile_wrapper_s = time.perf_counter() - compile_start

    return compiled, {
        "enabled": True,
        "backend": backend_name,
        "compile_api": "torch.compile",
        "boundary": boundary,
        "static_visual_pad_mode": static_visual_pad_mode,
        "static_visual_pad_tokens": int(getattr(wrapper, "static_pad_tokens", 0)),
        "static_visual_real_seq_len": int(getattr(wrapper, "static_real_seq_len", vision_tokens(item))),
        "static_visual_real_seq_mod16": int(getattr(wrapper, "static_real_seq_len", vision_tokens(item)) % 16),
        "static_visual_physical_seq_len": int(getattr(wrapper, "static_physical_seq_len", vision_tokens(item))),
        "static_visual_physical_seq_mod16": int(
            getattr(wrapper, "static_physical_seq_len", vision_tokens(item)) % 16
        ),
        "static_visual_physical_seq_mod128": int(
            getattr(wrapper, "static_physical_seq_len", vision_tokens(item)) % 128
        ),
        "fullgraph": True,
        "dynamic": False,
        "capture_scalar_outputs": True,
        "capture_scalar_outputs_previous": old_capture_scalar_outputs,
        "capture_scalar_outputs_scope": (
            "set globally for this process before constructing the compiled callable and kept enabled "
            "through first-call tracing because the vision path reads image_grid_thw with Tensor.item()."
        ),
        "compile_wrapper_s": float(compile_wrapper_s),
        "crop_id": str(item.entry.get("id")),
        "crop_sample_bucket": item.entry.get("crop_sample_bucket"),
        "vision_tokens": int(vision_tokens(item)),
        "image_grid_thw": tensor_grid(item),
        "cu_seqlens": [int(value) for value in wrapper.cu_seqlens_const.detach().cpu().reshape(-1).tolist()],
        "static_abs_pos_embed_shape": (
            [int(dim) for dim in wrapper.abs_pos_embed_const.shape] if boundary == "static_visual" else None
        ),
        "static_vision_rope_shape": (
            [int(dim) for dim in wrapper.vision_rope_cos_const.shape] if boundary == "static_visual" else None
        ),
        "note": (
            "The compiled callable is shape-specialized to exactly one selected crop. "
            "The input tensor is the already CPU-preprocessed crop patch tensor after transfer to the target device; "
            "boundary=get_image_features covers native-resolution visual encoder plus adaptive MLP projector; "
            "boundary=visual covers self.visual with only cu_seqlens precomputed outside the graph; "
            "boundary=static_visual covers patch embedding, add precomputed absolute position embeddings, "
            "encoder layers with precomputed vision RoPE, and post layernorm."
        ),
    }


@torch.inference_mode()
def run_vision_one(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    item: QueueInput,
    device: torch.device,
    vision_forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
    boundary: str = "get_image_features",
) -> torch.Tensor:
    pixel_values = item.pixel_values.to(device=device, dtype=model.visual.dtype)
    if vision_forward is not None:
        return vision_forward(pixel_values)
    if boundary == "static_visual":
        wrapper = SingleCropVisionFeatureModule(model, item.image_grid_thw, boundary=boundary, device=device).eval()
        return wrapper(pixel_values)
    if boundary == "visual":
        cu_seqlens = build_single_crop_vision_cu_seqlens(item.image_grid_thw, device=device)
        return model.visual(
            pixel_values=pixel_values.type(model.visual.dtype).unsqueeze(0),
            image_grid_thw=item.image_grid_thw,
            cu_seqlens=cu_seqlens,
        )
    if boundary != "get_image_features":
        raise ValueError(f"unsupported vision forward boundary: {boundary!r}; choices={VISION_FORWARD_BOUNDARY_CHOICES}")
    return model.get_image_features(pixel_values, item.image_grid_thw)


@torch.inference_mode()
def warmup_vision(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    inputs: list[QueueInput],
    device: torch.device,
    warmup_items: int,
    vision_forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
    boundary: str = "get_image_features",
) -> dict[str, Any]:
    count = min(max(0, int(warmup_items)), len(inputs))
    if count <= 0:
        return {"count": 0, "elapsed_s": 0.0, "item_ids": []}
    maybe_sync(device)
    start = time.perf_counter()
    outputs = []
    for item in inputs[:count]:
        outputs.append(
            run_vision_one(
                model=model,
                item=item,
                device=device,
                vision_forward=vision_forward,
                boundary=boundary,
            )
        )
    maybe_sync(device)
    elapsed = time.perf_counter() - start
    return {
        "count": int(count),
        "elapsed_s": float(elapsed),
        "item_ids": [str(item.entry.get("id")) for item in inputs[:count]],
        "projected_shapes": [[int(dim) for dim in output.shape] for output in outputs],
    }


@torch.inference_mode()
def run_sync_per_crop(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    inputs: list[QueueInput],
    device: torch.device,
    repeat_count: int = 1,
    vision_forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
    boundary: str = "get_image_features",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    for repeat_idx in range(max(1, int(repeat_count))):
        for original_idx, item in enumerate(inputs):
            idx = repeat_idx * len(inputs) + original_idx
            maybe_sync(device)
            start = time.perf_counter()
            output = run_vision_one(
                model=model,
                item=item,
                device=device,
                vision_forward=vision_forward,
                boundary=boundary,
            )
            maybe_sync(device)
            elapsed = time.perf_counter() - start
            rows.append(
                make_forward_row(
                    idx,
                    item,
                    output,
                    repeat_idx=repeat_idx,
                    original_idx=original_idx,
                    boundary=boundary,
                    elapsed_s=elapsed,
                )
            )
    total_s = time.perf_counter() - total_start
    return summarize_mode("sync_per_crop", rows=rows, total_s=total_s)


@torch.inference_mode()
def run_unsynced_loop(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    inputs: list[QueueInput],
    device: torch.device,
    repeat_count: int = 1,
    vision_forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
    boundary: str = "get_image_features",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    maybe_sync(device)
    start = time.perf_counter()
    for repeat_idx in range(max(1, int(repeat_count))):
        for original_idx, item in enumerate(inputs):
            idx = repeat_idx * len(inputs) + original_idx
            output = run_vision_one(
                model=model,
                item=item,
                device=device,
                vision_forward=vision_forward,
                boundary=boundary,
            )
            rows.append(
                make_forward_row(
                    idx,
                    item,
                    output,
                    repeat_idx=repeat_idx,
                    original_idx=original_idx,
                    boundary=boundary,
                )
            )
    maybe_sync(device)
    total_s = time.perf_counter() - start
    return summarize_mode("unsynced_loop", rows=rows, total_s=total_s)


def make_forward_row(
    idx: int,
    item: QueueInput,
    output: torch.Tensor,
    *,
    repeat_idx: int,
    original_idx: int,
    boundary: str,
    elapsed_s: float | None = None,
    profile_step: int | None = None,
) -> dict[str, Any]:
    output_token_count = int(output.shape[0])
    expected_projected_tokens = int(projected_tokens(item, merge_size=2))
    row: dict[str, Any] = {
        "idx": int(idx),
        "repeat_idx": int(repeat_idx),
        "original_idx": int(original_idx),
        "id": str(item.entry.get("id")),
        "page_index": int(item.entry.get("page_index", 0)),
        "layout_label": str(item.entry.get("layout_label", "")),
        "crop_sample_bucket": item.entry.get("crop_sample_bucket"),
        "forward_boundary": str(boundary),
        "vision_tokens": int(vision_tokens(item)),
        "output_tokens": output_token_count,
        "output_token_type": "projected_image_tokens" if boundary == "get_image_features" else "visual_tokens",
        "projected_image_tokens": output_token_count if boundary == "get_image_features" else expected_projected_tokens,
        "visual_output_tokens": output_token_count if boundary == "visual" else int(vision_tokens(item)),
        "input_tokens": int(item.input_ids.shape[1]),
        "image_grid_thw": tensor_grid(item),
    }
    if elapsed_s is not None:
        row["elapsed_s"] = float(elapsed_s)
    if profile_step is not None:
        row["profile_step"] = int(profile_step)
    return row


def summarize_mode(mode: str, *, rows: list[dict[str, Any]], total_s: float) -> dict[str, Any]:
    total_vision_tokens = int(sum(int(row.get("vision_tokens", 0)) for row in rows))
    total_projected_tokens = int(sum(int(row.get("projected_image_tokens", 0)) for row in rows))
    input_tokens = int(sum(int(row.get("input_tokens", 0)) for row in rows))
    elapsed_rows = [float(row["elapsed_s"]) for row in rows if "elapsed_s" in row]
    return {
        "mode": mode,
        "forward_boundary": rows[0].get("forward_boundary") if rows else None,
        "measurement_scope": (
            "per crop: CPU preprocessed pixel tensor -> device transfer -> selected forward boundary. "
            "get_image_features = native-resolution visual encoder + post layernorm + adaptive MLP projector; "
            "visual = self.visual only, i.e. patch embedding + interpolated position embedding + encoder + post layernorm."
        ),
        "sync_policy": (
            "device synchronize before and after every crop"
            if mode == "sync_per_crop"
            else "one device synchronize before the loop and one after the full loop; no per-crop sync"
        ),
        "count": int(len(rows)),
        "total_s": float(total_s),
        "items_per_s": tok_per_s(len(rows), total_s),
        "vision_tokens_per_s": tok_per_s(total_vision_tokens, total_s),
        "projected_image_tokens_per_s": tok_per_s(total_projected_tokens, total_s),
        "input_tokens_per_s": tok_per_s(input_tokens, total_s),
        "total_vision_tokens": int(total_vision_tokens),
        "total_projected_image_tokens": int(total_projected_tokens),
        "total_input_tokens": int(input_tokens),
        "per_crop_elapsed_s": stats(elapsed_rows) if elapsed_rows else None,
        "samples": rows[:16],
    }


@torch.inference_mode()
def run_mode(
    mode: str,
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    inputs: list[QueueInput],
    device: torch.device,
    repeat_count: int = 1,
    vision_forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
    boundary: str = "get_image_features",
) -> dict[str, Any]:
    if mode == "sync_per_crop":
        return run_sync_per_crop(
            model=model,
            inputs=inputs,
            device=device,
            repeat_count=repeat_count,
            vision_forward=vision_forward,
            boundary=boundary,
        )
    if mode == "unsynced_loop":
        return run_unsynced_loop(
            model=model,
            inputs=inputs,
            device=device,
            repeat_count=repeat_count,
            vision_forward=vision_forward,
            boundary=boundary,
        )
    raise AssertionError(mode)


@torch.inference_mode()
def profile_vision_mode(
    *,
    profile_root: Path,
    profile_mode: str,
    profile_metric: str,
    model: LocalPaddleOCRVLForConditionalGeneration,
    inputs: list[QueueInput],
    device: torch.device,
    dtype: torch.dtype,
    warmup_repeats: int,
    active_repeats: int,
    vision_forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
    boundary: str = "get_image_features",
) -> dict[str, Any]:
    if device.type != "npu":
        raise ValueError("--profile-dir requires --device npu:0; torch_npu profiler is NPU-only.")

    import torch_npu.profiler as npu_prof

    profile_run_dir = make_profile_run_dir(
        profile_root.expanduser().resolve(),
        mode=profile_mode,
        attention=get_vision_attention_impl(),
        dtype=dtype,
        crop_count=len(inputs),
        metric=profile_metric,
    )
    shutil.rmtree(profile_run_dir, ignore_errors=True)
    profile_run_dir.mkdir(parents=True, exist_ok=True)

    warmup_repeats = max(0, int(warmup_repeats))
    active_repeats = max(1, int(active_repeats))
    active_steps = active_repeats * len(inputs)

    maybe_sync(device)
    warmup_start = time.perf_counter()
    warmup_forward_count = 0
    for _ in range(warmup_repeats):
        for item in inputs:
            run_vision_one(
                model=model,
                item=item,
                device=device,
                vision_forward=vision_forward,
                boundary=boundary,
            )
            warmup_forward_count += 1
    maybe_sync(device)
    profiler_warmup_s = time.perf_counter() - warmup_start

    schedule = npu_prof.schedule(wait=0, warmup=0, active=active_steps, repeat=1)
    rows: list[dict[str, Any]] = []
    maybe_sync(device)
    profile_context_start = time.perf_counter()
    active_loop_start: float | None = None
    active_loop_end: float | None = None
    profiler_step_times_s: list[float] = []
    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        schedule=schedule,
        experimental_config=npu_profiler_config(profile_metric),
        on_trace_ready=npu_prof.tensorboard_trace_handler(str(profile_run_dir), analyse_flag=True),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    ) as profiler:
        profile_step = 0
        active_loop_start = time.perf_counter()
        for repeat_idx in range(active_repeats):
            for original_idx, item in enumerate(inputs):
                idx = repeat_idx * len(inputs) + original_idx
                bucket = str(item.entry.get("crop_sample_bucket") or "crop")
                with torch.profiler.record_function(f"paddle_ocr_vl.vision_prefill_profile.{bucket}"):
                    step_start = time.perf_counter()
                    output = run_vision_one(
                        model=model,
                        item=item,
                        device=device,
                        vision_forward=vision_forward,
                        boundary=boundary,
                    )
                    maybe_sync(device)
                    elapsed = time.perf_counter() - step_start
                rows.append(
                    make_forward_row(
                        idx,
                        item,
                        output,
                        repeat_idx=repeat_idx,
                        original_idx=original_idx,
                        boundary=boundary,
                        elapsed_s=elapsed,
                        profile_step=profile_step,
                    )
                )
                profiler_step_start = time.perf_counter()
                profiler.step()
                profiler_step_times_s.append(float(time.perf_counter() - profiler_step_start))
                profile_step += 1
        active_loop_end = time.perf_counter()
    maybe_sync(device)
    profile_context_wall_s = time.perf_counter() - profile_context_start

    forward_sync_s = float(sum(float(row.get("elapsed_s", 0.0)) for row in rows))
    profiler_step_s = float(sum(profiler_step_times_s))
    active_loop_wall_s = float((active_loop_end or profile_context_start) - (active_loop_start or profile_context_start))
    context_non_active_s = float(max(0.0, profile_context_wall_s - active_loop_wall_s))
    active_loop_unattributed_s = float(max(0.0, active_loop_wall_s - forward_sync_s - profiler_step_s))
    profiled_mode_context_result = summarize_mode("profile_context_wall", rows=rows, total_s=profile_context_wall_s)
    profiled_mode_forward_sync_result = summarize_mode("profile_forward_sync_only", rows=rows, total_s=forward_sync_s)

    summary = {
        "enabled": True,
        "profile_dir": str(profile_run_dir),
        "profile_mode": str(profile_mode),
        "profile_metric": str(profile_metric),
        "forward_boundary": str(boundary),
        "scope": (
            "same selected real OCR crops as the benchmark mode; CPU preprocessed pixel tensor -> device transfer "
            "-> selected forward boundary. get_image_features includes the adaptive MLP projector; visual stops after "
            "self.visual post layernorm."
        ),
        "post_warmup": True,
        "with_stack": True,
        "record_shapes": True,
        "profile_memory": False,
        "profiler_step_contract": "one profiler.step() is called after exactly one model.get_image_features() crop forward",
        "profile_warmup_repeats": int(warmup_repeats),
        "profile_active_repeats": int(active_repeats),
        "profile_warmup_forward_count": int(warmup_forward_count),
        "profile_active_steps": int(active_steps),
        "profile_warmup_s": float(profiler_warmup_s),
        "profile_context_wall_s": float(profile_context_wall_s),
        "profile_active_loop_wall_s": float(active_loop_wall_s),
        "profile_forward_sync_sum_s": float(forward_sync_s),
        "profile_profiler_step_sum_s": float(profiler_step_s),
        "profile_context_non_active_s": float(context_non_active_s),
        "profile_active_loop_unattributed_s": float(active_loop_unattributed_s),
        "profile_profiler_step_s": stats(profiler_step_times_s),
        "profiled_context_wall_result": profiled_mode_context_result,
        "profiled_forward_sync_result": profiled_mode_forward_sync_result,
        "profiled_mode_result": profiled_mode_context_result,
        "profiled_mode_result_note": (
            "Deprecated compatibility field. This uses full profiler context wall time and may include "
            "profiler.step(), trace handling, and export/finalization overhead. Prefer profiled_forward_sync_result "
            "for forward+sync timing and profile_profiler_step_sum_s / profile_context_non_active_s for overhead."
        ),
    }
    summary_path = profile_run_dir / "vision_prefill_profile_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--page-start", type=int, default=0)
    parser.add_argument("--num-pages", type=int, default=8)
    parser.add_argument("--layout-source", default="omnidocbench_gt", choices=["omnidocbench_gt"])
    parser.add_argument("--crop-padding", type=int, default=0)
    parser.add_argument("--min-crop-side", type=int, default=4)
    parser.add_argument("--skip-labels", default="")
    parser.add_argument("--include-ignored-gt", action="store_true")
    parser.add_argument("--include-empty-gt", action="store_true")
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--preprocessor-min-pixels",
        type=int,
        default=-1,
        help="Override preprocessor_config min_pixels for controlled crop-resolution vision benchmarks. -1 uses model config.",
    )
    parser.add_argument(
        "--preprocessor-max-pixels",
        type=int,
        default=-1,
        help="Override preprocessor_config max_pixels for controlled crop-resolution vision benchmarks. -1 uses model config.",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "fp32", "float32", "bf16", "bfloat16"])
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--vision-attention", default=os.environ.get(VISION_ATTENTION_ENV, "manual"), choices=VISION_ATTENTION_CHOICES)
    parser.add_argument(
        "--vision-prompt-fa-layout",
        default=os.environ.get(VISION_PROMPT_FA_LAYOUT_ENV, "bnsd"),
        choices=VISION_PROMPT_FA_LAYOUT_CHOICES,
    )
    parser.add_argument(
        "--modes",
        default="sync_per_crop,unsynced_loop",
        help="Comma/space separated modes. Choices: sync_per_crop, unsynced_loop.",
    )
    parser.add_argument("--warmup-items", type=int, default=1)
    parser.add_argument(
        "--crop-sample",
        default="all",
        choices=CROP_SAMPLE_CHOICES,
        help="Optionally reduce real OCR crops to representative size buckets before speed/profiler runs.",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=1,
        help="Repeat the selected crop list this many times for non-profiled benchmark modes.",
    )
    parser.add_argument(
        "--vision-compile-backend",
        default="none",
        choices=VISION_COMPILE_BACKEND_CHOICES,
        help=(
            "Compile the selected single-crop --vision-forward-boundary before benchmark/profiler runs. "
            "Requires --crop-sample small_only because the graph is shape-specialized to one crop."
        ),
    )
    parser.add_argument(
        "--vision-forward-boundary",
        default="get_image_features",
        choices=VISION_FORWARD_BOUNDARY_CHOICES,
        help=(
            "Forward boundary for both eager and compiled vision tests. get_image_features includes self.visual plus "
            "the adaptive MLP projector. visual compiles/runs self.visual with only cu_seqlens precomputed outside "
            "the graph. static_visual compiles/runs a single-crop static rewrite of self.visual with crop-grid "
            "constants, absolute position embeddings, and vision RoPE hoisted out of forward."
        ),
    )
    parser.add_argument(
        "--static-visual-pad-mode",
        default=os.environ.get("STATIC_VISUAL_PAD_MODE", "none"),
        choices=STATIC_VISUAL_PAD_MODE_CHOICES,
        help=(
            "Diagnostic workaround for boundary=static_visual. none preserves the exact old static wrapper. "
            "mask_pad_one appends one physical dummy token only when seq_len %% 16 == 0, matching the first "
            "successful compiled-visual downstream probe. mask_pad_to_128 pads seq_len %% 16 == 0 cases to "
            "the next CANN-friendly physical length (for S > 128, 128-aligned). Both masked modes block "
            "attention between real and dummy tokens with a BOOL mask, zero dummy rows between layers, and "
            "crop back to the original real token count before returning."
        ),
    )
    parser.add_argument(
        "--vision-compile-validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare the first compiled vision output against eager for the selected crop before timed runs.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="Optional NPU profiler output root. Captures one post-warmup run of --profile-mode.",
    )
    parser.add_argument(
        "--profile-mode",
        default="unsynced_loop",
        choices=MODE_CHOICES,
        help="Benchmark mode to rerun under torch_npu.profiler when --profile-dir is set.",
    )
    parser.add_argument(
        "--profile-metric",
        default="pipe",
        choices=PROFILE_METRIC_CHOICES,
        help="torch_npu profiler AiC metric for --profile-dir captures.",
    )
    parser.add_argument(
        "--profile-warmup-repeats",
        type=int,
        default=2,
        help="Profiler-only warmup repeats over selected crops before entering torch_npu.profiler.",
    )
    parser.add_argument(
        "--profile-active-repeats",
        type=int,
        default=3,
        help="Profiler active repeats over selected crops. One profiler step is one crop forward.",
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=0,
        help="Optional development cap after crop extraction/input preprocessing. 0 means all selected crops.",
    )
    parser.add_argument(
        "--crop-ids",
        default="",
        help=(
            "Comma/space separated crop ids to keep after preprocessing. Useful for shape-specialized compiled "
            "static_visual tests that must compare the same real crop across preprocessor settings."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if int(args.num_pages) <= 0:
        raise ValueError("--num-pages must be positive")
    modes = parse_modes(args.modes)
    if args.profile_dir is not None and str(args.profile_mode) not in modes:
        modes.append(str(args.profile_mode))

    os.environ[VISION_ATTENTION_ENV] = str(args.vision_attention)
    os.environ[VISION_PROMPT_FA_LAYOUT_ENV] = str(args.vision_prompt_fa_layout)

    model_dir = _resolve_model_dir(args.model)
    device = resolve_device(args.device)
    dtype = parse_vision_dtype(args.dtype)
    vision_forward_boundary = str(args.vision_forward_boundary)
    configure_npu_jit_compile(args.npu_jit_compile, device)

    page_load = load_pages_result(
        args.dataset_dir,
        page_start=int(args.page_start),
        num_pages=int(args.num_pages),
    )
    pages = page_load.pages

    layout_pages, layout_timing = build_omnidocbench_gt_layout_pages(
        pages,
        include_ignored=bool(args.include_ignored_gt),
        include_empty_gt=bool(args.include_empty_gt),
    )
    crops, crop_summary, crop_timing = build_detected_crops(pages=pages, layout_pages=layout_pages, args=args)
    if not crops:
        raise RuntimeError("OmniDocBench GT layout produced zero recognizer crops")
    raw_extracted_crop_count = int(len(crops))
    if int(args.max_crops) > 0:
        crops = crops[: int(args.max_crops)]
    if not crops:
        raise RuntimeError("zero crops after --max-crops")
    crop_ids = parse_crop_ids(str(args.crop_ids))
    crops, crop_id_filter_summary = filter_crops_by_crop_ids(crops, crop_ids)
    if not crops:
        raise RuntimeError("zero crops after --crop-ids")

    original_pre_cfg = load_preprocessor_config(model_dir)
    pre_cfg = dict(original_pre_cfg)
    if int(args.preprocessor_min_pixels) >= 0:
        pre_cfg["min_pixels"] = int(args.preprocessor_min_pixels)
    if int(args.preprocessor_max_pixels) >= 0:
        pre_cfg["max_pixels"] = int(args.preprocessor_max_pixels)
    if int(pre_cfg["min_pixels"]) > int(pre_cfg["max_pixels"]):
        raise ValueError(
            f"preprocessor min_pixels {pre_cfg['min_pixels']} exceeds max_pixels {pre_cfg['max_pixels']}"
        )
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    queue_inputs, input_build_summary = build_queue_inputs_from_crops(
        crops=crops,
        tokenizer=tokenizer,
        pre_cfg=pre_cfg,
        prompt_override=args.prompt,
    )
    if not queue_inputs:
        raise RuntimeError("zero queue inputs after --max-crops")
    raw_queue_input_count_before_crop_sample = int(len(queue_inputs))
    crop_id_filter_summary = add_queue_input_details_to_crop_id_filter(crop_id_filter_summary, queue_inputs)
    queue_inputs, crop_sample_summary = select_profile_crop_sample(queue_inputs, strategy=str(args.crop_sample))
    if not queue_inputs:
        raise RuntimeError("zero queue inputs after --crop-sample")

    setup_timing: dict[str, float] = {}
    maybe_sync(device)
    start = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device=device)
    maybe_sync(device)
    setup_timing["recognizer_model_load_s"] = time.perf_counter() - start

    vision_forward: Callable[[torch.Tensor], torch.Tensor] | None = None
    vision_compile_summary: dict[str, Any]
    if str(args.vision_compile_backend) != "none" and len(queue_inputs) != 1:
        raise ValueError(
            "--vision-compile-backend is shape-specialized and currently requires exactly one selected crop. "
            "Use --crop-sample small_only for the requested small-crop compile/profiler experiment."
        )
    if str(args.vision_compile_backend) != "none":
        target_item = queue_inputs[0]
        eager_ref: torch.Tensor | None = None
        validation_wrapper: SingleCropVisionFeatureModule | None = None
        validation_pixel_values: torch.Tensor | None = None
        original_visual_ref: torch.Tensor | None = None
        if bool(args.vision_compile_validate):
            validation_wrapper = SingleCropVisionFeatureModule(
                model,
                target_item.image_grid_thw,
                boundary=vision_forward_boundary,
                device=device,
                static_visual_pad_mode=str(args.static_visual_pad_mode),
            ).eval()
            validation_pixel_values = target_item.pixel_values.to(device=device, dtype=model.visual.dtype)
            maybe_sync(device)
            eager_start = time.perf_counter()
            eager_ref = validation_wrapper(validation_pixel_values)
            maybe_sync(device)
            eager_ref_s = time.perf_counter() - eager_start
            if vision_forward_boundary == "static_visual":
                maybe_sync(device)
                original_visual_start = time.perf_counter()
                original_visual_ref = run_vision_one(
                    model=model,
                    item=target_item,
                    device=device,
                    boundary="visual",
                )
                maybe_sync(device)
                original_visual_ref_s = time.perf_counter() - original_visual_start
            else:
                original_visual_ref_s = None
        else:
            eager_ref_s = None
            original_visual_ref_s = None

        vision_forward, vision_compile_summary = compile_single_crop_vision_forward(
            model=model,
            item=target_item,
            device=device,
            backend_name=str(args.vision_compile_backend),
            boundary=vision_forward_boundary,
            wrapper=validation_wrapper,
            static_visual_pad_mode=str(args.static_visual_pad_mode),
        )
        maybe_sync(device)
        first_call_start = time.perf_counter()
        if validation_pixel_values is not None:
            compiled_first = vision_forward(validation_pixel_values)
        else:
            compiled_first = run_vision_one(model=model, item=target_item, device=device, vision_forward=vision_forward)
        maybe_sync(device)
        compiled_first_call_s = time.perf_counter() - first_call_start
        vision_compile_summary["compiled_first_call_s"] = float(compiled_first_call_s)
        if eager_ref is not None:
            maybe_sync(device)
            second_call_start = time.perf_counter()
            compiled_second = vision_forward(validation_pixel_values)
            maybe_sync(device)
            compiled_second_call_s = time.perf_counter() - second_call_start

            maybe_sync(device)
            post_compile_eager_start = time.perf_counter()
            post_compile_eager = validation_wrapper(validation_pixel_values)
            maybe_sync(device)
            post_compile_eager_s = time.perf_counter() - post_compile_eager_start

            diff = (eager_ref.float() - compiled_first.float()).abs()
            repeat_diff = (compiled_first.float() - compiled_second.float()).abs()
            post_compile_eager_diff = (eager_ref.float() - post_compile_eager.float()).abs()
            vision_compile_summary["validation"] = {
                "enabled": True,
                "reference_scope": "same_wrapper_same_input_tensor",
                "input_shape": [int(dim) for dim in validation_pixel_values.shape],
                "input_dtype": str(validation_pixel_values.dtype),
                "eager_reference_s": float(eager_ref_s),
                "max_abs_diff": float(diff.max().detach().cpu().item()),
                "mean_abs_diff": float(diff.mean().detach().cpu().item()),
                "allclose_atol_5e_2_rtol_5e_2": bool(
                    torch.allclose(eager_ref.float(), compiled_first.float(), atol=5e-2, rtol=5e-2)
                ),
                "eager_shape": [int(dim) for dim in eager_ref.shape],
                "compiled_shape": [int(dim) for dim in compiled_first.shape],
                "compiled_second_call_s": float(compiled_second_call_s),
                "compiled_first_vs_second": {
                    "max_abs_diff": float(repeat_diff.max().detach().cpu().item()),
                    "mean_abs_diff": float(repeat_diff.mean().detach().cpu().item()),
                    "allclose_atol_5e_2_rtol_5e_2": bool(
                        torch.allclose(compiled_first.float(), compiled_second.float(), atol=5e-2, rtol=5e-2)
                    ),
                },
                "same_wrapper_eager_before_vs_after_compile": {
                    "post_compile_eager_s": float(post_compile_eager_s),
                    "max_abs_diff": float(post_compile_eager_diff.max().detach().cpu().item()),
                    "mean_abs_diff": float(post_compile_eager_diff.mean().detach().cpu().item()),
                    "allclose_atol_5e_2_rtol_5e_2": bool(
                        torch.allclose(eager_ref.float(), post_compile_eager.float(), atol=5e-2, rtol=5e-2)
                    ),
                },
            }
            if original_visual_ref is not None:
                static_diff = (original_visual_ref.float() - eager_ref.float()).abs()
                vision_compile_summary["validation"]["static_visual_vs_original_visual"] = {
                    "original_visual_reference_s": float(original_visual_ref_s),
                    "max_abs_diff": float(static_diff.max().detach().cpu().item()),
                    "mean_abs_diff": float(static_diff.mean().detach().cpu().item()),
                    "allclose_atol_5e_2_rtol_5e_2": bool(
                        torch.allclose(original_visual_ref.float(), eager_ref.float(), atol=5e-2, rtol=5e-2)
                    ),
                    "original_visual_shape": [int(dim) for dim in original_visual_ref.shape],
                    "static_visual_shape": [int(dim) for dim in eager_ref.shape],
                }
        else:
            vision_compile_summary["validation"] = {"enabled": False}
        setup_timing["vision_compile_wrapper_s"] = float(vision_compile_summary.get("compile_wrapper_s", 0.0))
        setup_timing["vision_compile_first_call_s"] = float(compiled_first_call_s)
    else:
        vision_forward, vision_compile_summary = compile_single_crop_vision_forward(
            model=model,
            item=queue_inputs[0],
            device=device,
            backend_name="none",
            boundary=vision_forward_boundary,
            static_visual_pad_mode=str(args.static_visual_pad_mode),
        )

    warmup = warmup_vision(
        model=model,
        inputs=queue_inputs,
        device=device,
        warmup_items=int(args.warmup_items),
        vision_forward=vision_forward,
        boundary=vision_forward_boundary,
    )

    mode_results: dict[str, Any] = {}
    for mode in modes:
        mode_results[mode] = run_mode(
            mode,
            model=model,
            inputs=queue_inputs,
            device=device,
            repeat_count=max(1, int(args.benchmark_repeats)),
            vision_forward=vision_forward,
            boundary=vision_forward_boundary,
        )

    comparisons: dict[str, Any] = {}
    if "sync_per_crop" in mode_results and "unsynced_loop" in mode_results:
        sync_s = float(mode_results["sync_per_crop"]["total_s"])
        unsynced_s = float(mode_results["unsynced_loop"]["total_s"])
        comparisons["unsynced_vs_sync_per_crop"] = {
            "speedup": (sync_s / unsynced_s) if unsynced_s > 0 else None,
            "saved_s": float(sync_s - unsynced_s),
            "sync_per_crop_total_s": sync_s,
            "unsynced_loop_total_s": unsynced_s,
            "note": (
                "This isolates per-crop device synchronization overhead for the same crop/preprocess/model path. "
                "Run order and warmup are reported because first-use kernel behavior can still affect small runs."
            ),
        }

    profiler_summary: dict[str, Any] = {"enabled": False}
    if args.profile_dir is not None:
        profiler_summary = profile_vision_mode(
            profile_root=args.profile_dir,
            profile_mode=str(args.profile_mode),
            profile_metric=str(args.profile_metric),
            model=model,
            inputs=queue_inputs,
            device=device,
            dtype=dtype,
            warmup_repeats=int(args.profile_warmup_repeats),
            active_repeats=int(args.profile_active_repeats),
            vision_forward=vision_forward,
            boundary=vision_forward_boundary,
        )
        baseline = mode_results[str(args.profile_mode)]
        profiled_context = profiler_summary["profiled_context_wall_result"]
        profiled_forward_sync = profiler_summary["profiled_forward_sync_result"]
        baseline_total_s = float(baseline["total_s"])
        profiled_context_total_s = float(profiled_context["total_s"])
        profiled_forward_sync_total_s = float(profiled_forward_sync["total_s"])
        baseline_items_per_s = float(baseline["items_per_s"])
        profiled_context_items_per_s = float(profiled_context["items_per_s"])
        profiled_forward_sync_items_per_s = float(profiled_forward_sync["items_per_s"])
        baseline_vision_tokens_per_s = float(baseline["vision_tokens_per_s"])
        profiled_context_vision_tokens_per_s = float(profiled_context["vision_tokens_per_s"])
        profiled_forward_sync_vision_tokens_per_s = float(profiled_forward_sync["vision_tokens_per_s"])
        baseline_forward_count = int(baseline["count"])
        profiled_forward_count = int(profiled_context["count"])
        comparisons[f"profiled_vs_unprofiled_{args.profile_mode}"] = {
            "mode": str(args.profile_mode),
            "baseline_forward_count": baseline_forward_count,
            "profiled_forward_count": profiled_forward_count,
            "forward_count_match": bool(baseline_forward_count == profiled_forward_count),
            "baseline_total_s": baseline_total_s,
            "profiled_context_total_s": profiled_context_total_s,
            "profiled_forward_sync_total_s": profiled_forward_sync_total_s,
            "profiled_context_total_s_over_baseline": (
                profiled_context_total_s / baseline_total_s if baseline_total_s > 0 else None
            ),
            "profiled_forward_sync_total_s_over_baseline": (
                profiled_forward_sync_total_s / baseline_total_s if baseline_total_s > 0 else None
            ),
            "baseline_items_per_s": baseline_items_per_s,
            "profiled_context_items_per_s": profiled_context_items_per_s,
            "profiled_forward_sync_items_per_s": profiled_forward_sync_items_per_s,
            "profiled_context_items_per_s_over_baseline": (
                profiled_context_items_per_s / baseline_items_per_s if baseline_items_per_s > 0 else None
            ),
            "profiled_forward_sync_items_per_s_over_baseline": (
                profiled_forward_sync_items_per_s / baseline_items_per_s if baseline_items_per_s > 0 else None
            ),
            "baseline_vision_tokens_per_s": baseline_vision_tokens_per_s,
            "profiled_context_vision_tokens_per_s": profiled_context_vision_tokens_per_s,
            "profiled_forward_sync_vision_tokens_per_s": profiled_forward_sync_vision_tokens_per_s,
            "profiled_context_vision_tokens_per_s_over_baseline": (
                profiled_context_vision_tokens_per_s / baseline_vision_tokens_per_s
                if baseline_vision_tokens_per_s > 0
                else None
            ),
            "profiled_forward_sync_vision_tokens_per_s_over_baseline": (
                profiled_forward_sync_vision_tokens_per_s / baseline_vision_tokens_per_s
                if baseline_vision_tokens_per_s > 0
                else None
            ),
            "profile_context_wall_s": profiler_summary.get("profile_context_wall_s"),
            "profile_active_loop_wall_s": profiler_summary.get("profile_active_loop_wall_s"),
            "profile_forward_sync_sum_s": profiler_summary.get("profile_forward_sync_sum_s"),
            "profile_profiler_step_sum_s": profiler_summary.get("profile_profiler_step_sum_s"),
            "profile_context_non_active_s": profiler_summary.get("profile_context_non_active_s"),
            "profile_active_loop_unattributed_s": profiler_summary.get("profile_active_loop_unattributed_s"),
            "profile_dir": profiler_summary.get("profile_dir"),
            "profile_metric": str(args.profile_metric),
            "benchmark_repeats": int(args.benchmark_repeats),
            "profile_warmup_repeats": int(args.profile_warmup_repeats),
            "profile_active_repeats": int(args.profile_active_repeats),
            "profiler_step_contract": profiler_summary.get("profiler_step_contract"),
            "note": (
                "Context timings include profiler.step(), trace handling, and possible export/finalization overhead. "
                "Forward-sync timings sum only the measured crop forward + device sync windows inside the profiler. "
                "Set --benchmark-repeats equal to --profile-active-repeats for the cleanest forward-count comparison."
            ),
        }

    output = {
        "experiment": "06_vision_prefill_only",
        "scope": (
            "full pages -> OmniDocBench GT layout boxes -> real OCR crops -> PaddleOCR-VL crop preprocessing "
            "-> vision device transfer + native-resolution visual encoder + adaptive MLP projector only"
        ),
        "not_run": [
            "document layout detector",
            "text token embedding",
            "image embedding scatter into text sequence",
            "mRoPE index construction",
            "KV cache allocation/prefill",
            "LM head prefill",
            "text decode/hot-swap",
            "OCR text validation or accuracy scoring",
        ],
        "model": str(model_dir),
        "dataset_dir": str(resolve_dataset_dir(args.dataset_dir)),
        "device": str(device),
        "dtype": str(dtype),
        "npu_jit_compile": str(args.npu_jit_compile),
        "vision_attention": get_vision_attention_impl(),
        "vision_prompt_fa_layout": get_vision_prompt_fa_layout(),
        "vision_forward_boundary": vision_forward_boundary,
        "page_start": int(args.page_start),
        "page_count": int(len(pages)),
        "page_load": page_load_summary(page_load),
        "layout": {
            "source": "omnidocbench_gt",
            "uses_ground_truth_boxes": True,
            "include_ignored_gt": bool(args.include_ignored_gt),
            "include_empty_gt": bool(args.include_empty_gt),
        },
        "recognizer_crop_count": int(len(queue_inputs)),
        "raw_queue_input_count_before_crop_sample": int(raw_queue_input_count_before_crop_sample),
        "raw_extracted_crop_count_before_max_crops": int(raw_extracted_crop_count),
        "max_crops": None if int(args.max_crops) <= 0 else int(args.max_crops),
        "crop_id_filter": crop_id_filter_summary,
        "crop_sample": crop_sample_summary,
        "benchmark_repeats": int(args.benchmark_repeats),
        "prompt_override": args.prompt,
        "crop_summary": crop_summary,
        "preprocessor": {
            "min_pixels": int(pre_cfg["min_pixels"]),
            "max_pixels": int(pre_cfg["max_pixels"]),
            "patch_size": int(pre_cfg["patch_size"]),
            "merge_size": int(pre_cfg["merge_size"]),
            "temporal_patch_size": int(pre_cfg["temporal_patch_size"]),
            "original_min_pixels": int(original_pre_cfg["min_pixels"]),
            "original_max_pixels": int(original_pre_cfg["max_pixels"]),
            "override_min_pixels": None if int(args.preprocessor_min_pixels) < 0 else int(args.preprocessor_min_pixels),
            "override_max_pixels": None if int(args.preprocessor_max_pixels) < 0 else int(args.preprocessor_max_pixels),
            "min_projected_image_tokens": int(pre_cfg["min_pixels"])
            // int(pre_cfg["patch_size"] * pre_cfg["merge_size"]) ** 2,
        },
        "input_summary": summarize_inputs(queue_inputs, merge_size=int(pre_cfg["merge_size"])),
        "layout_timing_s": layout_timing,
        "crop_timing_s": crop_timing,
        "input_build_summary_s": input_build_summary,
        "queue_input_timing_summary_s": aggregate_timing_dicts([item.timing_s for item in queue_inputs]),
        "setup_timing_s": setup_timing,
        "vision_compile": vision_compile_summary,
        "static_visual_pad_mode": str(args.static_visual_pad_mode),
        "warmup": warmup,
        "mode_order": modes,
        "mode_order_note": (
            "Modes run sequentially in the listed order after warmup. For tiny runs, rerun once or set "
            "--warmup-items high enough to avoid first-use effects."
        ),
        "modes": mode_results,
        "profiler": profiler_summary,
        "comparisons": comparisons,
    }

    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True, default=json_default))
    else:
        summary = {
            "experiment": output["experiment"],
            "device": output["device"],
            "vision_attention": output["vision_attention"],
            "page_count": output["page_count"],
            "recognizer_crop_count": output["recognizer_crop_count"],
            "comparisons": comparisons,
            "modes": {
                key: {
                    "total_s": value["total_s"],
                    "items_per_s": value["items_per_s"],
                    "vision_tokens_per_s": value["vision_tokens_per_s"],
                }
                for key, value in mode_results.items()
            },
        }
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
