#!/usr/bin/env python3
"""Inline single-layer PaddleOCR-VL vision TorchAir repro.

This is intentionally separate from vision_prefill_bench.py. It loads one real
baseline crop, builds the same static padded visual input, runs one inline
vision transformer layer eagerly and under torch.compile, then compares every
intermediate tensor returned by that one compiled graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from local_modeling_paddleocr_vl import (
    LocalPaddleOCRVLForConditionalGeneration,
    _activation,
    _resolve_model_dir,
    apply_rotary_pos_emb_vision,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = os.environ.get("MODEL", "/home/lukaiv/models/paddle_ocr_0_9b_v_1_6")
DEFAULT_BASELINE = SCRIPT_DIR / "baselines" / "promptfa_fp16_eager_64"
STAGE_NAMES = (
    "patch_flat",
    "patch_pad",
    "patch_pos",
    "ln1",
    "qkv",
    "qk_rope_v",
    "q_bnsd",
    "k_bnsd",
    "v_bnsd",
    "attn_kernel_bnsd",
    "attn_kernel_out",
    "attn_out_proj",
    "attn_residual",
    "ln2",
    "mlp_fc1",
    "mlp_act",
    "mlp_fc2",
    "layer0_out",
)


def parse_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name!r}")


def resolve_device(name: str) -> torch.device:
    if name.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("NPU device requested, but torch_npu is not importable.") from exc
    return torch.device(name)


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        import torch_npu

        torch_npu.npu.synchronize()


def configure_npu_jit_compile(mode: str, device: torch.device) -> None:
    if device.type != "npu" or mode == "default":
        return
    import torch_npu  # noqa: F401

    torch.npu.set_compile_mode(jit_compile=(mode == "on"))
    print(f"[npu] set torch.npu compile mode jit_compile={mode == 'on'}", file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def u_escape_path_component(value: str, *, uppercase_hex: bool = False) -> str:
    escaped: list[str] = []
    for char in str(value):
        code = ord(char)
        if code < 128:
            escaped.append(char)
        else:
            fmt = "04X" if uppercase_hex else "04x"
            escaped.append(f"#U{code:{fmt}}")
    return "".join(escaped)


def u_escape_relative_path(rel: str, *, uppercase_hex: bool = False) -> Path:
    path = Path(str(rel))
    return Path(*(u_escape_path_component(part, uppercase_hex=uppercase_hex) for part in path.parts))


def resolve_page_image_path(item: dict[str, Any], dataset_dir: Path | None) -> Path:
    source_image = Path(str(item.get("source_image", ""))).expanduser()
    if source_image.is_file():
        return source_image.resolve()

    rel = str(item.get("image_rel", ""))
    if not rel:
        raise FileNotFoundError(f"manifest item {item.get('id')} has no usable source_image or image_rel")
    candidates: list[Path] = []
    if dataset_dir is not None:
        images_dir = dataset_dir / "images"
        candidates.extend(
            [
                images_dir / rel,
                images_dir / u_escape_relative_path(rel, uppercase_hex=False),
                images_dir / u_escape_relative_path(rel, uppercase_hex=True),
            ]
        )
    candidates.append(Path(rel).expanduser())
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    raise FileNotFoundError(
        f"could not resolve page image for item {item.get('id')}; candidates="
        + ", ".join(str(path) for path in candidates)
    )


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    if height < factor:
        width = round((width * factor) / height)
        height = factor
    if width < factor:
        height = round((height * factor) / width)
        width = factor
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def load_preprocessor_config(model_dir: Path) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "max_pixels": 1003520,
        "merge_size": 2,
        "min_pixels": 112896,
        "patch_size": 14,
        "resample": 3,
        "rescale_factor": 1.0 / 255.0,
        "temporal_patch_size": 1,
    }
    path = model_dir / "preprocessor_config.json"
    if path.exists():
        cfg.update(json.loads(path.read_text(encoding="utf-8")))
    return cfg


def preprocess_crop(image: Image.Image, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if cfg["do_convert_rgb"]:
        image = image.convert("RGB")
    width, height = image.size
    patch_size = int(cfg["patch_size"])
    merge_size = int(cfg["merge_size"])
    temporal_patch_size = int(cfg["temporal_patch_size"])
    if temporal_patch_size != 1:
        raise ValueError(f"temporal_patch_size must be 1, got {temporal_patch_size}")
    resized_height, resized_width = height, width
    if cfg["do_resize"]:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=int(cfg["min_pixels"]),
            max_pixels=int(cfg["max_pixels"]),
        )
        image = image.resize((resized_width, resized_height), Image.Resampling(int(cfg["resample"])))
    array = np.asarray(image)
    if cfg["do_rescale"]:
        array = array.astype(np.float32) * float(cfg["rescale_factor"])
    else:
        array = array.astype(np.float32)
    if cfg["do_normalize"]:
        mean = np.array(cfg["image_mean"], dtype=np.float32)
        std = np.array(cfg["image_std"], dtype=np.float32)
        array = (array - mean) / std
    patches = array.transpose(2, 0, 1)[None, ...]
    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size
    patches = patches.reshape(grid_t, temporal_patch_size, channel, grid_h, patch_size, grid_w, patch_size)
    patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
    flatten_patches = patches.reshape(grid_t * grid_h * grid_w, channel, patch_size, patch_size)
    meta = {
        "crop_size": [int(width), int(height)],
        "resized_size": [int(resized_width), int(resized_height)],
        "image_grid_thw": [int(grid_t), int(grid_h), int(grid_w)],
        "vision_tokens": int(grid_t * grid_h * grid_w),
        "merge_size": int(merge_size),
    }
    return torch.from_numpy(flatten_patches), torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long), meta


def load_baseline_item(args: argparse.Namespace, model_dir: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    baseline = Path(args.baseline).expanduser().resolve()
    manifest_path = baseline / "reference_manifest.json" if baseline.is_dir() else baseline
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest["items"]
    item_index = int(args.item_index)
    if item_index < 0 or item_index >= len(items):
        raise ValueError(f"--item-index {item_index} is out of range for {len(items)} baseline items")
    item = items[item_index]
    dataset_raw = str(args.dataset_dir or manifest.get("build_summary", {}).get("page", {}).get("dataset_dir", "")).strip()
    dataset_dir = Path(dataset_raw).expanduser().resolve() if dataset_raw else None
    image_path = resolve_page_image_path(item, dataset_dir)
    bbox = item.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"baseline item {item.get('id')} lacks bbox_xyxy")
    with Image.open(image_path).convert("RGB") as image:
        crop = image.crop(tuple(int(value) for value in bbox)).copy()
    pixel_values, image_grid_thw, pre_meta = preprocess_crop(crop, load_preprocessor_config(model_dir))
    return pixel_values, image_grid_thw, {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "item_index": int(item_index),
        "id": str(item.get("id")),
        "layout_label": str(item.get("layout_label", "")),
        "source_image": str(image_path),
        "image_rel": str(item.get("image_rel", "")),
        "bbox_xyxy": [int(value) for value in bbox],
        "dataset_dir": str(dataset_dir) if dataset_dir is not None else None,
        "preprocess": pre_meta,
    }


def single_crop_grid_ints(image_grid_thw: torch.Tensor) -> tuple[int, int, int]:
    grid = image_grid_thw.detach().cpu().reshape(-1, 3)
    if int(grid.shape[0]) != 1:
        raise ValueError(f"expected one image grid, got {tuple(grid.shape)}")
    t, h, w = grid[0].tolist()
    return int(t), int(h), int(w)


def build_static_abs_pos_embed(model: LocalPaddleOCRVLForConditionalGeneration, image_grid_thw: torch.Tensor, device: torch.device) -> torch.Tensor:
    t, h, w = single_crop_grid_ints(image_grid_thw)
    embeddings = model.visual.vision_model.embeddings
    dummy = torch.empty((t * h * w, embeddings.embed_dim), device=device, dtype=embeddings.patch_embedding.weight.dtype)
    return embeddings.interpolate_pos_encoding(dummy, h, w).squeeze(0).repeat(t, 1).contiguous()


def build_static_vision_rope(model: LocalPaddleOCRVLForConditionalGeneration, image_grid_thw: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    _t, h, w = single_crop_grid_ints(image_grid_thw)
    encoder = model.visual.vision_model.encoder
    image_pids = torch.arange(int(image_grid_thw.prod().item()), device=device, dtype=torch.int64) % int(h * w)
    pids = torch.stack((image_pids // int(w), image_pids % int(w)), dim=-1)
    rotary_max = encoder.rotary_pos_emb(max(int(h), int(w)))
    rotary_embeddings = rotary_max[pids].flatten(1).repeat(1, 2)
    return rotary_embeddings.cos().contiguous(), rotary_embeddings.sin().contiguous()


def static_pad_tokens(real_seq_len: int, *, no_padding: bool = False) -> int:
    if no_padding:
        return 0
    minimum_physical_seq_len = int(real_seq_len) + 1
    alignment = 128 if minimum_physical_seq_len > 128 else 16
    remainder = minimum_physical_seq_len % alignment
    physical_seq_len = minimum_physical_seq_len if remainder == 0 else minimum_physical_seq_len + (alignment - remainder)
    return int(physical_seq_len - int(real_seq_len))


def build_static_pad_attention_mask(real_seq_len: int, pad_tokens: int, device: torch.device) -> torch.Tensor | None:
    if int(pad_tokens) <= 0:
        return None
    physical = int(real_seq_len) + int(pad_tokens)
    real = int(real_seq_len)
    mask = torch.zeros((1, 1, physical, physical), device=device, dtype=torch.bool)
    mask[..., :real, real:physical] = True
    mask[..., real:physical, :real] = True
    return mask.contiguous()


def import_torchair():
    try:
        import torchair
        from torchair.configs.compiler_config import CompilerConfig

        return torchair, CompilerConfig
    except Exception:
        from torch_npu.dynamo import torchair
        from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

        return torchair, CompilerConfig


def torchair_backend(*, run_eagerly: bool, mode: str):
    torchair, CompilerConfig = import_torchair()
    config = CompilerConfig()
    if mode != "default":
        config.mode = str(mode)
    if run_eagerly:
        config.debug.run_eagerly = True
    return torchair.get_npu_backend(compiler_config=config)


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    finite = torch.isfinite(detached.float())
    finite_values = detached.float()[finite]
    out: dict[str, Any] = {
        "shape": [int(dim) for dim in detached.shape],
        "stride": [int(dim) for dim in detached.stride()],
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "is_contiguous": bool(detached.is_contiguous()),
        "nonfinite_count": int((~finite).sum().item()),
    }
    if finite_values.numel() > 0:
        out.update(
            {
                "finite_min": float(finite_values.min().item()),
                "finite_max": float(finite_values.max().item()),
                "finite_mean_abs": float(finite_values.abs().mean().item()),
            }
        )
    return out


def diff_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, Any]:
    if tuple(lhs.shape) != tuple(rhs.shape):
        return {
            "shape_match": False,
            "lhs_shape": [int(dim) for dim in lhs.shape],
            "rhs_shape": [int(dim) for dim in rhs.shape],
        }
    lhs_f = lhs.detach().float()
    rhs_f = rhs.detach().float()
    finite = torch.isfinite(lhs_f) & torch.isfinite(rhs_f)
    diff = torch.abs(lhs_f - rhs_f)
    finite_diff = diff[finite]
    out: dict[str, Any] = {
        "shape_match": True,
        "shape": [int(dim) for dim in lhs.shape],
        "numel": int(lhs.numel()),
        "finite_pair_count": int(finite_diff.numel()),
        "lhs_nonfinite_count": int((~torch.isfinite(lhs_f)).sum().item()),
        "rhs_nonfinite_count": int((~torch.isfinite(rhs_f)).sum().item()),
        "diff_nonfinite_count": int((~torch.isfinite(diff)).sum().item()),
        "max_abs_diff": None if finite_diff.numel() == 0 else float(finite_diff.max().item()),
        "mean_abs_diff": None if finite_diff.numel() == 0 else float(finite_diff.mean().item()),
        "allclose_atol_5e_2_rtol_5e_2": bool(torch.allclose(lhs_f, rhs_f, atol=5e-2, rtol=5e-2)),
        "allclose_atol_1e_0_rtol_1e_0": bool(torch.allclose(lhs_f, rhs_f, atol=1.0, rtol=1.0)),
    }
    if finite_diff.numel() > 0:
        quantile_points = torch.tensor([0.25, 0.50, 0.75, 0.90, 0.95, 0.99], dtype=torch.float32)
        quantiles = torch.quantile(finite_diff.float(), quantile_points)
        out["abs_diff_quantiles"] = {
            "p25": float(quantiles[0].item()),
            "p50": float(quantiles[1].item()),
            "p75": float(quantiles[2].item()),
            "p90": float(quantiles[3].item()),
            "p95": float(quantiles[4].item()),
            "p99": float(quantiles[5].item()),
        }
    return out


class InlineSingleVisionLayer(torch.nn.Module):
    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        image_grid_thw: torch.Tensor,
        *,
        device: torch.device,
        layer_idx: int,
        attention_impl: str,
        promptfa_sparse_mode: int,
        ln_linear_mode: str,
        pre_promptfa_bridge: str,
        no_padding: bool,
    ):
        super().__init__()
        if attention_impl not in ("prompt_flash_attention", "manual"):
            raise ValueError(f"unsupported attention_impl={attention_impl!r}")
        if ln_linear_mode not in ("normal", "grouped_qkv", "grouped_qkv_mlp_fc1"):
            raise ValueError(f"unsupported ln_linear_mode={ln_linear_mode!r}")
        if pre_promptfa_bridge not in ("none", "contiguous", "clone", "transpose_roundtrip"):
            raise ValueError(f"unsupported pre_promptfa_bridge={pre_promptfa_bridge!r}")
        self.model = model
        self.layer_idx = int(layer_idx)
        self.attention_impl = str(attention_impl)
        self.promptfa_sparse_mode = int(promptfa_sparse_mode)
        self.ln_linear_mode = str(ln_linear_mode)
        self.pre_promptfa_bridge = str(pre_promptfa_bridge)
        self.real_seq_len = int(image_grid_thw.prod().item())
        self.pad_tokens = static_pad_tokens(self.real_seq_len, no_padding=bool(no_padding))
        self.physical_seq_len = int(self.real_seq_len + self.pad_tokens)

        abs_pos = build_static_abs_pos_embed(model, image_grid_thw, device)
        rope_cos, rope_sin = build_static_vision_rope(model, image_grid_thw, device)
        if self.pad_tokens:
            abs_pos = torch.cat(
                [abs_pos, torch.zeros(self.pad_tokens, abs_pos.shape[-1], device=device, dtype=abs_pos.dtype)],
                dim=0,
            ).contiguous()
            rope_cos = torch.cat(
                [rope_cos, torch.ones(self.pad_tokens, rope_cos.shape[-1], device=device, dtype=rope_cos.dtype)],
                dim=0,
            ).contiguous()
            rope_sin = torch.cat(
                [rope_sin, torch.zeros(self.pad_tokens, rope_sin.shape[-1], device=device, dtype=rope_sin.dtype)],
                dim=0,
            ).contiguous()
        self.register_buffer("abs_pos_embed_const", abs_pos, persistent=False)
        self.register_buffer("rope_cos_const", rope_cos, persistent=False)
        self.register_buffer("rope_sin_const", rope_sin, persistent=False)
        self.register_buffer(
            "pad_attention_mask",
            build_static_pad_attention_mask(self.real_seq_len, self.pad_tokens, device),
            persistent=False,
        )

    @staticmethod
    def grouped_linear(hidden_states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        import torch_npu

        group_list = torch.full((1,), hidden_states.shape[0], dtype=torch.int64, device=hidden_states.device)
        weight_3d = weight.transpose(0, 1).contiguous().unsqueeze(0)
        bias_arg = None if bias is None else [bias.contiguous().unsqueeze(0)]
        return torch_npu.npu_grouped_matmul(
            [hidden_states],
            [weight_3d],
            bias=bias_arg,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]

    def qkv_projection(self, attention: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.ln_linear_mode in ("grouped_qkv", "grouped_qkv_mlp_fc1"):
            qkv_weight = torch.cat(
                [attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight],
                dim=0,
            ).contiguous()
            if attention.q_proj.bias is None or attention.k_proj.bias is None or attention.v_proj.bias is None:
                qkv_bias = None
            else:
                qkv_bias = torch.cat(
                    [attention.q_proj.bias, attention.k_proj.bias, attention.v_proj.bias],
                    dim=0,
                ).contiguous()
            return self.grouped_linear(hidden_states, qkv_weight, qkv_bias)
        return torch.cat(
            [
                attention.q_proj(hidden_states),
                attention.k_proj(hidden_states),
                attention.v_proj(hidden_states),
            ],
            dim=-1,
        )

    def bridge_promptfa_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.pre_promptfa_bridge == "none":
            return tensor
        if self.pre_promptfa_bridge == "contiguous":
            return tensor.contiguous()
        if self.pre_promptfa_bridge == "clone":
            return tensor.clone()
        if self.pre_promptfa_bridge == "transpose_roundtrip":
            return tensor.transpose(-1, -2).contiguous().transpose(-1, -2).contiguous()
        raise RuntimeError(f"unreachable pre_promptfa_bridge={self.pre_promptfa_bridge!r}")

    def promptfa_or_manual(self, q_bnsd: torch.Tensor, k_bnsd: torch.Tensor, v_bnsd: torch.Tensor, attention: torch.nn.Module) -> torch.Tensor:
        if self.attention_impl == "manual":
            scores = torch.matmul(q_bnsd, k_bnsd.transpose(2, 3)) * attention.scaling
            if self.pad_attention_mask is not None:
                scores = scores.masked_fill(self.pad_attention_mask, torch.finfo(scores.dtype).min)
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_bnsd.dtype)
            return torch.matmul(probs, v_bnsd)

        import torch_npu

        mask_kwargs = {}
        sparse_mode = 0
        if self.pad_attention_mask is not None:
            mask_kwargs["atten_mask"] = self.pad_attention_mask.to(torch.bool).contiguous()
            sparse_mode = int(self.promptfa_sparse_mode)
        return torch_npu.npu_prompt_flash_attention(
            q_bnsd.contiguous(),
            k_bnsd.contiguous(),
            v_bnsd.contiguous(),
            num_heads=int(attention.num_heads),
            input_layout="BNSD",
            scale_value=float(attention.scaling),
            sparse_mode=sparse_mode,
            **mask_kwargs,
        )

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        transformer = self.model.visual.vision_model
        embeddings = transformer.embeddings
        layer = transformer.encoder.layers[self.layer_idx]
        attention = layer.self_attn
        mlp = layer.mlp

        pixel_values = pixel_values.to(dtype=embeddings.patch_embedding.weight.dtype)
        patch_flat = embeddings.patch_embedding(pixel_values).flatten(-2).squeeze(-1)
        if self.pad_tokens:
            patch_pad = torch.cat(
                [
                    patch_flat,
                    torch.zeros(
                        self.pad_tokens,
                        patch_flat.shape[-1],
                        device=patch_flat.device,
                        dtype=patch_flat.dtype,
                    ),
                ],
                dim=0,
            )
        else:
            patch_pad = patch_flat
        patch_pos = patch_pad + self.abs_pos_embed_const
        ln1 = layer.layer_norm1(patch_pos)

        seq_len = ln1.shape[0]
        qkv = self.qkv_projection(attention, ln1)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.view(seq_len, attention.num_heads, attention.head_dim)
        key = key.view(seq_len, attention.num_heads, attention.head_dim)
        value = value.view(seq_len, attention.num_heads, attention.head_dim)
        query, key = apply_rotary_pos_emb_vision(query, key, self.rope_cos_const, self.rope_sin_const)
        qk_rope_v = torch.cat(
            [query.reshape(seq_len, -1), key.reshape(seq_len, -1), value.reshape(seq_len, -1)],
            dim=-1,
        )
        q_bnsd = self.bridge_promptfa_input(query.transpose(0, 1).unsqueeze(0))
        k_bnsd = self.bridge_promptfa_input(key.transpose(0, 1).unsqueeze(0))
        v_bnsd = self.bridge_promptfa_input(value.transpose(0, 1).unsqueeze(0))
        attn_kernel_bnsd = self.promptfa_or_manual(q_bnsd, k_bnsd, v_bnsd, attention)
        attn_kernel_out = attn_kernel_bnsd.transpose(1, 2).contiguous().view(seq_len, -1)
        attn_out_proj = attention.out_proj(attn_kernel_out)
        attn_residual = patch_pos + attn_out_proj
        ln2 = layer.layer_norm2(attn_residual)
        if self.ln_linear_mode == "grouped_qkv_mlp_fc1":
            mlp_fc1 = self.grouped_linear(ln2, mlp.fc1.weight, mlp.fc1.bias)
        else:
            mlp_fc1 = mlp.fc1(ln2)
        mlp_act = _activation(mlp.hidden_act, mlp_fc1)
        mlp_fc2 = mlp.fc2(mlp_act)
        layer0_out = attn_residual + mlp_fc2
        return (
            patch_flat,
            patch_pad,
            patch_pos,
            ln1,
            qkv,
            qk_rope_v,
            q_bnsd,
            k_bnsd,
            v_bnsd,
            attn_kernel_bnsd,
            attn_kernel_out,
            attn_out_proj,
            attn_residual,
            ln2,
            mlp_fc1,
            mlp_act,
            mlp_fc2,
            layer0_out,
        )


def compare_outputs(lhs: tuple[torch.Tensor, ...], rhs: tuple[torch.Tensor, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, lhs_tensor, rhs_tensor in zip(STAGE_NAMES, lhs, rhs):
        diff = diff_stats(lhs_tensor.detach().cpu(), rhs_tensor.detach().cpu())
        rows.append(
            {
                "stage": name,
                "matches_5e_2": bool(diff.get("allclose_atol_5e_2_rtol_5e_2")),
                "matches_1e_0": bool(diff.get("allclose_atol_1e_0_rtol_1e_0")),
                "diff": diff,
                "lhs_summary": tensor_summary(lhs_tensor.detach().cpu()),
                "rhs_summary": tensor_summary(rhs_tensor.detach().cpu()),
            }
        )
    return rows


def first_bad_stage(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if not row["matches_5e_2"]:
            return row
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--item-index", type=int, default=0)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "float16", "fp32", "float32", "bf16", "bfloat16"))
    parser.add_argument("--npu-jit-compile", default="off", choices=("default", "off", "on"))
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--attention", default="prompt_flash_attention", choices=("prompt_flash_attention", "manual"))
    parser.add_argument("--promptfa-sparse-mode", type=int, default=1, choices=(0, 1))
    parser.add_argument("--ln-linear-mode", default="grouped_qkv_mlp_fc1", choices=("normal", "grouped_qkv", "grouped_qkv_mlp_fc1"))
    parser.add_argument("--pre-promptfa-bridge", default="none", choices=("none", "contiguous", "clone", "transpose_roundtrip"))
    parser.add_argument("--no-padding", action="store_true")
    parser.add_argument("--compile-backend", default="torchair", choices=("torchair", "default", "aot_eager", "inductor"))
    parser.add_argument("--torchair-mode", default="default", choices=("default", "max-autotune"))
    parser.add_argument("--torchair-run-eagerly", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    configure_npu_jit_compile(args.npu_jit_compile, device)
    model_dir = _resolve_model_dir(args.model)

    maybe_sync(device)
    start = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device=device).eval()
    maybe_sync(device)
    model_load_s = float(time.perf_counter() - start)

    pixel_values_cpu, image_grid_thw_cpu, item_meta = load_baseline_item(args, model_dir)
    pixel_values = pixel_values_cpu.to(device=device, dtype=model.visual.dtype)
    image_grid_thw = image_grid_thw_cpu.to(device=device)
    module = InlineSingleVisionLayer(
        model,
        image_grid_thw,
        device=device,
        layer_idx=int(args.layer_index),
        attention_impl=str(args.attention),
        promptfa_sparse_mode=int(args.promptfa_sparse_mode),
        ln_linear_mode=str(args.ln_linear_mode),
        pre_promptfa_bridge=str(args.pre_promptfa_bridge),
        no_padding=bool(args.no_padding),
    ).eval()

    maybe_sync(device)
    start = time.perf_counter()
    eager_before = module(pixel_values)
    maybe_sync(device)
    eager_before_s = float(time.perf_counter() - start)

    import torch._dynamo

    old_capture_scalar_outputs = bool(torch._dynamo.config.capture_scalar_outputs)
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.reset()
    compile_kwargs: dict[str, Any] = {"fullgraph": True, "dynamic": False}
    backend_meta: dict[str, Any]
    if args.compile_backend == "torchair":
        compile_kwargs["backend"] = torchair_backend(
            run_eagerly=bool(args.torchair_run_eagerly),
            mode=str(args.torchair_mode),
        )
        backend_meta = {
            "backend_kind": "torchair",
            "torchair_mode": str(args.torchair_mode),
            "torchair_run_eagerly": bool(args.torchair_run_eagerly),
        }
    elif args.compile_backend == "default":
        backend_meta = {"backend_kind": "torch_default"}
    else:
        compile_kwargs["backend"] = str(args.compile_backend)
        backend_meta = {"backend_kind": str(args.compile_backend)}

    try:
        maybe_sync(device)
        start = time.perf_counter()
        compiled = torch.compile(module, **compile_kwargs)
        maybe_sync(device)
        compile_wrapper_s = float(time.perf_counter() - start)

        maybe_sync(device)
        start = time.perf_counter()
        compiled_first = compiled(pixel_values)
        maybe_sync(device)
        compiled_first_s = float(time.perf_counter() - start)

        maybe_sync(device)
        start = time.perf_counter()
        compiled_second = compiled(pixel_values)
        maybe_sync(device)
        compiled_second_s = float(time.perf_counter() - start)

        maybe_sync(device)
        start = time.perf_counter()
        eager_after = module(pixel_values)
        maybe_sync(device)
        eager_after_s = float(time.perf_counter() - start)
    finally:
        torch._dynamo.config.capture_scalar_outputs = old_capture_scalar_outputs

    rows = compare_outputs(compiled_second, eager_before)
    eager_stability_rows = compare_outputs(eager_after, eager_before)
    compiled_stability_rows = compare_outputs(compiled_first, compiled_second)
    first_bad = first_bad_stage(rows)
    output = {
        "schema_version": 1,
        "experiment": "07_vision_prefill_optimization",
        "kind": "inline_single_layer_compile_repro",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model_dir),
        "device": str(device),
        "dtype": str(dtype),
        "model_load_s": model_load_s,
        "item": item_meta,
        "config": {
            "layer_index": int(args.layer_index),
            "attention": str(args.attention),
            "promptfa_sparse_mode": int(args.promptfa_sparse_mode),
            "ln_linear_mode": str(args.ln_linear_mode),
            "pre_promptfa_bridge": str(args.pre_promptfa_bridge),
            "no_padding": bool(args.no_padding),
            "real_seq_len": int(module.real_seq_len),
            "pad_tokens": int(module.pad_tokens),
            "physical_seq_len": int(module.physical_seq_len),
            "compile_backend": str(args.compile_backend),
            "fullgraph": True,
            "dynamic": False,
            "backend_meta": backend_meta,
        },
        "timing_s": {
            "eager_before": eager_before_s,
            "compile_wrapper": compile_wrapper_s,
            "compiled_first": compiled_first_s,
            "compiled_second": compiled_second_s,
            "eager_after": eager_after_s,
        },
        "summary": {
            "compiled_second_matches_eager_count": int(sum(bool(row["matches_5e_2"]) for row in rows)),
            "compiled_second_matches_eager_all": bool(all(bool(row["matches_5e_2"]) for row in rows)),
            "first_bad_stage": None
            if first_bad is None
            else {
                "stage": first_bad["stage"],
                "max_abs_diff": first_bad["diff"].get("max_abs_diff"),
                "mean_abs_diff": first_bad["diff"].get("mean_abs_diff"),
                "compiled_nonfinite_count": first_bad["diff"].get("lhs_nonfinite_count"),
                "eager_nonfinite_count": first_bad["diff"].get("rhs_nonfinite_count"),
            },
        },
        "compiled_second_vs_eager_before": rows,
        "eager_after_vs_eager_before": eager_stability_rows,
        "compiled_first_vs_second": compiled_stability_rows,
    }

    output_path_raw = str(args.output or "").strip()
    if output_path_raw:
        output_path = Path(output_path_raw).expanduser().resolve()
    else:
        out_dir = SCRIPT_DIR / "outputs" / f"inline_single_layer_repro_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        output_path = out_dir / "inline_single_layer_repro.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    print("EXP07_INLINE_SINGLE_LAYER_REPRO SUMMARY")
    print(f"output={output_path}")
    print(
        "config="
        f"attention={args.attention} ln_linear_mode={args.ln_linear_mode} "
        f"pre_promptfa_bridge={args.pre_promptfa_bridge} "
        f"physical_seq_len={module.physical_seq_len} real_seq_len={module.real_seq_len}"
    )
    print(f"first_bad_stage={output['summary']['first_bad_stage']}")
    print("EXP07_INLINE_SINGLE_LAYER_REPRO STAGE_TABLE")
    for row in rows:
        diff = row["diff"]
        print(
            f"stage={row['stage']} "
            f"match={row['matches_5e_2']} "
            f"shape={diff.get('shape')} "
            f"max_abs={diff.get('max_abs_diff')} "
            f"mean_abs={diff.get('mean_abs_diff')} "
            f"p50={diff.get('abs_diff_quantiles', {}).get('p50')} "
            f"p90={diff.get('abs_diff_quantiles', {}).get('p90')} "
            f"p99={diff.get('abs_diff_quantiles', {}).get('p99')} "
            f"compiled_nonfinite={diff.get('lhs_nonfinite_count')} "
            f"eager_nonfinite={diff.get('rhs_nonfinite_count')}"
        )


if __name__ == "__main__":
    main()
