#!/usr/bin/env python3
"""Validate batching only the PaddleOCR-VL vision transformer layers.

The intended split is:

  per crop, sequential: patch embedding + absolute position + padded prefix tensors
  batched/compiled:    encoder layers + post LayerNorm over [B, S_fixed, hidden]
  per crop, sequential: slice real rows + projector/text prefill/static decode checks

This script deliberately does not batch raw crops or patch embedding.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from local_modeling_paddleocr_vl import (
    LocalPaddleOCRVLForConditionalGeneration,
    _activation,
    apply_rotary_pos_emb_vision,
    attention_softmax,
    get_vision_attention_impl,
    get_vision_prompt_fa_layout,
    get_vision_softmax_dtype_mode,
    vision_prompt_flash_attention_bnsd,
)
from vision_prefill_bench import (
    DEFAULT_MODEL,
    DEFAULT_VISION_TORCHAIR_CACHE_DIR,
    STATIC_VISUAL_LN_IMPL_CHOICES,
    STATIC_VISUAL_LN_LINEAR_MODE_CHOICES,
    VISION_COMPILE_BACKEND_CHOICES,
    SingleCropStaticVisualModule,
    add_common_args,
    add_torchair_diagnostic_args,
    apply_runtime_env,
    build_inputs_from_manifest,
    clean_json,
    compute_prefill_state_from_visual_features,
    diff_stats,
    first_mismatch,
    generate_from_prefill_state,
    input_row,
    import_torchair,
    json_default,
    load_baseline_manifest,
    load_model_for_args,
    load_preprocessor_config,
    maybe_sync,
    resolve_dataset_dir,
    sha256_file,
    stats,
    topk_summary,
    torchair_cache_dir_for_static_visual,
    torchair_compiler_config,
    trim_after_eos,
    vision_compile_backend,
    vision_tokens,
)
from w8a8_vision import (
    VISION_LINEAR_QUANTIZATION_CHOICES,
    VISION_LINEAR_SITES,
    W8A8_WEIGHT_LAYOUT_CHOICES,
    packed_from_linears,
    resolve_weight_layout,
)


class BatchedStaticVisualEncoderModule(torch.nn.Module):
    """Compiled boundary for encoder layers + post LayerNorm only."""

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        *,
        fixed_physical_seq_len: int,
        ln_impl: str,
        ln_linear_mode: str,
        promptfa_pad_head_dim_to: int,
        linear_quantization: str = "none",
        w8a8_sites: tuple[str, ...] = VISION_LINEAR_SITES,
        w8a8_weight_layout: str = "auto",
        w8a8_static_scale_headroom: float = 1.05,
    ):
        super().__init__()
        if ln_impl not in STATIC_VISUAL_LN_IMPL_CHOICES:
            raise ValueError(f"unsupported LayerNorm impl={ln_impl!r}")
        if ln_linear_mode not in STATIC_VISUAL_LN_LINEAR_MODE_CHOICES:
            raise ValueError(f"unsupported LN-linear mode={ln_linear_mode!r}")
        if linear_quantization not in VISION_LINEAR_QUANTIZATION_CHOICES:
            raise ValueError(f"unsupported vision Linear quantization={linear_quantization!r}")
        unknown_w8a8_sites = set(w8a8_sites) - set(VISION_LINEAR_SITES)
        if unknown_w8a8_sites:
            raise ValueError(f"unsupported W8A8 sites={sorted(unknown_w8a8_sites)}")
        if linear_quantization != "none" and not w8a8_sites:
            raise ValueError("at least one W8A8 site is required when quantization is enabled")
        if w8a8_weight_layout not in W8A8_WEIGHT_LAYOUT_CHOICES:
            raise ValueError(f"unsupported W8A8 weight layout={w8a8_weight_layout!r}")
        if float(w8a8_static_scale_headroom) < 1.0:
            raise ValueError("W8A8 static scale headroom must be >= 1.0")
        self.model = model
        self.fixed_physical_seq_len = int(fixed_physical_seq_len)
        self.ln_impl = str(ln_impl)
        self.ln_linear_mode = str(ln_linear_mode)
        self.promptfa_pad_head_dim_to = int(promptfa_pad_head_dim_to)
        self.linear_quantization = str(linear_quantization)
        self.w8a8_sites = tuple(site for site in VISION_LINEAR_SITES if site in set(w8a8_sites))
        self.w8a8_weight_layout_requested = str(w8a8_weight_layout)
        self.w8a8_weight_layout = resolve_weight_layout(
            next(model.parameters()).device,
            self.w8a8_weight_layout_requested,
        )
        self.w8a8_static_scale_headroom = float(w8a8_static_scale_headroom)
        self.w8a8_layers = torch.nn.ModuleList()
        self._w8a8_prepared = False
        self._calibration_enabled = False
        self._calibration_maxima: dict[str, torch.Tensor] = {}

    def _static_layer_norm(self, layer_norm: torch.nn.LayerNorm, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.ln_impl == "module":
            return layer_norm(hidden_states)
        weight = layer_norm.weight
        bias = layer_norm.bias
        eps = float(layer_norm.eps)
        if self.ln_impl == "functional":
            return F.layer_norm(hidden_states, (hidden_states.shape[-1],), weight, bias, eps)
        if self.ln_impl == "manual_fp32":
            x = hidden_states.float()
            mean = x.mean(dim=-1, keepdim=True)
            centered = x - mean
            var = centered.pow(2).mean(dim=-1, keepdim=True)
            y = centered * torch.rsqrt(var + eps)
            y = y.to(dtype=hidden_states.dtype)
        elif self.ln_impl == "manual_fp16":
            mean = hidden_states.mean(dim=-1, keepdim=True)
            centered = hidden_states - mean
            var = centered.pow(2).mean(dim=-1, keepdim=True)
            y = centered * torch.rsqrt(var + eps)
        else:
            raise RuntimeError(f"unreachable ln_impl={self.ln_impl!r}")
        if weight is not None:
            y = y * weight
        if bias is not None:
            y = y + bias
        return y

    @staticmethod
    def _site_key(layer_index: int, site: str) -> str:
        return f"layer_{int(layer_index):02d}.{site}"

    def set_calibration_enabled(self, enabled: bool) -> None:
        if bool(enabled) and self.linear_quantization not in {"w8a8_static", "w8a8_static_pad64"}:
            raise ValueError("activation calibration is only used for static W8A8")
        if bool(enabled) and self._w8a8_prepared:
            raise RuntimeError("cannot calibrate after W8A8 weights have been prepared")
        self._calibration_enabled = bool(enabled)

    def _observe_activation(self, layer_index: int, site: str, hidden_states: torch.Tensor) -> None:
        if not self._calibration_enabled or site not in self.w8a8_sites:
            return
        key = self._site_key(layer_index, site)
        maximum = hidden_states.detach().float().abs().max()
        previous = self._calibration_maxima.get(key)
        self._calibration_maxima[key] = maximum if previous is None else torch.maximum(previous, maximum)

    def _static_input_scale(self, layer_index: int, site: str) -> float | None:
        if self.linear_quantization not in {"w8a8_static", "w8a8_static_pad64"}:
            return None
        key = self._site_key(layer_index, site)
        maximum = self._calibration_maxima.get(key)
        if maximum is None:
            raise RuntimeError(f"missing static W8A8 calibration maximum for {key}")
        value = float(maximum.item())
        return max(value * self.w8a8_static_scale_headroom / 127.0, torch.finfo(torch.float32).eps)

    def prepare_w8a8(self) -> dict[str, Any]:
        if self.linear_quantization == "none":
            return {"enabled": False, "mode": "none"}
        if self._w8a8_prepared:
            raise RuntimeError("W8A8 weights are already prepared")
        self._calibration_enabled = False
        layers = []
        for layer_index, encoder_layer in enumerate(self.model.visual.vision_model.encoder.layers):
            attention = encoder_layer.self_attn
            mlp = encoder_layer.mlp
            packed_modules: dict[str, torch.nn.Module] = {}
            if "qkv" in self.w8a8_sites:
                packed_modules["qkv"] = packed_from_linears(
                    [attention.q_proj, attention.k_proj, attention.v_proj],
                    mode=self.linear_quantization,
                    weight_layout=self.w8a8_weight_layout,
                    static_input_scale=self._static_input_scale(layer_index, "qkv"),
                )
            if "out_proj" in self.w8a8_sites:
                packed_modules["out_proj"] = packed_from_linears(
                    [attention.out_proj],
                    mode=self.linear_quantization,
                    weight_layout=self.w8a8_weight_layout,
                    static_input_scale=self._static_input_scale(layer_index, "out_proj"),
                )
            if "fc1" in self.w8a8_sites:
                packed_modules["fc1"] = packed_from_linears(
                    [mlp.fc1],
                    mode=self.linear_quantization,
                    weight_layout=self.w8a8_weight_layout,
                    static_input_scale=self._static_input_scale(layer_index, "fc1"),
                )
            if "fc2" in self.w8a8_sites:
                packed_modules["fc2"] = packed_from_linears(
                    [mlp.fc2],
                    mode=self.linear_quantization,
                    weight_layout=self.w8a8_weight_layout,
                    static_input_scale=self._static_input_scale(layer_index, "fc2"),
                )
            packed = torch.nn.ModuleDict(packed_modules)
            layers.append(packed)
        self.w8a8_layers.extend(layers)
        self._w8a8_prepared = True
        first = self.w8a8_layers[0]
        return {
            "enabled": True,
            "mode": self.linear_quantization,
            "sites": list(self.w8a8_sites),
            "weight_layout_requested": self.w8a8_weight_layout_requested,
            "weight_layout_resolved": self.w8a8_weight_layout,
            "static_scale_headroom": self.w8a8_static_scale_headroom,
            "calibrated_site_count": int(len(self._calibration_maxima)),
            "packed_linear_count": int(len(self.w8a8_layers) * len(self.w8a8_sites)),
            "first_layer": {name: module.metadata() for name, module in first.items()},
        }

    def calibration_summary(self) -> dict[str, Any]:
        maxima = {key: float(value.item()) for key, value in sorted(self._calibration_maxima.items())}
        scales = {
            key: max(value * self.w8a8_static_scale_headroom / 127.0, torch.finfo(torch.float32).eps)
            for key, value in maxima.items()
        }
        return {
            "enabled": bool(
                self.linear_quantization in {"w8a8_static", "w8a8_static_pad64"}
            ),
            "site_count": int(len(maxima)),
            "headroom": float(self.w8a8_static_scale_headroom),
            "maxima": maxima,
            "input_scales": scales,
        }

    def _quantized_linear(self, layer_index: int, site: str, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._w8a8_prepared:
            raise RuntimeError("W8A8 Linear requested before prepare_w8a8()")
        return self.w8a8_layers[layer_index][site](hidden_states)

    def _site_is_quantized(self, site: str) -> bool:
        return bool(self._w8a8_prepared and site in self.w8a8_sites)

    @staticmethod
    def _linear_maybe_grouped(hidden_states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        leading_shape = tuple(hidden_states.shape[:-1])
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        if flat.device.type != "npu":
            out = F.linear(flat, weight, bias)
        else:
            import torch_npu

            group_list = torch.full((1,), flat.shape[0], dtype=torch.int64, device=flat.device)
            weight_3d = weight.transpose(0, 1).contiguous().unsqueeze(0)
            bias_arg = None if bias is None else [bias.contiguous().unsqueeze(0)]
            out = torch_npu.npu_grouped_matmul(
                [flat],
                [weight_3d],
                bias=bias_arg,
                group_list=group_list,
                split_item=2,
                group_type=0,
                group_list_type=1,
            )[0]
        return out.reshape(*leading_shape, out.shape[-1])

    def _qkv_projection(
        self,
        attention: torch.nn.Module,
        hidden_states: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        self._observe_activation(layer_index, "qkv", hidden_states)
        if self._site_is_quantized("qkv"):
            return self._quantized_linear(layer_index, "qkv", hidden_states)
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
            return self._linear_maybe_grouped(hidden_states, qkv_weight, qkv_bias)
        return torch.cat(
            [
                attention.q_proj(hidden_states),
                attention.k_proj(hidden_states),
                attention.v_proj(hidden_states),
            ],
            dim=-1,
        )

    def _mlp(self, mlp: torch.nn.Module, hidden_states: torch.Tensor, layer_index: int) -> torch.Tensor:
        self._observe_activation(layer_index, "fc1", hidden_states)
        if self._site_is_quantized("fc1"):
            fc1_out = self._quantized_linear(layer_index, "fc1", hidden_states)
        elif self.ln_linear_mode == "grouped_qkv_mlp_fc1":
            fc1_out = self._linear_maybe_grouped(hidden_states, mlp.fc1.weight, mlp.fc1.bias)
        else:
            fc1_out = mlp.fc1(hidden_states)
        activated = _activation(mlp.hidden_act, fc1_out)
        self._observe_activation(layer_index, "fc2", activated)
        if self._site_is_quantized("fc2"):
            return self._quantized_linear(layer_index, "fc2", activated)
        return mlp.fc2(activated)

    def _promptfa_call_head_dim(self, attention: torch.nn.Module) -> int:
        real_head_dim = int(attention.head_dim)
        target = int(self.promptfa_pad_head_dim_to)
        if target <= 0:
            return real_head_dim
        if target < real_head_dim:
            raise ValueError(f"PromptFA pad target must be >= {real_head_dim}, got {target}")
        return target

    def _pad_promptfa_head_dim(self, tensor: torch.Tensor, attention: torch.nn.Module) -> torch.Tensor:
        call_head_dim = int(self._promptfa_call_head_dim(attention))
        real_head_dim = int(tensor.shape[-1])
        if call_head_dim == real_head_dim:
            return tensor
        return F.pad(tensor, (0, call_head_dim - real_head_dim))

    def _attention(
        self,
        attention: torch.nn.Module,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        batch_size, seq_length, _hidden = hidden_states.shape
        qkv = self._qkv_projection(attention, hidden_states, layer_index)
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)
        query_states = query_states.view(batch_size, seq_length, attention.num_heads, attention.head_dim)
        key_states = key_states.view(batch_size, seq_length, attention.num_heads, attention.head_dim)
        value_states = value_states.view(batch_size, seq_length, attention.num_heads, attention.head_dim)
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, rope_cos, rope_sin)
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()
        attention_impl = get_vision_attention_impl()
        if attention_impl == "prompt_flash_attention":
            if get_vision_prompt_fa_layout() != "bnsd":
                raise ValueError("batched encoder currently supports PromptFA layout bnsd only")
            query_states = self._pad_promptfa_head_dim(query_states, attention).contiguous()
            key_states = self._pad_promptfa_head_dim(key_states, attention).contiguous()
            value_states = self._pad_promptfa_head_dim(value_states, attention).contiguous()
            attn_output = vision_prompt_flash_attention_bnsd(
                query_states,
                key_states,
                value_states,
                num_heads=int(attention.num_heads),
                scale=float(attention.scaling),
                atten_mask=attention_mask,
            )
            if int(attn_output.shape[-1]) != int(attention.head_dim):
                attn_output = attn_output[..., : int(attention.head_dim)].contiguous()
        elif attention_impl == "manual":
            # Express the head-wise batched products explicitly. TorchAir/GE can
            # lower 4-D torch.matmul through a plain MatMul and confuse the head
            # dimension with a broadcast dimension (for this model, k-axis 16
            # versus 1). Flattening B and H makes BatchMatMul semantics explicit
            # while preserving the exact eager attention calculation.
            num_heads = int(attention.num_heads)
            head_dim = int(attention.head_dim)
            query_bh = query_states.reshape(batch_size * num_heads, seq_length, head_dim)
            key_bh = key_states.reshape(batch_size * num_heads, seq_length, head_dim)
            value_bh = value_states.reshape(batch_size * num_heads, seq_length, head_dim)
            scores = torch.bmm(query_bh, key_bh.transpose(1, 2)).view(
                batch_size,
                num_heads,
                seq_length,
                seq_length,
            ) * attention.scaling
            scores = scores.masked_fill(attention_mask, torch.finfo(scores.dtype).min)
            probs = attention_softmax(
                scores,
                dim=-1,
                output_dtype=query_states.dtype,
                mode=get_vision_softmax_dtype_mode(),
            )
            attn_output = torch.bmm(
                probs.reshape(batch_size * num_heads, seq_length, seq_length),
                value_bh,
            ).view(batch_size, num_heads, seq_length, head_dim)
        else:
            raise ValueError(f"unknown vision attention implementation: {attention_impl!r}")
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        self._observe_activation(layer_index, "out_proj", attn_output)
        if self._site_is_quantized("out_proj"):
            return self._quantized_linear(layer_index, "out_proj", attn_output)
        return attention.out_proj(attn_output)

    def _encoder_layer(
        self,
        encoder_layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        attn_input = self._static_layer_norm(encoder_layer.layer_norm1, hidden_states)
        hidden_states = hidden_states + self._attention(
            encoder_layer.self_attn,
            attn_input,
            rope_cos,
            rope_sin,
            attention_mask,
            layer_index,
        )
        mlp_input = self._static_layer_norm(encoder_layer.layer_norm2, hidden_states)
        return hidden_states + self._mlp(encoder_layer.mlp, mlp_input, layer_index)

    def forward(
        self,
        prefix_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = prefix_hidden_states
        transformer = self.model.visual.vision_model
        for layer_index, encoder_layer in enumerate(transformer.encoder.layers):
            hidden_states = self._encoder_layer(
                encoder_layer,
                hidden_states,
                rope_cos,
                rope_sin,
                attention_mask,
                layer_index,
            )
        return self._static_layer_norm(transformer.post_layernorm, hidden_states)


def decode_safely(tokenizer: Tokenizer, token_ids: list[int]) -> tuple[str, list[int]]:
    vocab_size = int(tokenizer.get_vocab_size())
    invalid = [int(value) for value in token_ids if int(value) < 0 or int(value) >= vocab_size]
    if invalid:
        return "", invalid[:16]
    return tokenizer.decode(token_ids, skip_special_tokens=True), []


def encoder_cache_dir(
    cache_root: Path,
    *,
    batch_size: int,
    fixed_physical_seq_len: int,
    dtype: torch.dtype,
    ln_impl: str,
    ln_linear_mode: str,
    promptfa_call_head_dim: int,
    promptfa_layout: str,
    promptfa_mask_sparse_mode: int,
    torchair_mode: str,
    linear_quantization: str,
    w8a8_sites: tuple[str, ...],
    w8a8_weight_layout: str,
    w8a8_static_scale_headroom: float,
) -> Path:
    dummy_grid = torch.tensor([[int(batch_size), 1, int(fixed_physical_seq_len)]], dtype=torch.long)
    base = torchair_cache_dir_for_static_visual(
        cache_root,
        fixed_physical_seq_len=int(fixed_physical_seq_len),
        real_seq_len=int(fixed_physical_seq_len),
        image_grid_thw=dummy_grid,
        dtype=dtype,
        ln_impl=ln_impl,
        ln_linear_mode=ln_linear_mode,
        promptfa_call_head_dim=int(promptfa_call_head_dim),
        promptfa_layout=promptfa_layout,
        promptfa_mask_sparse_mode=int(promptfa_mask_sparse_mode),
        torchair_mode=torchair_mode,
    )
    attention = get_vision_attention_impl().replace("/", "_").replace(" ", "_")
    quant_suffix = (
        f"quant-{linear_quantization}_sites-{'-'.join(w8a8_sites)}_layout-{w8a8_weight_layout}_"
        f"headroom-{float(w8a8_static_scale_headroom):g}"
    )
    return base.parent / f"encoder_only_{attention}_B{int(batch_size)}_{quant_suffix}_{base.name}"


def compile_encoder_forward(
    module: BatchedStaticVisualEncoderModule,
    *,
    backend_name: str,
    device: torch.device,
    use_cache_compile: bool,
    cache_root: Path,
    batch_size: int,
    fixed_physical_seq_len: int,
    dtype: torch.dtype,
    torchair_mode: str,
    torchair_run_eagerly: bool,
    torchair_graph_dump_type: str,
    torchair_graph_dump_dir: str,
    torchair_msit_dump_kind: str,
    torchair_msit_dump_dir: str,
    torchair_msit_dump_mode: str,
    torchair_msit_dump_token: str,
    torchair_msit_dump_layer: str,
    torchair_msit_fusion_switch_file: str,
    promptfa_mask_sparse_mode: int,
) -> tuple[Callable[..., torch.Tensor], dict[str, Any]]:
    if backend_name == "none":
        return module, {
            "enabled": False,
            "compile_api": None,
            "backend": "none",
            "uses_torchair_cache_compile": False,
            "torchair_ge_cache": False,
        }
    if use_cache_compile:
        if backend_name != "torchair" or device.type != "npu":
            raise ValueError("encoder cache_compile requires --vision-compile-backend torchair on NPU")
        if torchair_run_eagerly or torchair_graph_dump_type != "none" or torchair_msit_dump_kind != "none":
            raise ValueError("cache_compile is not a diagnostic mode; disable it for run-eagerly/dumps/MSIT")
        torchair, _CompilerConfig = import_torchair()
        config, backend_meta = torchair_compiler_config(
            torchair_mode=torchair_mode,
            torchair_run_eagerly=False,
            torchair_graph_dump_type="none",
            torchair_graph_dump_dir=None,
            torchair_msit_dump_kind="none",
            torchair_msit_dump_dir=None,
            torchair_msit_dump_mode=torchair_msit_dump_mode,
            torchair_msit_dump_token="",
            torchair_msit_dump_layer="",
            torchair_msit_fusion_switch_file=None,
        )
        first_attention = module.model.visual.vision_model.encoder.layers[0].self_attn
        cache_dir = encoder_cache_dir(
            cache_root,
            batch_size=batch_size,
            fixed_physical_seq_len=fixed_physical_seq_len,
            dtype=dtype,
            ln_impl=module.ln_impl,
            ln_linear_mode=module.ln_linear_mode,
            promptfa_call_head_dim=module._promptfa_call_head_dim(first_attention),
            promptfa_layout=get_vision_prompt_fa_layout(),
            promptfa_mask_sparse_mode=promptfa_mask_sparse_mode,
            torchair_mode=torchair_mode,
            linear_quantization=module.linear_quantization,
            w8a8_sites=module.w8a8_sites,
            w8a8_weight_layout=module.w8a8_weight_layout,
            w8a8_static_scale_headroom=module.w8a8_static_scale_headroom,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        maybe_sync(device)
        start = time.perf_counter()
        compiled = torchair.inference.cache_compile(
            module.forward,
            config=config,
            dynamic=False,
            cache_dir=str(cache_dir),
            ge_cache=True,
        )
        maybe_sync(device)
        return compiled, {
            "enabled": True,
            "backend": "torchair",
            "compile_api": "torchair.inference.cache_compile",
            "compile_wrapper_s": float(time.perf_counter() - start),
            "compile_backend_meta": backend_meta,
            "uses_torchair_cache_compile": True,
            "torchair_ge_cache": True,
            "torchair_cache_dir": str(cache_dir),
            "fullgraph": True,
            "dynamic": False,
        }
    import torch._dynamo as torch_dynamo

    old_capture_scalar_outputs = bool(torch_dynamo.config.capture_scalar_outputs)
    torch_dynamo.config.capture_scalar_outputs = True
    torch_dynamo.reset()
    backend, backend_meta = vision_compile_backend(
        backend_name,
        device,
        torchair_mode=torchair_mode,
        torchair_run_eagerly=torchair_run_eagerly,
        torchair_graph_dump_type=torchair_graph_dump_type,
        torchair_graph_dump_dir=torchair_graph_dump_dir,
        torchair_msit_dump_kind=torchair_msit_dump_kind,
        torchair_msit_dump_dir=torchair_msit_dump_dir,
        torchair_msit_dump_mode=torchair_msit_dump_mode,
        torchair_msit_dump_token=torchair_msit_dump_token,
        torchair_msit_dump_layer=torchair_msit_dump_layer,
        torchair_msit_fusion_switch_file=torchair_msit_fusion_switch_file,
    )
    compile_kwargs: dict[str, Any] = {"fullgraph": True, "dynamic": False}
    if backend is not None:
        compile_kwargs["backend"] = backend
    maybe_sync(device)
    start = time.perf_counter()
    compiled = torch.compile(module, **compile_kwargs)
    maybe_sync(device)
    return compiled, {
        "enabled": True,
        "backend": str(backend_name),
        "compile_api": "torch.compile",
        "compile_wrapper_s": float(time.perf_counter() - start),
        "compile_backend_meta": backend_meta,
        "capture_scalar_outputs_previous": old_capture_scalar_outputs,
        "uses_torchair_cache_compile": False,
        "torchair_ge_cache": False,
        "fullgraph": True,
        "dynamic": False,
    }


def build_prefix_batch(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    batch_items: list[Any],
    device: torch.device,
    fixed_physical_seq_len: int,
    ln_impl: str,
    ln_linear_mode: str,
    promptfa_pad_head_dim_to: int,
    debug_no_padding: bool,
    debug_min_pad_tokens: int,
    debug_pad_to_multiple: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    prefix_rows = []
    rope_cos_rows = []
    rope_sin_rows = []
    mask_rows = []
    metas = []
    for item in batch_items:
        wrapper = SingleCropStaticVisualModule(
            model,
            item.image_grid_thw,
            device=device,
            fixed_physical_seq_len=fixed_physical_seq_len,
            debug_no_padding=debug_no_padding,
            debug_min_pad_tokens=debug_min_pad_tokens,
            debug_pad_to_multiple=debug_pad_to_multiple,
            ln_impl=ln_impl,
            ln_linear_mode=ln_linear_mode,
            promptfa_pad_head_dim_to=promptfa_pad_head_dim_to,
        ).eval()
        pixel_values = item.pixel_values.to(device=device, dtype=model.visual.dtype)
        prefix = wrapper.forward_prefix(pixel_values)
        if int(prefix.shape[0]) != int(fixed_physical_seq_len):
            raise RuntimeError(
                f"prefix physical seq {int(prefix.shape[0])} != fixed S {int(fixed_physical_seq_len)}"
            )
        mask = wrapper.static_pad_attention_mask
        if mask is None:
            mask = torch.zeros(
                (1, 1, int(fixed_physical_seq_len), int(fixed_physical_seq_len)),
                device=device,
                dtype=torch.bool,
            )
        prefix_rows.append(prefix)
        rope_cos_rows.append(wrapper.vision_rope_cos_const)
        rope_sin_rows.append(wrapper.vision_rope_sin_const)
        mask_rows.append(mask)
        metas.append(
            {
                "id": str(item.entry.get("id")),
                "real_seq_len": int(wrapper.static_real_seq_len),
                "physical_seq_len": int(wrapper.static_physical_seq_len),
                "pad_tokens": int(wrapper.static_pad_tokens),
                "image_grid_thw": [int(value) for value in item.image_grid_thw.flatten().tolist()],
            }
        )
    return (
        torch.stack(prefix_rows, dim=0).contiguous(),
        torch.stack(rope_cos_rows, dim=0).contiguous(),
        torch.stack(rope_sin_rows, dim=0).contiguous(),
        torch.cat(mask_rows, dim=0).contiguous(),
        metas,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, timing_default="standard")
    parser.add_argument("--baseline", default=str(Path(__file__).resolve().parent / "baselines" / "promptfa_fp16_eager_64"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "outputs" / "static_visual_batched_encoder.json"))
    parser.add_argument("--candidate-name", default="batched_encoder_candidate")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--vision-compile-backend", default="torchair", choices=VISION_COMPILE_BACKEND_CHOICES)
    parser.add_argument("--vision-use-torchair-cache-compile", action="store_true")
    parser.add_argument("--vision-torchair-cache-dir", default=str(DEFAULT_VISION_TORCHAIR_CACHE_DIR))
    parser.add_argument("--static-visual-fixed-physical-seq-len", type=int, default=1024)
    parser.add_argument("--static-visual-ln-impl", default="manual_fp32", choices=STATIC_VISUAL_LN_IMPL_CHOICES)
    parser.add_argument(
        "--static-visual-ln-linear-mode",
        default="grouped_qkv_mlp_fc1",
        choices=STATIC_VISUAL_LN_LINEAR_MODE_CHOICES,
    )
    parser.add_argument("--static-visual-promptfa-pad-head-dim-to", type=int, default=80)
    parser.add_argument(
        "--vision-linear-quantization",
        default="none",
        choices=VISION_LINEAR_QUANTIZATION_CHOICES,
    )
    parser.add_argument(
        "--w8a8-weight-layout",
        default="auto",
        choices=W8A8_WEIGHT_LAYOUT_CHOICES,
    )
    parser.add_argument(
        "--w8a8-sites",
        default=",".join(VISION_LINEAR_SITES),
        help="comma-separated subset of qkv,out_proj,fc1,fc2",
    )
    parser.add_argument("--w8a8-static-calibration-batches", type=int, default=2)
    parser.add_argument("--w8a8-static-scale-headroom", type=float, default=1.05)
    parser.add_argument("--encoder-timing-repeats", type=int, default=1)
    parser.add_argument("--warmup-encoder-first-batch", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    add_torchair_diagnostic_args(parser)
    parser.add_argument("--debug-static-visual-no-padding", action="store_true")
    parser.add_argument("--debug-static-visual-min-pad-tokens", type=int, default=0)
    parser.add_argument("--debug-static-visual-pad-to-multiple", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    apply_runtime_env(args)
    if int(args.static_visual_fixed_physical_seq_len) <= 0:
        raise ValueError("--static-visual-fixed-physical-seq-len must be >0 for batched encoder validation")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be >0")
    if int(args.encoder_timing_repeats) <= 0:
        raise ValueError("--encoder-timing-repeats must be >0")
    if str(args.vision_linear_quantization) == "w8a8_static" and int(args.w8a8_static_calibration_batches) <= 0:
        raise ValueError("static W8A8 requires --w8a8-static-calibration-batches >0")
    w8a8_sites = tuple(site.strip() for site in str(args.w8a8_sites).split(",") if site.strip())
    unknown_w8a8_sites = set(w8a8_sites) - set(VISION_LINEAR_SITES)
    if unknown_w8a8_sites:
        raise ValueError(f"unknown --w8a8-sites values: {sorted(unknown_w8a8_sites)}")

    baseline_path = Path(args.baseline).expanduser().resolve()
    manifest = load_baseline_manifest(baseline_path)
    baseline_dir = baseline_path if baseline_path.is_dir() else baseline_path.parent
    tensor_dir = baseline_dir / str(manifest.get("tensor_dir", "tensors"))
    model, model_dir, device, dtype = load_model_for_args(args)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    eos_token_id = int(model.config.eos_token_id)
    dataset_dir = resolve_dataset_dir(args.dataset_dir or manifest["build_summary"]["page"]["dataset_dir"])
    inputs = build_inputs_from_manifest(manifest=manifest, model_dir=model_dir, tokenizer=tokenizer, dataset_dir=dataset_dir)
    merge_size = int(load_preprocessor_config(model_dir)["merge_size"])
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
                    "layout_label": str(item.entry.get("layout_label", "")),
                    "vision_tokens": int(real_tokens),
                    "reason": "real_visual_tokens_exceed_fixed_physical_seq_len",
                }
            )
            continue
        eligible_pairs.append((manifest_index, item))
    if int(args.max_items) > 0:
        eligible_pairs = eligible_pairs[: int(args.max_items)]
    batchable_count = (len(eligible_pairs) // batch_size) * batch_size
    dropped_tail = eligible_pairs[batchable_count:]
    selected_pairs = eligible_pairs[:batchable_count]
    for manifest_index, item in dropped_tail:
        excluded_rows.append(
            {
                "manifest_index": int(manifest_index),
                "id": str(item.entry.get("id")),
                "layout_label": str(item.entry.get("layout_label", "")),
                "vision_tokens": int(vision_tokens(item)),
                "reason": "not_enough_items_for_full_transformer_batch",
            }
        )
    if not selected_pairs:
        raise ValueError(
            f"no full batches available after filtering: eligible_after_max_items={len(eligible_pairs)} batch_size={batch_size}"
        )
    batches = [selected_pairs[idx : idx + batch_size] for idx in range(0, len(selected_pairs), batch_size)]

    encoder_module = BatchedStaticVisualEncoderModule(
        model,
        fixed_physical_seq_len=fixed_s,
        ln_impl=str(args.static_visual_ln_impl),
        ln_linear_mode=str(args.static_visual_ln_linear_mode),
        promptfa_pad_head_dim_to=int(args.static_visual_promptfa_pad_head_dim_to),
        linear_quantization=str(args.vision_linear_quantization),
        w8a8_sites=w8a8_sites,
        w8a8_weight_layout=str(args.w8a8_weight_layout),
        w8a8_static_scale_headroom=float(args.w8a8_static_scale_headroom),
    ).eval()
    calibration_meta: dict[str, Any] = {
        "enabled": False,
        "requested_batches": int(args.w8a8_static_calibration_batches),
        "completed_batches": 0,
        "elapsed_s": 0.0,
    }
    if str(args.vision_linear_quantization) == "w8a8_static":
        encoder_module.set_calibration_enabled(True)
        calibration_batches = batches[: int(args.w8a8_static_calibration_batches)]
        maybe_sync(device)
        calibration_start = time.perf_counter()
        for calibration_pairs in calibration_batches:
            calibration_items = [item for _manifest_index, item in calibration_pairs]
            prefix, rope_cos, rope_sin, mask, _prefix_meta = build_prefix_batch(
                model=model,
                batch_items=calibration_items,
                device=device,
                fixed_physical_seq_len=fixed_s,
                ln_impl=str(args.static_visual_ln_impl),
                ln_linear_mode=str(args.static_visual_ln_linear_mode),
                promptfa_pad_head_dim_to=int(args.static_visual_promptfa_pad_head_dim_to),
                debug_no_padding=bool(args.debug_static_visual_no_padding),
                debug_min_pad_tokens=int(args.debug_static_visual_min_pad_tokens),
                debug_pad_to_multiple=int(args.debug_static_visual_pad_to_multiple),
            )
            encoder_module(prefix, rope_cos, rope_sin, mask)
        maybe_sync(device)
        encoder_module.set_calibration_enabled(False)
        calibration_meta = {
            "enabled": True,
            "requested_batches": int(args.w8a8_static_calibration_batches),
            "completed_batches": int(len(calibration_batches)),
            "elapsed_s": float(time.perf_counter() - calibration_start),
            **encoder_module.calibration_summary(),
        }
    maybe_sync(device)
    quant_prepare_start = time.perf_counter()
    quantization_meta = encoder_module.prepare_w8a8()
    maybe_sync(device)
    quantization_meta["prepare_s"] = float(time.perf_counter() - quant_prepare_start)
    quantization_meta["calibration"] = calibration_meta
    encoder_forward, compile_meta = compile_encoder_forward(
        encoder_module,
        backend_name=str(args.vision_compile_backend),
        device=device,
        use_cache_compile=bool(args.vision_use_torchair_cache_compile),
        cache_root=Path(args.vision_torchair_cache_dir).expanduser().resolve(),
        batch_size=batch_size,
        fixed_physical_seq_len=fixed_s,
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

    rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    first_call_meta: dict[str, Any] = {}
    if str(args.vision_compile_backend) != "none" or bool(args.warmup_encoder_first_batch):
        first_pairs = batches[0]
        first_items = [item for _manifest_index, item in first_pairs]
        prefix, rope_cos, rope_sin, mask, _prefix_meta = build_prefix_batch(
            model=model,
            batch_items=first_items,
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
        first_start = time.perf_counter()
        first_out = encoder_forward(prefix, rope_cos, rope_sin, mask)
        maybe_sync(device)
        first_call_meta = {
            "compiled_first_call_s": float(time.perf_counter() - first_start)
            if str(args.vision_compile_backend) != "none"
            else None,
            "eager_warmup_first_call_s": float(time.perf_counter() - first_start)
            if str(args.vision_compile_backend) == "none"
            else None,
            "first_call_kind": "compiled" if str(args.vision_compile_backend) != "none" else "eager_warmup",
            "first_output_shape": [int(dim) for dim in first_out.shape],
            "first_output_nonfinite_count": int((~torch.isfinite(first_out.float())).sum().item()),
        }
        if "capture_scalar_outputs_previous" in compile_meta:
            import torch._dynamo as torch_dynamo

            torch_dynamo.config.capture_scalar_outputs = bool(compile_meta["capture_scalar_outputs_previous"])
            first_call_meta["capture_scalar_outputs_restored_after_first_call"] = True

    total_effective_tokens = 0
    total_physical_tokens = 0
    total_encoder_s = 0.0
    total_prefix_s = 0.0
    total_visual_s = 0.0
    for batch_idx, batch_pairs in enumerate(batches):
        manifest_indices = [int(manifest_index) for manifest_index, _item in batch_pairs]
        batch_items = [item for _manifest_index, item in batch_pairs]
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
        prefix_s = float(time.perf_counter() - prefix_start)
        maybe_sync(device)
        encoder_start = time.perf_counter()
        for _repeat_index in range(int(args.encoder_timing_repeats)):
            physical_outputs = encoder_forward(prefix, rope_cos, rope_sin, mask)
        maybe_sync(device)
        encoder_repeated_s = float(time.perf_counter() - encoder_start)
        encoder_s = float(encoder_repeated_s / int(args.encoder_timing_repeats))
        effective_tokens = int(sum(vision_tokens(item) for item in batch_items))
        physical_tokens = int(batch_size * fixed_s)
        total_effective_tokens += effective_tokens
        total_physical_tokens += physical_tokens
        total_encoder_s += encoder_s
        total_prefix_s += prefix_s
        total_visual_s += prefix_s + encoder_s
        batch_rows.append(
            {
                "batch_index": int(batch_idx),
                "manifest_indices": manifest_indices,
                "ids": [str(item.entry.get("id")) for item in batch_items],
                "effective_tokens": int(effective_tokens),
                "physical_tokens": int(physical_tokens),
                "prefix_build_s": float(prefix_s),
                "batched_encoder_s": float(encoder_s),
                "batched_encoder_repeated_s": float(encoder_repeated_s),
                "encoder_timing_repeats": int(args.encoder_timing_repeats),
                "prefix_plus_encoder_s": float(prefix_s + encoder_s),
                "encoder_effective_tokens_per_s": float(effective_tokens / encoder_s) if encoder_s > 0 else None,
                "encoder_physical_tokens_per_s": float(physical_tokens / encoder_s) if encoder_s > 0 else None,
                "prefix_meta": clean_json(prefix_meta),
                "output_nonfinite_count": int((~torch.isfinite(physical_outputs.float())).sum().item()),
            }
        )
        for local_idx, (manifest_index, item) in enumerate(batch_pairs):
            baseline_item = manifest["items"][manifest_index]
            tensor_path = tensor_dir / str(baseline_item["tensor_file"])
            if sha256_file(tensor_path) != str(baseline_item["tensor_sha256"]):
                raise RuntimeError(f"baseline tensor sha256 mismatch: {tensor_path}")
            baseline_payload = torch.load(tensor_path, map_location="cpu")
            baseline_tensors = baseline_payload["tensors"]
            real_tokens = int(vision_tokens(item))
            candidate_visual = physical_outputs[local_idx, :real_tokens, :].detach()
            reference_visual = baseline_tensors["visual_features"].to(device=device, dtype=model.visual.dtype)
            candidate_prefill = compute_prefill_state_from_visual_features(
                model=model,
                item=item,
                device=device,
                cache_length=int(args.cache_length),
                visual_features=candidate_visual,
            )
            reference_prefill = compute_prefill_state_from_visual_features(
                model=model,
                item=item,
                device=device,
                cache_length=int(args.cache_length),
                visual_features=reference_visual,
            )
            reference_topk = topk_summary(reference_prefill["prefill_logits"])
            candidate_topk = topk_summary(candidate_prefill["prefill_logits"])
            generation: dict[str, Any] = {"skipped": bool(args.skip_generation)}
            texts: dict[str, Any] = {"skipped": bool(args.skip_generation)}
            if not bool(args.skip_generation):
                reference_ids = generate_from_prefill_state(
                    model=model,
                    prefill=reference_prefill,
                    max_new_tokens=int(args.max_new_tokens),
                    eos_token_id=eos_token_id,
                )
                candidate_ids = generate_from_prefill_state(
                    model=model,
                    prefill=candidate_prefill,
                    max_new_tokens=int(args.max_new_tokens),
                    eos_token_id=eos_token_id,
                )
                reference_tokens = trim_after_eos(
                    [int(value) for value in reference_ids[0].detach().cpu().tolist()],
                    eos_token_id,
                )
                candidate_tokens = trim_after_eos(
                    [int(value) for value in candidate_ids[0].detach().cpu().tolist()],
                    eos_token_id,
                )
                reference_text, reference_invalid = decode_safely(tokenizer, reference_tokens)
                candidate_text, candidate_invalid = decode_safely(tokenizer, candidate_tokens)
                generation = {
                    "skipped": False,
                    "reference_trimmed_token_count": int(len(reference_tokens)),
                    "candidate_trimmed_token_count": int(len(candidate_tokens)),
                    "generated_trimmed_match": bool(reference_tokens == candidate_tokens),
                    "first_mismatch": first_mismatch(reference_tokens, candidate_tokens),
                    "invalid_token_count": int(len(reference_invalid) + len(candidate_invalid)),
                    "reference_length_cap_hit": bool(eos_token_id not in reference_tokens and len(reference_tokens) >= int(args.max_new_tokens)),
                    "candidate_length_cap_hit": bool(eos_token_id not in candidate_tokens and len(candidate_tokens) >= int(args.max_new_tokens)),
                }
                texts = {
                    "skipped": False,
                    "reference": reference_text,
                    "candidate": candidate_text,
                    "match": bool(reference_text == candidate_text),
                    "ground_truth_sample": str(item.entry.get("ground_truth", ""))[:500],
                }
            diffs = {
                "visual_features": diff_stats(candidate_visual.cpu(), baseline_tensors["visual_features"]),
                "image_embeds": diff_stats(candidate_prefill["image_embeds"].cpu(), reference_prefill["image_embeds"].cpu()),
                "prefill_logits": diff_stats(candidate_prefill["prefill_logits"].cpu(), reference_prefill["prefill_logits"].cpu()),
            }
            rows.append(
                {
                    "index": int(manifest_index),
                    "batch_index": int(batch_idx),
                    "batch_local_index": int(local_idx),
                    **input_row(item, merge_size=merge_size),
                    "candidate_physical_vision_tokens": int(fixed_s),
                    "diffs": diffs,
                    "reference_topk": reference_topk,
                    "candidate_topk": candidate_topk,
                    "argmax_match": bool(int(reference_topk["argmax"]) == int(candidate_topk["argmax"])),
                    "generation": generation,
                    "texts": texts,
                    "candidate_visual_nonfinite_count": int((~torch.isfinite(candidate_visual.float())).sum().item()),
                }
            )
        print(
            f"BATCH {batch_idx + 1}/{len(batches)} effective_tokens={effective_tokens} "
            f"physical_tokens={physical_tokens} encoder_s={encoder_s:.4f} prefix_s={prefix_s:.4f}",
            flush=True,
        )

    compile_meta.update(first_call_meta)
    output = {
        "schema_version": 1,
        "experiment": "07_vision_prefill_optimization",
        "kind": "static_visual_batched_encoder_validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "name": str(args.candidate_name),
            "device": str(device),
            "dtype": str(dtype),
            "batch_size": int(batch_size),
            "vision_compile_backend": str(args.vision_compile_backend),
            "compile_api": compile_meta.get("compile_api"),
            "uses_torchair_cache_compile": bool(compile_meta.get("uses_torchair_cache_compile", False)),
            "vision_attention": get_vision_attention_impl(),
            "vision_prompt_fa_layout": get_vision_prompt_fa_layout(),
            "vision_prompt_fa_mask_sparse_mode": int(args.vision_prompt_fa_mask_sparse_mode),
            "static_visual_fixed_physical_seq_len": int(fixed_s),
            "static_visual_ln_impl": str(args.static_visual_ln_impl),
            "static_visual_ln_linear_mode": str(args.static_visual_ln_linear_mode),
            "static_visual_promptfa_pad_head_dim_to": int(args.static_visual_promptfa_pad_head_dim_to),
            "vision_linear_quantization": str(args.vision_linear_quantization),
            "w8a8_sites": list(encoder_module.w8a8_sites),
            "w8a8_weight_layout_requested": str(args.w8a8_weight_layout),
            "w8a8_weight_layout_resolved": str(encoder_module.w8a8_weight_layout),
            "w8a8_static_calibration_batches": int(args.w8a8_static_calibration_batches),
            "w8a8_static_scale_headroom": float(args.w8a8_static_scale_headroom),
            "warmup_encoder_first_batch": bool(args.warmup_encoder_first_batch),
            "encoder_timing_repeats": int(args.encoder_timing_repeats),
            "batched_boundary": "encoder_layers_plus_post_layernorm_only",
            "prefix_boundary": "per_crop_patch_embedding_plus_abs_pos_plus_padding_outside_compile",
            "max_new_tokens": int(args.max_new_tokens),
            "skip_generation": bool(args.skip_generation),
        },
        "compile": clean_json(compile_meta),
        "quantization": clean_json(quantization_meta),
        "baseline": {
            "path": str(baseline_path),
            "item_count": int(manifest.get("item_count", len(manifest.get("items", [])))),
        },
        "compared_count": int(len(rows)),
        "summary": {
            "batch_count": int(len(batch_rows)),
            "batch_size": int(batch_size),
            "argmax_match_count": int(sum(bool(row["argmax_match"]) for row in rows)),
            "visual_nonfinite_item_count": int(sum(int(row["candidate_visual_nonfinite_count"]) > 0 for row in rows)),
            "generated_trimmed_match_count": int(
                sum(bool(row.get("generation", {}).get("generated_trimmed_match", False)) for row in rows)
            )
            if not bool(args.skip_generation)
            else None,
            "text_match_count": int(sum(bool(row.get("texts", {}).get("match", False)) for row in rows))
            if not bool(args.skip_generation)
            else None,
            "invalid_token_count": int(sum(int(row.get("generation", {}).get("invalid_token_count", 0)) for row in rows)),
            "length_cap_hit_count": int(
                sum(
                    bool(
                        row.get("generation", {}).get("reference_length_cap_hit", False)
                        or row.get("generation", {}).get("candidate_length_cap_hit", False)
                    )
                    for row in rows
                )
            ),
            "bucket_filter": {
                "fixed_physical_seq_len": int(fixed_s),
                "manifest_item_count": int(len(inputs)),
                "eligible_count_before_max_items": int(
                    len([item for item in inputs if int(vision_tokens(item)) <= fixed_s])
                ),
                "selected_count": int(len(rows)),
                "excluded_count": int(len(excluded_rows)),
                "excluded_reason_counts": dict(sorted(Counter(row["reason"] for row in excluded_rows).items())),
                "first_excluded": clean_json(excluded_rows[:16]),
            },
            "encoder_effective_tokens_per_s": float(total_effective_tokens / total_encoder_s) if total_encoder_s > 0 else None,
            "encoder_physical_tokens_per_s": float(total_physical_tokens / total_encoder_s) if total_encoder_s > 0 else None,
            "prefix_plus_encoder_effective_tokens_per_s": float(total_effective_tokens / total_visual_s) if total_visual_s > 0 else None,
            "prefix_plus_encoder_physical_tokens_per_s": float(total_physical_tokens / total_visual_s) if total_visual_s > 0 else None,
            "total_effective_tokens": int(total_effective_tokens),
            "total_physical_tokens": int(total_physical_tokens),
            "total_prefix_build_s": float(total_prefix_s),
            "total_batched_encoder_s": float(total_encoder_s),
            "total_prefix_plus_encoder_s": float(total_visual_s),
            "batch_timing_s": {
                "prefix_build_s": stats([float(row["prefix_build_s"]) for row in batch_rows]),
                "batched_encoder_s": stats([float(row["batched_encoder_s"]) for row in batch_rows]),
                "prefix_plus_encoder_s": stats([float(row["prefix_plus_encoder_s"]) for row in batch_rows]),
            },
            "visual_features": {
                "max_abs_diff": stats(
                    [
                        float(row["diffs"]["visual_features"]["max_abs_diff"])
                        for row in rows
                        if row["diffs"]["visual_features"].get("max_abs_diff") is not None
                    ]
                )
            },
            "prefill_logits": {
                "max_abs_diff": stats(
                    [
                        float(row["diffs"]["prefill_logits"]["max_abs_diff"])
                        for row in rows
                        if row["diffs"]["prefill_logits"].get("max_abs_diff") is not None
                    ]
                )
            },
            "first_generation_mismatches": [
                {
                    "index": int(row["index"]),
                    "id": str(row["id"]),
                    "first_mismatch": row.get("generation", {}).get("first_mismatch"),
                    "reference_text": str(row.get("texts", {}).get("reference", ""))[:300],
                    "candidate_text": str(row.get("texts", {}).get("candidate", ""))[:300],
                }
                for row in rows
                if not bool(row.get("generation", {}).get("generated_trimmed_match", True))
            ][:8],
        },
        "batches": clean_json(batch_rows),
        "items": clean_json(rows),
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    print(json.dumps({"batched_encoder_output": str(output_path), "summary": output["summary"]}, indent=2, default=json_default))


if __name__ == "__main__":
    main()
