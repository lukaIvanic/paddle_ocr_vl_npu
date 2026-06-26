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
    ):
        super().__init__()
        if ln_impl not in STATIC_VISUAL_LN_IMPL_CHOICES:
            raise ValueError(f"unsupported LayerNorm impl={ln_impl!r}")
        if ln_linear_mode not in STATIC_VISUAL_LN_LINEAR_MODE_CHOICES:
            raise ValueError(f"unsupported LN-linear mode={ln_linear_mode!r}")
        self.model = model
        self.fixed_physical_seq_len = int(fixed_physical_seq_len)
        self.ln_impl = str(ln_impl)
        self.ln_linear_mode = str(ln_linear_mode)
        self.promptfa_pad_head_dim_to = int(promptfa_pad_head_dim_to)

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

    def _qkv_projection(self, attention: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
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

    def _mlp(self, mlp: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.ln_linear_mode == "grouped_qkv_mlp_fc1":
            fc1_out = self._linear_maybe_grouped(hidden_states, mlp.fc1.weight, mlp.fc1.bias)
        else:
            fc1_out = mlp.fc1(hidden_states)
        return mlp.fc2(_activation(mlp.hidden_act, fc1_out))

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
    ) -> torch.Tensor:
        batch_size, seq_length, _hidden = hidden_states.shape
        qkv = self._qkv_projection(attention, hidden_states)
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
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        return attention.out_proj(attn_output)

    def _encoder_layer(
        self,
        encoder_layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_input = self._static_layer_norm(encoder_layer.layer_norm1, hidden_states)
        hidden_states = hidden_states + self._attention(
            encoder_layer.self_attn,
            attn_input,
            rope_cos,
            rope_sin,
            attention_mask,
        )
        mlp_input = self._static_layer_norm(encoder_layer.layer_norm2, hidden_states)
        return hidden_states + self._mlp(encoder_layer.mlp, mlp_input)

    def forward(
        self,
        prefix_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = prefix_hidden_states
        transformer = self.model.visual.vision_model
        for encoder_layer in transformer.encoder.layers:
            hidden_states = self._encoder_layer(encoder_layer, hidden_states, rope_cos, rope_sin, attention_mask)
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
    return base.parent / f"encoder_only_B{int(batch_size)}_{base.name}"


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
    import torch._dynamo

    old_capture_scalar_outputs = bool(torch._dynamo.config.capture_scalar_outputs)
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.reset()
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

    encoder_module = BatchedStaticVisualEncoderModule(
        model,
        fixed_physical_seq_len=fixed_s,
        ln_impl=str(args.static_visual_ln_impl),
        ln_linear_mode=str(args.static_visual_ln_linear_mode),
        promptfa_pad_head_dim_to=int(args.static_visual_promptfa_pad_head_dim_to),
    ).eval()
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

    batches = [selected_pairs[idx : idx + batch_size] for idx in range(0, len(selected_pairs), batch_size)]
    rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    first_call_meta: dict[str, Any] = {}
    if str(args.vision_compile_backend) != "none":
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
            "compiled_first_call_s": float(time.perf_counter() - first_start),
            "first_output_shape": [int(dim) for dim in first_out.shape],
            "first_output_nonfinite_count": int((~torch.isfinite(first_out.float())).sum().item()),
        }
        if "capture_scalar_outputs_previous" in compile_meta:
            import torch._dynamo

            torch._dynamo.config.capture_scalar_outputs = bool(compile_meta["capture_scalar_outputs_previous"])
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
        physical_outputs = encoder_forward(prefix, rope_cos, rope_sin, mask)
        maybe_sync(device)
        encoder_s = float(time.perf_counter() - encoder_start)
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
            "batched_boundary": "encoder_layers_plus_post_layernorm_only",
            "prefix_boundary": "per_crop_patch_embedding_plus_abs_pos_plus_padding_outside_compile",
            "max_new_tokens": int(args.max_new_tokens),
            "skip_generation": bool(args.skip_generation),
        },
        "compile": clean_json(compile_meta),
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
