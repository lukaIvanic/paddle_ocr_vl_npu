"""Unified eager and compiled model execution for text decode."""

from __future__ import annotations

import hashlib
import importlib
import json
import time
import types
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from _support.model.compile_utils import TORCHAIR_EXECUTION_MODE, cache_key_part, compile_backend, import_torchair, short_file_hash, torch_npu_version_label, torchair_version_label
from _support.model.decode_token_embedding import decode_token_embedding, register_decode_token_embedding_converter
from _support.model.decode_linear_matmul_v3 import decode_linear_matmul_v3, register_decode_linear_matmul_v3_converter
from _support.model.decode_qkv_split import decode_qkv_split, register_decode_qkv_split_converter
from _support.model.decode_position_add import decode_position_add, register_decode_position_add_converter
from _support.model.decode_rope_lookup import decode_rope_lookup, register_decode_rope_lookup_converter
from _support.model.decode_kv_scatter import decode_kv_scatter, register_decode_kv_scatter_converter
from _support.model.decode_kv_scatter_query import decode_kv_scatter_query, register_decode_kv_scatter_query_converter
from _support.model.decode_gqa_increfa_aiv import decode_gqa_incre_flash_attention_aiv, register_decode_gqa_increfa_aiv_converter
from _support.model.decode_gqa_increfa_mixed import decode_gqa_incre_flash_attention_mixed, decode_gqa_incre_flash_attention_mixed24, register_decode_gqa_increfa_mixed_converter, register_decode_gqa_increfa_mixed24_converter
from _support.model.decode_packed_qkv_rope_gqa_mixed24 import decode_packed_qkv_rope_gqa_mixed24, register_decode_packed_qkv_rope_gqa_mixed24_converter
from _support.model.decode_gqa_attention_aiv import decode_gqa_attention_aiv, register_decode_gqa_attention_aiv_converter
from _support.model.decode_gqa_attention_mixed24 import decode_gqa_attention_mixed24, register_decode_gqa_attention_mixed24_converter
from _support.model.decode_swiglu import decode_swiglu, register_decode_swiglu_converter
from _support.model.config import PaddleOCRTextConfig
from _support.model.gqa_increfa_aiv import gqa_incre_flash_attention_aiv, register_gqa_increfa_aiv_converter
from _support.utils.timing import synchronize

if TYPE_CHECKING:
    from paddle_ocr_vl_1_6_modeling import LocalPaddleOCRVLForConditionalGeneration


FRACTAL_NZ = 29
DECODE_LINEAR_WEIGHT_FORMAT = "decode_nz"
DECODE_LINEAR_WEIGHT_FALLBACK = "decode_native_fallback"
DECODE_ATTENTION = "increfa"
DECODE_CACHE_UPDATE = "npu_scatter"


@dataclass(frozen=True)
class DecodeOptimizationConfig:
    """Experimental implementation choices for the text-decode lab.

    Production callers keep the baseline defaults.  The lab selects one
    named preset so every compiled graph has an explicit, reproducible
    implementation contract.
    """

    name: str
    hoist_mrope: bool = False
    packed_qkv: bool = False
    rms_norm: str = "manual"
    rotary: str = "manual"
    rotary_factors: str = "mrope"
    packed_mlp: bool = False
    npu_swiglu: bool = False
    add_rms_norm: bool = False
    attention: str = "gqa"
    increfa_length_mode: str = "mask"
    # None preserves the torch-npu production default by omitting the keyword.
    # Experimental precision lanes can explicitly select 0 (high precision).
    increfa_inner_precise: int | None = None
    stage_aware_weight_prefetch: bool = False
    post_scatter_kv_prefetch: bool = False
    weight_prefetch_timing: str = "before_attention"
    complete_layer_prefetch_ahead: int = 0
    prefetch_next_iteration: bool = False
    zero_residual_first_rms_norm: bool = False
    packed_kv_scatter: bool = False
    vector_add_rms_norm: bool = False
    gqa_aiv_vector_core_count: int = 0
    super_kernel_scope: bool = False
    super_kernel_options: str = (
        "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
        "preload-code=per-func:early-start=1:split-mode=4"
    )
    ascendc_token_embedding: bool = False
    ascendc_linear: bool = False
    ascendc_qkv_split: bool = False
    ascendc_position_add: bool = False
    ascendc_rope_lookup: bool = False
    ascendc_kv_scatter: bool = False
    ascendc_kv_scatter_query: bool = False
    ascendc_decode_gqa: bool = False
    ascendc_decode_gqa_mixed: bool = False
    ascendc_decode_gqa_mixed24: bool = False
    ascendc_packed_qkv_rope_gqa_mixed24: bool = False
    ascendc_decode_gqa_attention: bool = False
    ascendc_decode_gqa_attention_mixed24: bool = False
    ascendc_swiglu: bool = False


DECODE_OPTIMIZATION_PRESETS: dict[str, DecodeOptimizationConfig] = {
    "baseline": DecodeOptimizationConfig(name="baseline"),
    "mrope_hoist": DecodeOptimizationConfig(
        name="mrope_hoist",
        hoist_mrope=True,
    ),
    "packed_qkv": DecodeOptimizationConfig(
        name="packed_qkv",
        packed_qkv=True,
    ),
    "npu_rms_norm": DecodeOptimizationConfig(
        name="npu_rms_norm",
        rms_norm="npu",
    ),
    "npu_apply_rotary": DecodeOptimizationConfig(
        name="npu_apply_rotary",
        hoist_mrope=True,
        rotary="npu_apply",
    ),
    "npu_rotary_mul": DecodeOptimizationConfig(
        name="npu_rotary_mul",
        hoist_mrope=True,
        rotary="npu_rotary_mul",
    ),
    "packed_mlp": DecodeOptimizationConfig(
        name="packed_mlp",
        packed_mlp=True,
    ),
    "packed_mlp_swiglu": DecodeOptimizationConfig(
        name="packed_mlp_swiglu",
        packed_mlp=True,
        npu_swiglu=True,
    ),
    "npu_add_rms_norm": DecodeOptimizationConfig(
        name="npu_add_rms_norm",
        rms_norm="npu",
        add_rms_norm=True,
    ),
    "combined_apply": DecodeOptimizationConfig(
        name="combined_apply",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
    ),
    "combined_apply_spec_prefetch_mrope": DecodeOptimizationConfig(
        name="combined_apply_spec_prefetch_mrope",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="mrope",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
        post_scatter_kv_prefetch=True,
        weight_prefetch_timing="complete_layer_ahead",
        complete_layer_prefetch_ahead=1,
    ),
    "combined_apply_spec_prefetch_rope_lut": DecodeOptimizationConfig(
        name="combined_apply_spec_prefetch_rope_lut",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
        post_scatter_kv_prefetch=True,
        weight_prefetch_timing="complete_layer_ahead",
        complete_layer_prefetch_ahead=1,
    ),
    "combined_apply_superkernel_b1": DecodeOptimizationConfig(
        name="combined_apply_superkernel_b1",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        super_kernel_scope=True,
        ascendc_token_embedding=True,
        ascendc_linear=True,
        ascendc_qkv_split=True,
        ascendc_rope_lookup=True,
        ascendc_kv_scatter=True,
    ),
    "paddle_decoder_megakernel_b1": DecodeOptimizationConfig(
        name="paddle_decoder_megakernel_b1",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        attention="gqa_aiv",
        gqa_aiv_vector_core_count=16,
        super_kernel_scope=True,
        ascendc_token_embedding=True,
        ascendc_linear=True,
        ascendc_qkv_split=True,
        ascendc_rope_lookup=True,
        ascendc_kv_scatter_query=True,
        ascendc_swiglu=True,
    ),
    "combined_apply_pse_sentinel": DecodeOptimizationConfig(
        name="combined_apply_pse_sentinel",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        increfa_length_mode="pse_sentinel",
    ),
    "combined_apply_pse_sentinel_high_precision": DecodeOptimizationConfig(
        name="combined_apply_pse_sentinel_high_precision",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        increfa_length_mode="pse_sentinel",
        increfa_inner_precise=0,
    ),
    "combined_apply_static_actual": DecodeOptimizationConfig(
        name="combined_apply_static_actual",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        increfa_length_mode="static_actual",
    ),
    "combined_apply_mha_repeat": DecodeOptimizationConfig(
        name="combined_apply_mha_repeat",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="mha_repeat",
    ),
    "combined_apply_gqa_aiv_b1": DecodeOptimizationConfig(
        name="combined_apply_gqa_aiv_b1",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="gqa_aiv",
        gqa_aiv_vector_core_count=16,
    ),
    "combined_apply_gqa_aiv_b1_split_k32_control": DecodeOptimizationConfig(
        name="combined_apply_gqa_aiv_b1_split_k32_control",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="gqa_aiv",
        gqa_aiv_vector_core_count=32,
    ),
    "combined_apply_gqa_aiv_b1_split_k32_pairwise_sync_control": DecodeOptimizationConfig(
        name="combined_apply_gqa_aiv_b1_split_k32_pairwise_sync_control",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="gqa_aiv",
        gqa_aiv_vector_core_count=32,
    ),
    "combined_apply_gqa_aiv_b1_split_k32_two_way_reduce_control": DecodeOptimizationConfig(
        name="combined_apply_gqa_aiv_b1_split_k32_two_way_reduce_control",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="gqa_aiv",
        gqa_aiv_vector_core_count=32,
    ),
    "combined_apply_gqa_aiv_b1_split_k32_local_partial_reduce_control": DecodeOptimizationConfig(
        name="combined_apply_gqa_aiv_b1_split_k32_local_partial_reduce_control",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="gqa_aiv",
        gqa_aiv_vector_core_count=32,
    ),
    "combined_apply_gqa_aiv_b1_split_k48_control": DecodeOptimizationConfig(
        name="combined_apply_gqa_aiv_b1_split_k48_control",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="gqa_aiv",
        gqa_aiv_vector_core_count=48,
    ),
    "combined_apply_mha_cache": DecodeOptimizationConfig(
        name="combined_apply_mha_cache",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="mha_cache",
    ),
    "combined_apply_mha_cache_prefetch_kv": DecodeOptimizationConfig(
        name="combined_apply_mha_cache_prefetch_kv",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="mha_cache",
        post_scatter_kv_prefetch=True,
    ),
    "combined_apply_manual_attention": DecodeOptimizationConfig(
        name="combined_apply_manual_attention",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="manual",
    ),
    "combined_apply_manual_attention_unscaled": DecodeOptimizationConfig(
        name="combined_apply_manual_attention_unscaled",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="manual_unscaled",
    ),
    "combined_apply_prefetch": DecodeOptimizationConfig(
        name="combined_apply_prefetch",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
    ),
    "combined_apply_prefetch_scalar_rope": DecodeOptimizationConfig(
        name="combined_apply_prefetch_scalar_rope",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="scalar",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
    ),
    "combined_apply_prefetch_rope_lut": DecodeOptimizationConfig(
        name="combined_apply_prefetch_rope_lut",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
    ),
    "combined_apply_prefetch_rope_lut_pseudo_b2": DecodeOptimizationConfig(
        name="combined_apply_prefetch_rope_lut_pseudo_b2",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        attention="gqa_pseudo_b2",
        stage_aware_weight_prefetch=True,
    ),
    "combined_apply_pse_prefetch_rope_lut": DecodeOptimizationConfig(
        name="combined_apply_pse_prefetch_rope_lut",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        increfa_length_mode="pse_sentinel",
        stage_aware_weight_prefetch=True,
    ),
    "combined_apply_kv_prefetch_before_mlp_rope_lut": DecodeOptimizationConfig(
        name="combined_apply_kv_prefetch_before_mlp_rope_lut",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
        post_scatter_kv_prefetch=True,
    ),
    "combined_apply_kv_prefetch_rope_lut": DecodeOptimizationConfig(
        name="combined_apply_kv_prefetch_rope_lut",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        post_scatter_kv_prefetch=True,
    ),
    "combined_apply_kv_then_mlp_prefetch_rope_lut": DecodeOptimizationConfig(
        name="combined_apply_kv_then_mlp_prefetch_rope_lut",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
        post_scatter_kv_prefetch=True,
        weight_prefetch_timing="after_attention",
    ),
    "combined_apply_complete_layer_prefetch1_rope_lut": DecodeOptimizationConfig(
        name="combined_apply_complete_layer_prefetch1_rope_lut",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
        post_scatter_kv_prefetch=True,
        weight_prefetch_timing="complete_layer_ahead",
        complete_layer_prefetch_ahead=1,
    ),
    "combined_apply_mixed_m16": DecodeOptimizationConfig(
        name="combined_apply_mixed_m16",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
        post_scatter_kv_prefetch=True,
        weight_prefetch_timing="complete_layer_ahead",
        complete_layer_prefetch_ahead=1,
    ),
    "combined_apply_complete_layer_prefetch1_rope_lut_high_precision": DecodeOptimizationConfig(
        name="combined_apply_complete_layer_prefetch1_rope_lut_high_precision",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        increfa_inner_precise=0,
        stage_aware_weight_prefetch=True,
        post_scatter_kv_prefetch=True,
        weight_prefetch_timing="complete_layer_ahead",
        complete_layer_prefetch_ahead=1,
    ),
    "combined_apply_complete_layer_prefetch2_rope_lut": DecodeOptimizationConfig(
        name="combined_apply_complete_layer_prefetch2_rope_lut",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
        post_scatter_kv_prefetch=True,
        weight_prefetch_timing="complete_layer_ahead",
        complete_layer_prefetch_ahead=2,
    ),
    "combined_apply_complete_layer_prefetch1_rope_lut_zero_first_norm": DecodeOptimizationConfig(
        name="combined_apply_complete_layer_prefetch1_rope_lut_zero_first_norm",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
        post_scatter_kv_prefetch=True,
        weight_prefetch_timing="complete_layer_ahead",
        complete_layer_prefetch_ahead=1,
        zero_residual_first_rms_norm=True,
    ),
    "combined_apply_complete_layer_prefetch1_rope_lut_packed_kv": DecodeOptimizationConfig(
        name="combined_apply_complete_layer_prefetch1_rope_lut_packed_kv",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
        post_scatter_kv_prefetch=True,
        weight_prefetch_timing="complete_layer_ahead",
        complete_layer_prefetch_ahead=1,
        packed_kv_scatter=True,
    ),
    "combined_apply_complete_layer_prefetch1_rope_lut_zero_first_norm_packed_kv": DecodeOptimizationConfig(
        name="combined_apply_complete_layer_prefetch1_rope_lut_zero_first_norm_packed_kv",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
        post_scatter_kv_prefetch=True,
        weight_prefetch_timing="complete_layer_ahead",
        complete_layer_prefetch_ahead=1,
        zero_residual_first_rms_norm=True,
        packed_kv_scatter=True,
    ),
    "combined_apply_kv_then_mlp_prefetch_rope_lut_pseudo_b2": DecodeOptimizationConfig(
        name="combined_apply_kv_then_mlp_prefetch_rope_lut_pseudo_b2",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        attention="gqa_pseudo_b2",
        stage_aware_weight_prefetch=True,
        post_scatter_kv_prefetch=True,
        weight_prefetch_timing="after_attention",
    ),
    "combined_apply_prefetch_rope_lut_no_norm": DecodeOptimizationConfig(
        name="combined_apply_prefetch_rope_lut_no_norm",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="identity",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
    ),
    "combined_apply_prefetch_rope_lut_vector_norm": DecodeOptimizationConfig(
        name="combined_apply_prefetch_rope_lut_vector_norm",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        rotary_factors="lookup",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
        vector_add_rms_norm=True,
    ),
    "combined_apply_prefetch_no_rope": DecodeOptimizationConfig(
        name="combined_apply_prefetch_no_rope",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="identity",
        add_rms_norm=True,
        stage_aware_weight_prefetch=True,
    ),
    "combined_apply_prefetch_no_increfa": DecodeOptimizationConfig(
        name="combined_apply_prefetch_no_increfa",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        add_rms_norm=True,
        attention="no_increfa",
        stage_aware_weight_prefetch=True,
    ),
    "combined_apply_all": DecodeOptimizationConfig(
        name="combined_apply_all",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_apply",
        packed_mlp=True,
        npu_swiglu=True,
        add_rms_norm=True,
    ),
    "combined_rotary_mul": DecodeOptimizationConfig(
        name="combined_rotary_mul",
        hoist_mrope=True,
        packed_qkv=True,
        rms_norm="npu",
        rotary="npu_rotary_mul",
        add_rms_norm=True,
    ),
}

# Retain explicit compiler-layout controls for the one-task decoder experiment.
# The production-named preset above uses the installed CANN 9.0 defaults for
# code splitting, code preload, and early start.  These variants isolate the
# three controls without relaxing strict scope checking or changing the graph.
_PADDLE_DECODER_MEGAKERNEL_B1 = DECODE_OPTIMIZATION_PRESETS[
    "paddle_decoder_megakernel_b1"
]
DECODE_OPTIMIZATION_PRESETS.update(
    {
        "paddle_decoder_megakernel_b1_split1_nopreload": replace(
            _PADDLE_DECODER_MEGAKERNEL_B1,
            name="paddle_decoder_megakernel_b1_split1_nopreload",
            super_kernel_options=(
                "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
                "preload-code=none:early-start=0:split-mode=1"
            ),
        ),
        "paddle_decoder_megakernel_b1_split4_nopreload": replace(
            _PADDLE_DECODER_MEGAKERNEL_B1,
            name="paddle_decoder_megakernel_b1_split4_nopreload",
            super_kernel_options=(
                "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
                "preload-code=none:early-start=0:split-mode=4"
            ),
        ),
        "paddle_decoder_megakernel_b1_split4_perfunc_early0": replace(
            _PADDLE_DECODER_MEGAKERNEL_B1,
            name="paddle_decoder_megakernel_b1_split4_perfunc_early0",
            super_kernel_options=(
                "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
                "preload-code=per-func:early-start=0:split-mode=4"
            ),
        ),
        "paddle_decoder_megakernel_b1_feed_sync_split1": replace(
            _PADDLE_DECODER_MEGAKERNEL_B1,
            name="paddle_decoder_megakernel_b1_feed_sync_split1",
            super_kernel_options=(
                "feed-sync-all=1:stream-fusion=0:strict-scope-check=abort:"
                "preload-code=none:early-start=0:split-mode=1"
            ),
        ),
        "paddle_decoder_megakernel_b1_feed_sync": replace(
            _PADDLE_DECODER_MEGAKERNEL_B1,
            name="paddle_decoder_megakernel_b1_feed_sync",
            super_kernel_options=(
                "feed-sync-all=1:stream-fusion=0:strict-scope-check=abort:"
                "preload-code=per-func:early-start=1:split-mode=4"
            ),
        ),
        "paddle_decoder_megakernel_b1_fused_gqa": replace(
            _PADDLE_DECODER_MEGAKERNEL_B1,
            name="paddle_decoder_megakernel_b1_fused_gqa",
            ascendc_kv_scatter_query=False,
            ascendc_decode_gqa=True,
            super_kernel_options=(
                "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
                "preload-code=per-func:early-start=1:split-mode=4"
            ),
        ),
    }
)

_PADDLE_DECODER_MEGAKERNEL_B1_FUSED_GQA = DECODE_OPTIMIZATION_PRESETS[
    "paddle_decoder_megakernel_b1_fused_gqa"
]
DECODE_OPTIMIZATION_PRESETS.update(
    {
        "paddle_decoder_megakernel_b1_fused_gqa_split4_nopreload": replace(
            _PADDLE_DECODER_MEGAKERNEL_B1_FUSED_GQA,
            name="paddle_decoder_megakernel_b1_fused_gqa_split4_nopreload",
            super_kernel_options=(
                "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
                "preload-code=none:early-start=0:split-mode=4"
            ),
        ),
        "paddle_decoder_megakernel_b1_fused_gqa_split1_nopreload": replace(
            _PADDLE_DECODER_MEGAKERNEL_B1_FUSED_GQA,
            name="paddle_decoder_megakernel_b1_fused_gqa_split1_nopreload",
            super_kernel_options=(
                "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
                "preload-code=none:early-start=0:split-mode=1"
            ),
        ),
        "paddle_decoder_megakernel_b1_fused_gqa_feed_sync": replace(
            _PADDLE_DECODER_MEGAKERNEL_B1_FUSED_GQA,
            name="paddle_decoder_megakernel_b1_fused_gqa_feed_sync",
            super_kernel_options=(
                "feed-sync-all=1:stream-fusion=0:strict-scope-check=abort:"
                "preload-code=per-func:early-start=1:split-mode=4"
            ),
        ),
        "paddle_decoder_megakernel_b1_fused_gqa_feed_sync_split1": replace(
            _PADDLE_DECODER_MEGAKERNEL_B1_FUSED_GQA,
            name="paddle_decoder_megakernel_b1_fused_gqa_feed_sync_split1",
            super_kernel_options=(
                "feed-sync-all=1:stream-fusion=0:strict-scope-check=abort:"
                "preload-code=none:early-start=0:split-mode=1"
            ),
        ),
        "paddle_decoder_megakernel_b1_nonsplit_gqa": replace(
            _PADDLE_DECODER_MEGAKERNEL_B1,
            name="paddle_decoder_megakernel_b1_nonsplit_gqa",
            gqa_aiv_vector_core_count=16,
            ascendc_decode_gqa_attention=True,
            super_kernel_options=(
                "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
                "preload-code=per-func:early-start=1:split-mode=4"
            ),
        ),
        "paddle_decoder_megakernel_b1_attention_mixed24": replace(
            _PADDLE_DECODER_MEGAKERNEL_B1,
            name="paddle_decoder_megakernel_b1_attention_mixed24",
            gqa_aiv_vector_core_count=16,
            ascendc_decode_gqa_attention_mixed24=True,
            super_kernel_options=(
                "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
                "preload-code=per-func:early-start=0:split-mode=4"
            ),
        ),
    }
)

# Keep the mixed 1:1 task-geometry operator on an independent identity and
# preset.  This makes it impossible for a full-model run to silently select
# the older zero-cube AIV operator or one of its cached TorchAir graphs.
DECODE_OPTIMIZATION_PRESETS.update(
    {
        "paddle_decoder_megakernel_b1_fused_gqa_mixed": replace(
            _PADDLE_DECODER_MEGAKERNEL_B1_FUSED_GQA,
            name="paddle_decoder_megakernel_b1_fused_gqa_mixed",
            ascendc_decode_gqa=False,
            ascendc_decode_gqa_mixed=True,
            super_kernel_options=(
                "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
                "preload-code=per-func:early-start=0:split-mode=4"
            ),
        ),
        "paddle_decoder_megakernel_b1_fused_gqa_mixed_feed_sync": replace(
            _PADDLE_DECODER_MEGAKERNEL_B1_FUSED_GQA,
            name="paddle_decoder_megakernel_b1_fused_gqa_mixed_feed_sync",
            ascendc_decode_gqa=False,
            ascendc_decode_gqa_mixed=True,
            super_kernel_options=(
                "feed-sync-all=1:stream-fusion=0:strict-scope-check=abort:"
                "preload-code=per-func:early-start=0:split-mode=4"
            ),
        ),
        "paddle_decoder_megakernel_b1_fused_gqa_mixed24": replace(
            _PADDLE_DECODER_MEGAKERNEL_B1_FUSED_GQA,
            name="paddle_decoder_megakernel_b1_fused_gqa_mixed24",
            ascendc_decode_gqa=False,
            ascendc_decode_gqa_mixed24=True,
            super_kernel_options=(
                "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
                "preload-code=per-func:early-start=0:split-mode=4"
            ),
        ),
        "paddle_decoder_superkernel_b1_simple_gqa_mixed24": replace(
            _PADDLE_DECODER_MEGAKERNEL_B1_FUSED_GQA,
            name="paddle_decoder_superkernel_b1_simple_gqa_mixed24",
            # Keep the one custom subfunction that has already passed strict
            # SuperKernel validation.  GatherV2 and the TBE
            # Add/Cast/Cos/Sin scalar-RoPE path cannot enter strict scope, so
            # retain the small AscendC embedding and lookup prologue.  SplitV
            # is also a non-fusible TBE kernel, so use the independent QKV
            # split with its explicit producer barrier.  SwishMul is also TBE,
            # so use the independent SwiGLU subfunction with the same explicit
            # producer-handoff rule.
            rotary_factors="lookup",
            ascendc_token_embedding=True,
            ascendc_qkv_split=True,
            ascendc_rope_lookup=True,
            ascendc_position_add=False,
            ascendc_swiglu=True,
            ascendc_decode_gqa=False,
            ascendc_decode_gqa_mixed24=True,
            super_kernel_options=(
                "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
                "preload-code=per-func:early-start=0:split-mode=4"
            ),
        ),
        "paddle_decoder_megakernel_b1_packed_qkv_rope_gqa_mixed24": replace(
            _PADDLE_DECODER_MEGAKERNEL_B1_FUSED_GQA,
            name="paddle_decoder_megakernel_b1_packed_qkv_rope_gqa_mixed24",
            ascendc_qkv_split=False,
            ascendc_rope_lookup=False,
            ascendc_decode_gqa=False,
            ascendc_packed_qkv_rope_gqa_mixed24=True,
            super_kernel_options=(
                "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
                "preload-code=per-func:early-start=0:split-mode=4"
            ),
        ),
    }
)


def decode_optimization_names() -> tuple[str, ...]:
    return tuple(DECODE_OPTIMIZATION_PRESETS)


DECODE_OPTIMIZATION_PRESETS[
    "combined_apply_complete_layer_prefetch1_rope_lut_loop_prefetch"
] = replace(
    DECODE_OPTIMIZATION_PRESETS["combined_apply_complete_layer_prefetch1_rope_lut"],
    name="combined_apply_complete_layer_prefetch1_rope_lut_loop_prefetch",
    prefetch_next_iteration=True,
)

DECODE_OPTIMIZATION_PRESETS[
    "combined_apply_complete_layer_prefetch1_rope_lut_packed_mlp"
] = replace(
    DECODE_OPTIMIZATION_PRESETS["combined_apply_complete_layer_prefetch1_rope_lut"],
    name="combined_apply_complete_layer_prefetch1_rope_lut_packed_mlp",
    packed_mlp=True,
    npu_swiglu=True,
)


def resolve_decode_optimization(
    optimization: str | DecodeOptimizationConfig,
) -> DecodeOptimizationConfig:
    if isinstance(optimization, DecodeOptimizationConfig):
        return optimization
    try:
        return DECODE_OPTIMIZATION_PRESETS[str(optimization)]
    except KeyError as exc:
        raise ValueError(
            f"unknown decode optimization {optimization!r}; expected one of "
            f"{decode_optimization_names()}"
        ) from exc


@dataclass
class LocalPaddleOCRVLStaticCache:
    """Fixed-shape KV tensors shared by prefill and continuous decode."""

    key_caches: tuple[torch.Tensor, ...]
    value_caches: tuple[torch.Tensor, ...]
    cache_length: int
    packed_kv_caches: tuple[torch.Tensor, ...] | None = None

    @classmethod
    def allocate(
        cls,
        config: PaddleOCRTextConfig,
        *,
        batch_size: int,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
        init_mode: str = "zeros",
        num_key_value_heads: int | None = None,
        packed_kv: bool = False,
    ) -> "LocalPaddleOCRVLStaticCache":
        cache_heads = (
            int(config.num_key_value_heads)
            if num_key_value_heads is None
            else int(num_key_value_heads)
        )
        if cache_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        cache_shape = (
            int(batch_size),
            cache_heads,
            int(cache_length),
            int(config.head_dim),
        )
        key_caches = []
        value_caches = []
        packed_kv_caches = []
        for _layer_idx in range(config.num_hidden_layers):
            if packed_kv:
                packed_shape = (2 * int(batch_size), *cache_shape[1:])
                if init_mode == "zeros":
                    packed_cache = torch.zeros(
                        packed_shape, device=device, dtype=dtype
                    )
                elif init_mode == "empty":
                    packed_cache = torch.empty(
                        packed_shape, device=device, dtype=dtype
                    )
                else:
                    raise ValueError(
                        f"unknown static cache init_mode: {init_mode!r}"
                    )
                key_cache = packed_cache[: int(batch_size)]
                value_cache = packed_cache[int(batch_size) :]
                packed_kv_caches.append(packed_cache)
            elif init_mode == "zeros":
                key_cache = torch.zeros(
                    cache_shape, device=device, dtype=dtype
                )
                value_cache = torch.zeros_like(key_cache)
            elif init_mode == "empty":
                key_cache = torch.empty(
                    cache_shape, device=device, dtype=dtype
                )
                value_cache = torch.empty_like(key_cache)
            else:
                raise ValueError(
                    f"unknown static cache init_mode: {init_mode!r}"
                )
            key_caches.append(key_cache)
            value_caches.append(value_cache)
        return cls(
            tuple(key_caches),
            tuple(value_caches),
            int(cache_length),
            tuple(packed_kv_caches) if packed_kv else None,
        )

    def layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.key_caches[int(layer_idx)],
            self.value_caches[int(layer_idx)],
        )

    def flat_tensors(self) -> tuple[torch.Tensor, ...]:
        if self.packed_kv_caches is not None:
            return self.packed_kv_caches
        return (*self.key_caches, *self.value_caches)

    def logical_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return the ordinary K-then-V view, independent of physical storage."""
        return (*self.key_caches, *self.value_caches)


def _linear_tokenwise(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Apply a Linear through a compiler-safe 2-D token matrix."""
    leading_shape = x.shape[:-1]
    token_matrix = x.reshape(-1, x.shape[-1])
    if getattr(linear, "_decode_use_matmul_v3", False):
        if linear.bias is not None:
            raise ValueError("the B1 MatMulV3 decode path requires no bias")
        output = decode_linear_matmul_v3(token_matrix, linear.weight)
    else:
        output = linear(token_matrix)
    return output.reshape(*leading_shape, output.shape[-1])


def _packed_linear(
    modules: tuple[nn.Linear, ...],
) -> nn.Linear:
    first = modules[0]
    if any(module.in_features != first.in_features for module in modules):
        raise ValueError("packed Linear inputs must share in_features")
    biases = tuple(module.bias for module in modules)
    if any(bias is None for bias in biases) and not all(
        bias is None for bias in biases
    ):
        raise ValueError("packed Linear inputs must use the same bias contract")
    packed = nn.Linear(
        first.in_features,
        sum(module.out_features for module in modules),
        bias=biases[0] is not None,
        device=first.weight.device,
        dtype=first.weight.dtype,
    )
    with torch.no_grad():
        packed.weight.copy_(
            torch.cat([module.weight for module in modules], dim=0)
        )
        if packed.bias is not None:
            packed.bias.copy_(
                torch.cat(
                    [bias for bias in biases if bias is not None],
                    dim=0,
                )
            )
    return packed


def prepare_decode_optimization_modules(
    model: "LocalPaddleOCRVLForConditionalGeneration",
    optimization: str | DecodeOptimizationConfig,
) -> DecodeOptimizationConfig:
    """Create packed projections once, before weight-format conversion."""
    config = resolve_decode_optimization(optimization)
    for layer in model.model.layers:
        attention = layer.self_attn
        if config.packed_qkv and not hasattr(
            attention, "decode_qkv_proj"
        ):
            attention.decode_qkv_proj = _packed_linear(
                (attention.q_proj, attention.k_proj, attention.v_proj)
            )
        mlp = layer.mlp
        if config.packed_mlp and not hasattr(
            mlp, "decode_gate_up_proj"
        ):
            mlp.decode_gate_up_proj = _packed_linear(
                (mlp.gate_proj, mlp.up_proj)
            )
    if config.ascendc_linear:
        linear_modules = [
            module
            for module in model.model.modules()
            if isinstance(module, nn.Linear)
        ]
        linear_modules.append(model.lm_head)
        for linear in linear_modules:
            if linear.bias is not None:
                raise ValueError(
                    "the Paddle B1 MatMulV3 mega-kernel path requires "
                    "bias-free decoder linears"
                )
            linear._decode_use_matmul_v3 = True
    return config


def load_decode_vocab_token_ids(
    path: Path,
    *,
    full_vocab_size: int,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Load an explicit native-token decode vocabulary.

    The file contains token IDs, not text.  This deliberately has no tokenizer
    dependency: generated text must never be decoded and re-encoded to build a
    compact output head.
    """
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    raw_ids = payload.get("token_ids") if isinstance(payload, dict) else payload
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError(f"decode vocabulary {resolved} has no token_ids list")
    token_ids = tuple(int(value) for value in raw_ids)
    if len(set(token_ids)) != len(token_ids):
        raise ValueError(f"decode vocabulary {resolved} contains duplicate IDs")
    invalid = [
        token_id
        for token_id in token_ids
        if not 0 <= token_id < int(full_vocab_size)
    ]
    if invalid:
        raise ValueError(
            f"decode vocabulary {resolved} has IDs outside [0, "
            f"{int(full_vocab_size)}): {invalid[:16]}"
        )
    digest_payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(digest_payload).hexdigest()
    declared_digest = (
        payload.get("token_ids_sha256")
        if isinstance(payload, dict)
        else None
    )
    if declared_digest is not None and str(declared_digest) != digest:
        raise ValueError(
            f"decode vocabulary {resolved} digest mismatch: "
            f"declared={declared_digest} actual={digest}"
        )
    metadata = {
        "enabled": True,
        "path": str(resolved),
        "full_vocab_size": int(full_vocab_size),
        "selected_vocab_size": len(token_ids),
        "token_ids_sha256": digest,
        "source": payload.get("source") if isinstance(payload, dict) else None,
        "selection": (
            payload.get("selection")
            if isinstance(payload, dict)
            else None
        ),
    }
    return token_ids, metadata


def prepare_decode_compact_lm_head(
    model: "LocalPaddleOCRVLForConditionalGeneration",
    token_ids: tuple[int, ...],
) -> nn.Linear:
    """Gather selected full-head rows into a decode-only LM head."""
    if hasattr(model, "decode_lm_head"):
        raise ValueError("model already has a decode-only LM head")
    full_head = model.lm_head
    index = torch.tensor(
        token_ids,
        device=full_head.weight.device,
        dtype=torch.int64,
    )
    compact = nn.Linear(
        int(full_head.in_features),
        len(token_ids),
        bias=False,
        device=full_head.weight.device,
        dtype=full_head.weight.dtype,
    )
    with torch.no_grad():
        compact.weight.copy_(full_head.weight.index_select(0, index))
    compact.weight.requires_grad_(False)
    model.decode_lm_head = compact
    model.register_buffer(
        "decode_token_id_map",
        index,
        persistent=False,
    )
    return compact


def prepare_decode_weight_prefetch(
    model: "LocalPaddleOCRVLForConditionalGeneration",
    optimization: str | DecodeOptimizationConfig,
) -> None:
    """Install the proven stage-aware decode weight-prefetch schedule."""
    config = resolve_decode_optimization(optimization)
    if not config.stage_aware_weight_prefetch:
        return
    layers = model.model.layers
    decode_lm_head = getattr(model, "decode_lm_head", model.lm_head)
    def mlp_weights(mlp: nn.Module) -> tuple[torch.Tensor, ...]:
        # Prefetch the allocation actually consumed by decode, not the original
        # separate weights retained for prefill when gate/up are packed.
        if config.packed_mlp:
            return (mlp.decode_gate_up_proj.weight, mlp.down_proj.weight)
        return (mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight)

    def complete_layer_weights(layer: nn.Module) -> tuple[torch.Tensor, ...]:
        return (
            layer.self_attn.decode_qkv_proj.weight,
            layer.self_attn.o_proj.weight,
            *mlp_weights(layer.mlp),
        )

    for index, layer in enumerate(layers):
        layer.self_attn._decode_prefetch_current_mlp = mlp_weights(layer.mlp)
        layer.mlp._decode_prefetch_next_attention = (
            (
                layers[index + 1].self_attn.decode_qkv_proj.weight,
                layers[index + 1].self_attn.o_proj.weight,
            )
            if index + 1 < len(layers)
            else (decode_lm_head.weight,)
        )
        future_weights: list[torch.Tensor] = []
        for offset in range(1, config.complete_layer_prefetch_ahead + 1):
            future_index = index + offset
            if future_index < len(layers):
                future_weights.extend(complete_layer_weights(layers[future_index]))
        if index + 1 >= len(layers):
            future_weights.append(decode_lm_head.weight)
            if config.prefetch_next_iteration:
                # The autoregressive loop revisits layer zero next. This is
                # only a cache hint: no future request/token/KV is consumed.
                future_weights.extend(complete_layer_weights(layers[0]))
        layer._decode_prefetch_future_layers = tuple(future_weights)


def build_static_decode_bool_mask(
    cache_position: torch.Tensor,
    cache_length: int,
    kv_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    cache_position = cache_position.reshape(-1).to(dtype=torch.int64)
    if kv_positions is None:
        kv_positions = torch.arange(
            int(cache_length),
            device=cache_position.device,
            dtype=torch.int64,
        )
    return (
        kv_positions.unsqueeze(0) > cache_position.unsqueeze(1)
    ).view(cache_position.shape[0], 1, 1, int(cache_length))


def update_decode_kv_cache_(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    use_scatter_pa: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_scatter_pa:
        return decode_kv_scatter(
            key_cache,
            value_cache,
            cache_position,
            key_states,
            value_states,
        )
    positions = (
        cache_position.reshape(-1)
        .to(device=key_cache.device, dtype=torch.int64)
        .contiguous()
    )
    if key_cache.device.type == "npu":
        import torch_npu

        torch_npu.scatter_update_(
            key_cache, positions, key_states.contiguous(), 2
        )
        torch_npu.scatter_update_(
            value_cache, positions, value_states.contiguous(), 2
        )
        return key_cache, value_cache
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()
    batch_indices = torch.arange(
        int(key_cache.shape[0]),
        device=key_cache.device,
        dtype=torch.int64,
    )
    key_cache[batch_indices, :, positions, :] = key_states.squeeze(2)
    value_cache[batch_indices, :, positions, :] = value_states.squeeze(2)
    return key_cache, value_cache


def update_decode_packed_kv_cache_(
    packed_kv_cache: torch.Tensor,
    cache_position: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    """Write K and V through one stock ScatterUpdate invocation."""
    if packed_kv_cache.device.type != "npu":
        raise ValueError("packed KV scatter is an NPU-only decode lab path")
    import torch_npu

    positions = (
        cache_position.reshape(-1)
        .to(device=packed_kv_cache.device, dtype=torch.int64)
        .contiguous()
    )
    packed_positions = torch.cat((positions, positions), dim=0)
    packed_states = torch.cat((key_states, value_states), dim=0).contiguous()
    torch_npu.scatter_update_(
        packed_kv_cache,
        packed_positions,
        packed_states,
        2,
    )
    return packed_kv_cache


def _prepare_multimodal_rotary_factors(
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    mrope_section: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform the MRoPE section selection once per decode step."""
    section = [int(value) for value in mrope_section] * 2
    prepared = []
    for factors in position_embeddings:
        prepared.append(
            torch.cat(
                [
                    part[index % 3]
                    for index, part in enumerate(
                        factors.split(section, dim=-1)
                    )
                ],
                dim=-1,
            )
            .unsqueeze(1)
            .contiguous()
        )
    return prepared[0], prepared[1]


def _prepare_scalar_rotary_factors(
    rotary_emb: nn.Module,
    inputs_embeds: torch.Tensor,
    position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build decode RoPE factors directly from the one shared MRoPE axis."""
    freqs = (
        position.reshape(-1, 1).float()
        * rotary_emb.inv_freq.reshape(1, -1).float()
    )
    emb = torch.cat((freqs, freqs), dim=-1)
    return (
        emb.cos().to(dtype=inputs_embeds.dtype).view(
            inputs_embeds.shape[0], 1, 1, -1
        ),
        emb.sin().to(dtype=inputs_embeds.dtype).view(
            inputs_embeds.shape[0], 1, 1, -1
        ),
    )


def _lookup_scalar_rotary_factors(
    rotary_emb: nn.Module,
    position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select packed cosine/sine rows from the persistent decode RoPE LUT."""
    selected = torch.index_select(
        rotary_emb.decode_rope_factor_lut,
        1,
        position.reshape(-1).to(dtype=torch.int64),
    )
    cos, sin = selected.unbind(dim=0)
    cos = cos.unsqueeze(1).unsqueeze(1)
    sin = sin.unsqueeze(1).unsqueeze(1)
    return cos, sin


def prepare_decode_rope_factor_lut(
    model: "LocalPaddleOCRVLForConditionalGeneration",
    optimization: str | DecodeOptimizationConfig,
    *,
    cache_length: int,
    dtype: torch.dtype,
) -> None:
    """Create the final decode cos/sin table once, outside the graph."""
    config = resolve_decode_optimization(optimization)
    if config.rotary_factors != "lookup":
        return
    rotary_emb = model.model.rotary_emb
    positions = torch.arange(
        int(cache_length),
        device=rotary_emb.inv_freq.device,
        dtype=torch.float32,
    )
    freqs = positions.reshape(-1, 1) * rotary_emb.inv_freq.reshape(1, -1).float()
    emb = torch.cat((freqs, freqs), dim=-1)
    factor_lut = torch.stack((emb.cos(), emb.sin()), dim=0).to(dtype=dtype)
    rotary_emb.register_buffer(
        "decode_rope_factor_lut",
        factor_lut.contiguous(),
        persistent=False,
    )


def _project_decode_qkv(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    optimization: DecodeOptimizationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not optimization.packed_qkv:
        return attention.project_qkv(hidden_states)
    batch, query_length, _hidden = hidden_states.shape
    qkv = _linear_tokenwise(
        attention.decode_qkv_proj,
        hidden_states,
    )
    if optimization.ascendc_qkv_split:
        return decode_qkv_split(qkv)
    q_size = int(attention.num_heads * attention.head_dim)
    kv_size = int(attention.num_key_value_heads * attention.head_dim)
    query_states, key_states, value_states = qkv.split(
        (q_size, kv_size, kv_size),
        dim=-1,
    )
    query_states = query_states.view(
        batch,
        query_length,
        attention.num_heads,
        attention.head_dim,
    ).transpose(1, 2)
    key_states = key_states.view(
        batch,
        query_length,
        attention.num_key_value_heads,
        attention.head_dim,
    ).transpose(1, 2)
    value_states = value_states.view(
        batch,
        query_length,
        attention.num_key_value_heads,
        attention.head_dim,
    ).transpose(1, 2)
    return query_states, key_states, value_states


def _apply_decode_rotary(
    attention: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    prepared_factors: tuple[torch.Tensor, torch.Tensor] | None,
    optimization: DecodeOptimizationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prepared_factors is None:
        return attention.apply_rotary(
            query_states,
            key_states,
            position_embeddings,
        )
    if optimization.rotary == "identity":
        # Lab-only full-graph ablation. Keep dynamic Q/K tensors and remove
        # only the rotary operation so unprofiled timing measures its actual
        # marginal contribution to the compiled decode step.
        return query_states, key_states
    cos, sin = prepared_factors
    if optimization.rotary == "manual":
        half = query_states.shape[-1] // 2
        query_rotated = torch.cat(
            (-query_states[..., half:], query_states[..., :half]),
            dim=-1,
        )
        key_rotated = torch.cat(
            (-key_states[..., half:], key_states[..., :half]),
            dim=-1,
        )
        return (
            (query_states * cos) + (query_rotated * sin),
            (key_states * cos) + (key_rotated * sin),
        )

    import torch_npu

    if optimization.rotary == "npu_rotary_mul":
        return (
            torch_npu.npu_rotary_mul(
                query_states.contiguous(),
                cos,
                sin,
                rotary_mode="half",
            ),
            torch_npu.npu_rotary_mul(
                key_states.contiguous(),
                cos,
                sin,
                rotary_mode="half",
            ),
        )
    if optimization.rotary == "npu_apply":
        query_bsnd = query_states.transpose(1, 2).contiguous()
        key_bsnd = key_states.transpose(1, 2).contiguous()
        query_bsnd, key_bsnd = torch_npu.npu_apply_rotary_pos_emb(
            query_bsnd,
            key_bsnd,
            cos,
            sin,
            layout="BSND",
            rotary_mode="half",
        )
        return (
            query_bsnd.transpose(1, 2),
            key_bsnd.transpose(1, 2),
        )
    raise ValueError(
        f"unsupported decode rotary implementation: "
        f"{optimization.rotary!r}"
    )


def _decode_rms_norm(
    norm: nn.Module,
    hidden_states: torch.Tensor,
    optimization: DecodeOptimizationConfig,
) -> torch.Tensor:
    if optimization.rms_norm == "identity":
        return hidden_states
    if optimization.rms_norm == "manual":
        return norm(hidden_states)
    if optimization.rms_norm != "npu":
        raise ValueError(
            f"unsupported decode RMSNorm implementation: "
            f"{optimization.rms_norm!r}"
        )
    import torch_npu

    return torch_npu.npu_rms_norm(
        hidden_states,
        norm.weight,
        norm.variance_epsilon,
    )[0]


def _decode_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    normalized, _rstd, summed = torch_npu.npu_add_rms_norm(
        x,
        residual,
        norm.weight,
        norm.variance_epsilon,
    )
    return normalized, summed


@torch.library.custom_op(
    "paddleocr_vl::vector_add_rms_norm",
    mutates_args=(),
)
def _vector_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eager reference for the lab-only compiled VectorAddRmsNorm op."""
    import torch_npu

    return torch_npu.npu_add_rms_norm(x, residual, weight, epsilon)


@_vector_add_rms_norm.register_fake
def _vector_add_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del residual, weight, epsilon
    rstd_shape = (*x.shape[:-1], 1)
    return (
        torch.empty_like(x),
        torch.empty(rstd_shape, dtype=torch.float32, device=x.device),
        torch.empty_like(x),
    )


_VECTOR_ADD_RMS_NORM_CONVERTER_REGISTERED = False


def _register_vector_add_rms_norm_converter() -> None:
    """Lower the lab custom op to its installed CANN GE operator."""
    global _VECTOR_ADD_RMS_NORM_CONVERTER_REGISTERED
    if _VECTOR_ADD_RMS_NORM_CONVERTER_REGISTERED:
        return

    import importlib

    torchair, _CompilerConfig = import_torchair()
    converter_module = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.fx2ge_converter"
    )
    ge_module = importlib.import_module(f"{torchair.__name__}.ge")
    register_converter = converter_module.register_fx_node_ge_converter
    ge_custom_op = ge_module.custom_op
    op = torch.ops.paddleocr_vl.vector_add_rms_norm.default

    @register_converter(op)
    def _convert_vector_add_rms_norm(
        x: Any,
        residual: Any,
        weight: Any,
        epsilon: float,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        return ge_custom_op(
            "VectorAddRmsNorm",
            x,
            residual,
            weight,
            float(epsilon),
        )

    _VECTOR_ADD_RMS_NORM_CONVERTER_REGISTERED = True


def _decode_add_with_optional_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm: nn.Module,
    optimization: DecodeOptimizationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if optimization.rms_norm == "identity":
        summed = x + residual
        return summed, summed
    if optimization.vector_add_rms_norm:
        normalized, _rstd, summed = _vector_add_rms_norm(
            x,
            residual,
            norm.weight,
            norm.variance_epsilon,
        )
        return normalized, summed
    return _decode_add_rms_norm(x, residual, norm)


def _decode_mlp(
    mlp: nn.Module,
    hidden_states: torch.Tensor,
    optimization: DecodeOptimizationConfig,
) -> torch.Tensor:
    if optimization.ascendc_linear and not optimization.packed_mlp:
        gate = _linear_tokenwise(mlp.gate_proj, hidden_states)
        up = _linear_tokenwise(mlp.up_proj, hidden_states)
        activated = (
            decode_swiglu(gate, up)
            if optimization.ascendc_swiglu
            else torch.nn.functional.silu(gate) * up
        )
        output = _linear_tokenwise(mlp.down_proj, activated)
    elif not optimization.packed_mlp:
        output = mlp(hidden_states)
    else:
        gate_up = _linear_tokenwise(
            mlp.decode_gate_up_proj,
            hidden_states,
        )
        if optimization.npu_swiglu:
            import torch_npu

            activated = torch_npu.npu_swiglu(gate_up, dim=-1)
        else:
            gate, up = gate_up.chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
        output = _linear_tokenwise(mlp.down_proj, activated)
    if (
        optimization.stage_aware_weight_prefetch
        and optimization.weight_prefetch_timing != "complete_layer_ahead"
    ):
        import torch_npu

        for weight in mlp._decode_prefetch_next_attention:
            torch_npu.npu_prefetch(
                weight,
                output,
                int(weight.numel() * weight.element_size()),
            )
    return output


def _decode_attention(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    prepared_factors: tuple[torch.Tensor, torch.Tensor] | None,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    attention_mask: torch.Tensor | None,
    pse_shift: torch.Tensor | None,
    actual_seq_lengths: list[int] | None,
    optimization: DecodeOptimizationConfig,
    packed_kv_cache: torch.Tensor | None = None,
    packed_factor_lut: torch.Tensor | None = None,
    packed_rope_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        optimization.stage_aware_weight_prefetch
        and optimization.weight_prefetch_timing == "before_attention"
    ):
        import torch_npu

        for weight in attention._decode_prefetch_current_mlp:
            torch_npu.npu_prefetch(
                weight,
                hidden_states,
                int(weight.numel() * weight.element_size()),
            )
    if optimization.ascendc_packed_qkv_rope_gqa_mixed24:
        if packed_factor_lut is None or packed_rope_delta is None:
            raise ValueError("packed QKV/RoPE GQA requires its factor LUT and delta")
        if attention_mask is None:
            raise ValueError("packed QKV/RoPE GQA requires persistent mask scratch")
        if pse_shift is not None or actual_seq_lengths is not None:
            raise ValueError("packed QKV/RoPE GQA requires masked static attention")
        packed_qkv = _linear_tokenwise(
            attention.decode_qkv_proj,
            hidden_states,
        )
        attention_output = decode_packed_qkv_rope_gqa_mixed24(
            packed_qkv,
            key_cache,
            value_cache,
            attention_mask,
            cache_position,
            packed_factor_lut,
            packed_rope_delta,
            num_heads=int(attention.num_heads),
            num_key_value_heads=int(attention.num_key_value_heads),
            scale_value=float(attention.scaling),
            inner_precise=1,
            vector_core_count=16,
        )
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .reshape(1, 1, attention.num_heads * attention.head_dim)
        )
        return _linear_tokenwise(attention.o_proj, attention_output)
    query_states, key_states, value_states = _project_decode_qkv(
        attention,
        hidden_states,
        optimization,
    )
    query_states, key_states = _apply_decode_rotary(
        attention,
        query_states,
        key_states,
        position_embeddings,
        prepared_factors,
        optimization,
    )
    if (
        optimization.ascendc_decode_gqa
        or optimization.ascendc_decode_gqa_mixed
        or optimization.ascendc_decode_gqa_mixed24
    ):
        if attention_mask is None:
            raise ValueError("fused decode GQA requires persistent mask scratch")
        if pse_shift is not None or actual_seq_lengths is not None:
            raise ValueError("fused decode GQA requires masked static attention")
        if optimization.ascendc_decode_gqa_mixed24:
            fused_attention = decode_gqa_incre_flash_attention_mixed24
        elif optimization.ascendc_decode_gqa_mixed:
            fused_attention = decode_gqa_incre_flash_attention_mixed
        else:
            fused_attention = decode_gqa_incre_flash_attention_aiv
        attention_output = fused_attention(
            query_states,
            key_cache,
            value_cache,
            attention_mask,
            cache_position,
            key_states,
            value_states,
            num_heads=int(attention.num_heads),
            num_key_value_heads=int(attention.num_key_value_heads),
            scale_value=float(attention.scaling),
            inner_precise=1,
            vector_core_count=16,
        )
        batch_size = query_states.shape[0]
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .reshape(
                batch_size,
                1,
                attention.num_heads * attention.head_dim,
            )
        )
        return _linear_tokenwise(attention.o_proj, attention_output)
    if optimization.attention == "mha_cache":
        if query_states.device.type != "npu":
            raise ValueError("mha_cache is an NPU-only decode lab path")
        groups = int(attention.num_key_value_groups)
        batch_size, kv_heads, token_count, head_dim = key_states.shape
        expected_heads = int(attention.num_heads)
        if int(key_cache.shape[1]) != expected_heads:
            raise ValueError(
                "mha_cache requires a fully expanded decode arena: "
                f"expected {expected_heads} heads, got {int(key_cache.shape[1])}"
            )
        key_states = (
            key_states[:, :, None, :, :]
            .expand(batch_size, kv_heads, groups, token_count, head_dim)
            .reshape(batch_size, expected_heads, token_count, head_dim)
            .contiguous()
        )
        value_states = (
            value_states[:, :, None, :, :]
            .expand(batch_size, kv_heads, groups, token_count, head_dim)
            .reshape(batch_size, expected_heads, token_count, head_dim)
            .contiguous()
        )
    if optimization.packed_kv_scatter:
        if packed_kv_cache is None:
            raise ValueError("packed KV scatter requires its packed cache tensor")
        batch_size = int(query_states.shape[0])
        packed_kv_cache = update_decode_packed_kv_cache_(
            packed_kv_cache,
            cache_position,
            key_states,
            value_states,
        )
        key_cache = packed_kv_cache[:batch_size]
        value_cache = packed_kv_cache[batch_size:]
    elif optimization.ascendc_kv_scatter_query:
        query_states, attention_mask = decode_kv_scatter_query(
            query_states,
            key_cache,
            value_cache,
            cache_position,
            key_states,
            value_states,
            variant=(
                "mixed24"
                if optimization.ascendc_decode_gqa_attention_mixed24
                else "v4"
            ),
        )
    else:
        key_cache, value_cache = update_decode_kv_cache_(
            key_cache,
            value_cache,
            cache_position,
            key_states,
            value_states,
            use_scatter_pa=optimization.ascendc_kv_scatter,
        )
    if optimization.post_scatter_kv_prefetch:
        if key_cache.device.type != "npu":
            raise ValueError(
                "post-scatter K/V prefetch is an NPU-only decode lab path"
            )
        import torch_npu

        torch_npu.npu_prefetch(
            key_cache,
            key_states,
            int(key_cache.numel() * key_cache.element_size()),
        )
        torch_npu.npu_prefetch(
            value_cache,
            value_states,
            int(value_cache.numel() * value_cache.element_size()),
        )
    if query_states.device.type != "npu":
        additive_mask = attention_mask
        if additive_mask is not None and additive_mask.dtype == torch.bool:
            additive_mask = torch.zeros_like(
                additive_mask, dtype=query_states.dtype
            ).masked_fill(
                additive_mask,
                torch.finfo(query_states.dtype).min,
            )
        return attention.attend(
            query_states,
            key_cache,
            value_cache,
            additive_mask,
        )

    if optimization.attention == "no_increfa":
        # Lab-only full-graph ablation.  Keep QKV projection, RoPE, cache
        # writes, and output projection, but substitute dynamic query data for
        # the IncreFA result.  A dynamic substitute prevents TorchAir from
        # constant-folding the downstream output projection.
        batch = query_states.shape[0]
        attention_output = (
            query_states.transpose(1, 2)
            .contiguous()
            .reshape(batch, 1, attention.num_heads * attention.head_dim)
        )
        return _linear_tokenwise(attention.o_proj, attention_output)

    if optimization.attention in ("manual", "manual_unscaled"):
        # Lab-only decomposition of one-token GQA decode.  It deliberately
        # exposes QK, score scaling, masking, softmax, and PV as separate graph
        # operations so their kernels can be profiled against IncreFA.  The
        # unscaled lane omits only the numerical 1/sqrt(head_dim) multiply; it
        # is intentionally incorrect and exists only to measure that Vector
        # operation's cost.
        groups = int(attention.num_key_value_groups)
        batch_size, kv_heads, kv_length, head_dim = key_cache.shape
        query_heads = int(attention.num_heads)
        key_manual = (
            key_cache[:, :, None, :, :]
            .expand(batch_size, kv_heads, groups, kv_length, head_dim)
            .reshape(batch_size * query_heads, kv_length, head_dim)
        )
        value_manual = (
            value_cache[:, :, None, :, :]
            .expand(batch_size, kv_heads, groups, kv_length, head_dim)
            .reshape(batch_size * query_heads, kv_length, head_dim)
        )
        query_manual = query_states.reshape(
            batch_size * query_heads, 1, head_dim
        )
        scores = torch.bmm(
            query_manual,
            key_manual.transpose(1, 2),
        ).view(batch_size, query_heads, 1, kv_length)
        if optimization.attention == "manual":
            scores = scores * float(attention.scaling)
        if pse_shift is not None:
            scores = scores + pse_shift
        if attention_mask is not None:
            scores = scores.masked_fill(
                attention_mask,
                torch.finfo(scores.dtype).min,
            )
        probabilities = torch.softmax(scores.float(), dim=-1).to(
            dtype=query_states.dtype
        )
        attention_output = torch.bmm(
            probabilities.reshape(
                batch_size * query_heads, 1, kv_length
            ),
            value_manual,
        ).view(batch_size, query_heads, 1, head_dim)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, 1, query_heads * head_dim)
        )
        return _linear_tokenwise(attention.o_proj, attention_output)

    import torch_npu

    batch = query_states.shape[0]
    key_for_attention = key_cache
    value_for_attention = value_cache
    num_key_value_heads = int(attention.num_key_value_heads)
    if optimization.attention == "mha_repeat":
        groups = int(attention.num_key_value_groups)
        batch_size, kv_heads, kv_length, head_dim = key_cache.shape
        key_for_attention = (
            key_cache[:, :, None, :, :]
            .expand(batch_size, kv_heads, groups, kv_length, head_dim)
            .reshape(batch_size, kv_heads * groups, kv_length, head_dim)
            .contiguous()
        )
        value_for_attention = (
            value_cache[:, :, None, :, :]
            .expand(batch_size, kv_heads, groups, kv_length, head_dim)
            .reshape(batch_size, kv_heads * groups, kv_length, head_dim)
            .contiguous()
        )
        num_key_value_heads = 0
    elif optimization.attention == "mha_cache":
        if int(key_cache.shape[1]) != int(attention.num_heads):
            raise ValueError("mha_cache received a non-expanded decode arena")
        num_key_value_heads = 0
    elif optimization.attention not in (
        "gqa",
        "gqa_aiv",
        "gqa_pseudo_b2",
    ):
        raise ValueError(
            f"unsupported decode attention implementation: "
            f"{optimization.attention!r}"
        )

    if optimization.attention == "gqa_aiv":
        if pse_shift is not None or actual_seq_lengths is not None:
            raise ValueError(
                "gqa_aiv requires masked IncreFA with no PSE or actual lengths"
            )
        if attention_mask is None:
            raise ValueError("gqa_aiv requires the static bool attention mask")
        if optimization.ascendc_decode_gqa_attention_mixed24:
            attention_fn = decode_gqa_attention_mixed24
        elif optimization.ascendc_decode_gqa_attention:
            attention_fn = decode_gqa_attention_aiv
        else:
            attention_fn = gqa_incre_flash_attention_aiv
        attention_output = attention_fn(
            query_states.contiguous(),
            key_for_attention.contiguous(),
            value_for_attention.contiguous(),
            attention_mask.contiguous(),
            num_heads=int(attention.num_heads),
            num_key_value_heads=num_key_value_heads,
            scale_value=float(attention.scaling),
            inner_precise=1,
            vector_core_count=optimization.gqa_aiv_vector_core_count,
        )
    else:
        pseudo_b2 = optimization.attention == "gqa_pseudo_b2"
        if pseudo_b2:
            if batch != 1:
                raise ValueError("pseudo-B2 GQA requires physical batch size one")
            if int(attention.num_heads) != 16:
                raise ValueError("pseudo-B2 GQA requires 16 query heads")
            if num_key_value_heads != 2:
                raise ValueError("pseudo-B2 GQA requires two KV heads")
            if pse_shift is not None or actual_seq_lengths is not None:
                raise ValueError("pseudo-B2 GQA requires mask-only IncreFA")
            if attention_mask is None:
                raise ValueError("pseudo-B2 GQA requires a boolean mask")
            query_for_attention = query_states.contiguous().view(
                2, 8, 1, int(attention.head_dim)
            )
            key_for_attention = key_for_attention.view(
                2, 1, int(key_for_attention.shape[2]), int(attention.head_dim)
            )
            value_for_attention = value_for_attention.view(
                2,
                1,
                int(value_for_attention.shape[2]),
                int(attention.head_dim),
            )
            mask_for_attention = attention_mask
            call_num_heads = 8
            call_num_key_value_heads = 1
        else:
            query_for_attention = query_states
            mask_for_attention = attention_mask
            call_num_heads = int(attention.num_heads)
            call_num_key_value_heads = num_key_value_heads
        increfa_kwargs: dict[str, object] = {}
        if optimization.increfa_inner_precise is not None:
            increfa_kwargs["inner_precise"] = int(
                optimization.increfa_inner_precise
            )
        attention_output = torch_npu.npu_incre_flash_attention(
            query_for_attention.contiguous(),
            key_for_attention.contiguous(),
            value_for_attention.contiguous(),
            pse_shift=pse_shift,
            atten_mask=(
                None
                if mask_for_attention is None
                else mask_for_attention.contiguous()
            ),
            actual_seq_lengths=actual_seq_lengths,
            num_heads=call_num_heads,
            num_key_value_heads=call_num_key_value_heads,
            input_layout="BNSD",
            scale_value=float(attention.scaling),
            **increfa_kwargs,
        )
        if pseudo_b2:
            attention_output = attention_output.view(
                1, 16, 1, int(attention.head_dim)
            )
    attention_output = (
        attention_output.transpose(1, 2)
        .contiguous()
        .reshape(batch, 1, attention.num_heads * attention.head_dim)
    )
    if (
        optimization.stage_aware_weight_prefetch
        and optimization.weight_prefetch_timing == "after_attention"
    ):
        for weight in attention._decode_prefetch_current_mlp:
            torch_npu.npu_prefetch(
                weight,
                attention_output,
                int(weight.numel() * weight.element_size()),
            )
    return _linear_tokenwise(attention.o_proj, attention_output)


def run_text_decode_transformer(
    text_model: nn.Module,
    *,
    inputs_embeds: torch.Tensor,
    cache_position: torch.Tensor,
    rope_deltas: torch.Tensor,
    key_caches: tuple[torch.Tensor, ...],
    value_caches: tuple[torch.Tensor, ...],
    packed_kv_caches: tuple[torch.Tensor, ...] | None = None,
    cache_length: int,
    attention_mask: torch.Tensor | None = None,
    static_kv_positions: torch.Tensor | None = None,
    optimization: str | DecodeOptimizationConfig = "baseline",
) -> torch.Tensor:
    """Execute the complete one-token transformer decode stage."""
    optimization = resolve_decode_optimization(optimization)
    batch_size, seq_length, _hidden = inputs_embeds.shape
    if seq_length != 1:
        raise ValueError(
            f"static decode expects exactly one token, got "
            f"seq_length={seq_length}"
        )
    cache_position = cache_position.reshape(-1).to(
        device=inputs_embeds.device, dtype=torch.int64
    )
    if cache_position.numel() == 1:
        cache_position = cache_position.expand(batch_size)
    if cache_position.numel() != batch_size:
        raise ValueError(
            "cache_position must be scalar or batch-shaped, got "
            f"{tuple(cache_position.shape)}"
        )
    if attention_mask is None:
        attention_mask = build_static_decode_bool_mask(
            cache_position,
            cache_length,
            kv_positions=static_kv_positions,
        )
    if optimization.attention == "gqa_pseudo_b2":
        if batch_size != 1:
            raise ValueError("pseudo-B2 GQA requires physical batch size one")
        attention_mask = attention_mask.expand(2, -1, -1, -1).contiguous()
    pse_shift: torch.Tensor | None = None
    actual_seq_lengths: list[int] | None = None
    if optimization.increfa_length_mode == "pse_sentinel":
        # The 310P masked-GQA kernel can deadlock when the valid prefix is an
        # exact 1280-token internal tile. Keep one always-present PSE graph:
        # expose one otherwise-masked cache position only at those boundaries,
        # then suppress it additively. The PSE is zero at all other positions.
        effective_lengths = cache_position.view(batch_size, 1, 1, 1) + 1
        physical_positions = torch.arange(
            int(cache_length),
            device=inputs_embeds.device,
            dtype=torch.int64,
        ).view(1, 1, 1, int(cache_length))
        boundary = (
            (effective_lengths.remainder(1280) == 0)
            & (effective_lengths < int(cache_length))
        )
        sentinel = boundary & (physical_positions == effective_lengths)
        attention_mask = attention_mask & ~sentinel
        pse_shift = torch.zeros(
            (
                batch_size,
                int(text_model.config.num_attention_heads),
                1,
                int(cache_length),
            ),
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        ).masked_fill(
            sentinel.expand(
                batch_size,
                int(text_model.config.num_attention_heads),
                1,
                int(cache_length),
            ),
            torch.finfo(inputs_embeds.dtype).min,
        )
    elif optimization.increfa_length_mode == "static_actual":
        # Deliberately constant for the static BxKV graph. The boolean mask
        # still carries each row's logical prefix length.
        actual_seq_lengths = [int(cache_length)] * int(batch_size)
    elif optimization.increfa_length_mode != "mask":
        raise ValueError(
            "unsupported IncreFA length mode: "
            f"{optimization.increfa_length_mode!r}"
        )
    cache_position_2d = cache_position.view(batch_size, 1)
    rope_deltas_i64 = rope_deltas.to(
        device=inputs_embeds.device, dtype=torch.int64
    )
    packed_factor_lut: torch.Tensor | None = None
    packed_rope_delta: torch.Tensor | None = None
    if optimization.ascendc_packed_qkv_rope_gqa_mixed24:
        packed_factor_lut = text_model.rotary_emb.decode_rope_factor_lut
        packed_rope_delta = rope_deltas_i64
        prepared_factors = None
        position_embeddings = (packed_factor_lut, packed_factor_lut)
    elif (
        optimization.rotary_factors == "lookup"
        and optimization.ascendc_rope_lookup
    ):
        prepared_factors = decode_rope_lookup(
            text_model.rotary_emb.decode_rope_factor_lut,
            cache_position_2d,
            rope_deltas_i64,
        )
        position_embeddings = prepared_factors
    else:
        decode_position = (
            decode_position_add(cache_position_2d, rope_deltas_i64)
            if optimization.ascendc_position_add
            else cache_position_2d + rope_deltas_i64
        )
        if optimization.rotary_factors == "mrope":
            position_ids = decode_position.unsqueeze(0).expand(3, -1, -1)
            position_embeddings = text_model.rotary_emb(
                inputs_embeds, position_ids
            )
            prepared_factors = (
                _prepare_multimodal_rotary_factors(
                    position_embeddings,
                    text_model.layers[0].self_attn.mrope_section,
                )
                if optimization.hoist_mrope
                else None
            )
        elif optimization.rotary_factors == "scalar":
            prepared_factors = _prepare_scalar_rotary_factors(
                text_model.rotary_emb,
                inputs_embeds,
                decode_position,
            )
            position_embeddings = prepared_factors
        elif optimization.rotary_factors == "lookup":
            prepared_factors = _lookup_scalar_rotary_factors(
                text_model.rotary_emb,
                decode_position,
            )
            position_embeddings = prepared_factors
        else:
            raise ValueError(
                "unsupported decode rotary-factor mode: "
                f"{optimization.rotary_factors!r}"
            )
    hidden_states = inputs_embeds
    if optimization.name == "baseline":
        for layer_idx, layer in enumerate(text_model.layers):
            residual = hidden_states
            attention_input = layer.input_layernorm(hidden_states)
            attention_output = _decode_attention(
                layer.self_attn,
                attention_input,
                position_embeddings,
                None,
                key_caches[layer_idx],
                value_caches[layer_idx],
                cache_position,
                attention_mask,
                pse_shift,
                actual_seq_lengths,
                optimization,
                packed_kv_cache=(
                    packed_kv_caches[layer_idx]
                    if packed_kv_caches is not None
                    else None
                ),
                packed_factor_lut=packed_factor_lut,
                packed_rope_delta=packed_rope_delta,
            )
            hidden_states = layer.apply_blocks(residual, attention_output)
        return text_model.norm(hidden_states)

    if optimization.add_rms_norm:
        residual: torch.Tensor | None = None
        for layer_idx, layer in enumerate(text_model.layers):
            if optimization.complete_layer_prefetch_ahead:
                import torch_npu

                for weight in layer._decode_prefetch_future_layers:
                    torch_npu.npu_prefetch(
                        weight,
                        hidden_states,
                        int(weight.numel() * weight.element_size()),
                    )
            if residual is None:
                if optimization.zero_residual_first_rms_norm:
                    attention_input, residual = _decode_add_rms_norm(
                        hidden_states,
                        torch.zeros_like(hidden_states),
                        layer.input_layernorm,
                    )
                else:
                    attention_input = _decode_rms_norm(
                        layer.input_layernorm,
                        hidden_states,
                        optimization,
                    )
                    residual = hidden_states
            else:
                attention_input, residual = _decode_add_with_optional_rms_norm(
                    hidden_states,
                    residual,
                    layer.input_layernorm,
                    optimization,
                )
            attention_output = _decode_attention(
                layer.self_attn,
                attention_input,
                position_embeddings,
                prepared_factors,
                key_caches[layer_idx],
                value_caches[layer_idx],
                cache_position,
                attention_mask,
                pse_shift,
                actual_seq_lengths,
                optimization,
                packed_kv_cache=(
                    packed_kv_caches[layer_idx]
                    if packed_kv_caches is not None
                    else None
                ),
                packed_factor_lut=packed_factor_lut,
                packed_rope_delta=packed_rope_delta,
            )
            mlp_input, residual = _decode_add_with_optional_rms_norm(
                attention_output,
                residual,
                layer.post_attention_layernorm,
                optimization,
            )
            hidden_states = _decode_mlp(
                layer.mlp,
                mlp_input,
                optimization,
            )
        hidden_states, _residual = _decode_add_with_optional_rms_norm(
            hidden_states,
            residual,
            text_model.norm,
            optimization,
        )
        return hidden_states

    for layer_idx, layer in enumerate(text_model.layers):
        residual = hidden_states
        attention_input = _decode_rms_norm(
            layer.input_layernorm,
            hidden_states,
            optimization,
        )
        attention_output = _decode_attention(
            layer.self_attn,
            attention_input,
            position_embeddings,
            prepared_factors,
            key_caches[layer_idx],
            value_caches[layer_idx],
            cache_position,
            attention_mask,
            pse_shift,
            actual_seq_lengths,
            optimization,
            packed_kv_cache=(
                packed_kv_caches[layer_idx]
                if packed_kv_caches is not None
                else None
            ),
            packed_factor_lut=packed_factor_lut,
            packed_rope_delta=packed_rope_delta,
        )
        if (
            optimization.rms_norm == "manual"
            and not optimization.packed_mlp
        ):
            hidden_states = layer.apply_blocks(residual, attention_output)
            continue
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = _decode_rms_norm(
            layer.post_attention_layernorm,
            hidden_states,
            optimization,
        )
        hidden_states = _decode_mlp(
            layer.mlp,
            hidden_states,
            optimization,
        )
        hidden_states = residual + hidden_states
    return _decode_rms_norm(text_model.norm, hidden_states, optimization)


def cast_decode_linear_weights_to_nz(
    model: "LocalPaddleOCRVLForConditionalGeneration",
) -> dict[str, object]:
    """Prepare all text-decode Linear weights in NPU FRACTAL_NZ format."""
    modules = [
        (f"model.{name}", module)
        for name, module in model.model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    modules.append(("lm_head", model.lm_head))
    if hasattr(model, "decode_lm_head"):
        modules.append(("decode_lm_head", model.decode_lm_head))
    non_npu_modules = [
        (name, str(module.weight.device))
        for name, module in modules
        if module.weight.device.type != "npu"
    ]
    if non_npu_modules:
        return {
            "requested_mode": DECODE_LINEAR_WEIGHT_FORMAT,
            "mode": DECODE_LINEAR_WEIGHT_FALLBACK,
            "effective_mode": DECODE_LINEAR_WEIGHT_FALLBACK,
            "target_format": "FRACTAL_NZ",
            "target_format_code": FRACTAL_NZ,
            "target_count": len(modules),
            "cast_count": 0,
            "converted_count": 0,
            "already_nz_count": 0,
            "skipped": True,
            "skip_reason": "requires_npu_resident_weights",
            "fallback_reason": "requires_npu_resident_weights",
            "non_npu_modules_sample": non_npu_modules[:16],
            "all_after_are_nz": False,
        }

    import torch_npu

    before_formats: dict[str, int] = {}
    after_formats: dict[str, int] = {}
    converted: list[str] = []
    already_nz: list[str] = []
    failures: list[dict[str, object]] = []
    cast_count = 0
    for name, module in modules:
        before = int(torch_npu.get_npu_format(module.weight))
        before_formats[name] = before
        if before == FRACTAL_NZ:
            already_nz.append(name)
            after_formats[name] = before
            continue
        cast_count += 1
        try:
            module.weight.data = torch_npu.npu_format_cast(
                module.weight.data, FRACTAL_NZ
            )
        except Exception as exc:
            failures.append(
                {
                    "module": name,
                    "before_format": before,
                    "error": repr(exc),
                }
            )
            break
        after = int(torch_npu.get_npu_format(module.weight))
        after_formats[name] = after
        if before != FRACTAL_NZ and after == FRACTAL_NZ:
            converted.append(name)
        else:
            failures.append(
                {
                    "module": name,
                    "before_format": before,
                    "after_format": after,
                    "error": "npu_format_cast_did_not_produce_fractal_nz",
                }
            )
            break
    all_after_are_nz = len(after_formats) == len(modules) and all(
        value == FRACTAL_NZ for value in after_formats.values()
    )
    if all_after_are_nz:
        effective_mode = DECODE_LINEAR_WEIGHT_FORMAT
    elif converted:
        effective_mode = "decode_mixed_format"
    else:
        effective_mode = DECODE_LINEAR_WEIGHT_FALLBACK
    return {
        "requested_mode": DECODE_LINEAR_WEIGHT_FORMAT,
        "mode": effective_mode,
        "effective_mode": effective_mode,
        "target_format": "FRACTAL_NZ",
        "target_format_code": FRACTAL_NZ,
        "target_count": len(modules),
        "cast_count": cast_count,
        "converted_count": len(converted),
        "already_nz_count": len(already_nz),
        "converted_modules_sample": converted[:16],
        "before_formats_sample": dict(list(before_formats.items())[:16]),
        "after_formats_sample": dict(list(after_formats.items())[:16]),
        "all_after_are_nz": all_after_are_nz,
        "fallback_reason": failures[0]["error"] if failures else None,
        "failures_sample": failures[:16],
    }

class TextDecodeStage(torch.nn.Module):
    """One fixed-shape autoregressive text step.

    The same module is called directly for eager execution or wrapped by the
    selected compiler. Cache tensors stay flat at the boundary so the compiled
    graph can mutate the persistent decode arena in place.
    """

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        optimization: str | DecodeOptimizationConfig = "baseline",
        *,
        cache_length: int | None = None,
    ):
        super().__init__()
        self.model = model
        self.num_layers = int(model.config.text_config.num_hidden_layers)
        self.optimization = resolve_decode_optimization(optimization)
        self._super_kernel_scope = None
        if self.optimization.super_kernel_scope:
            if cache_length is None:
                raise ValueError(
                    "the Paddle decoder SuperKernel requires cache_length"
                )
            parameter = next(model.parameters())
            if (
                self.optimization.ascendc_decode_gqa
                or self.optimization.ascendc_decode_gqa_mixed
                or self.optimization.ascendc_decode_gqa_mixed24
                or self.optimization.ascendc_packed_qkv_rope_gqa_mixed24
            ) and int(cache_length) != 1024:
                raise ValueError(
                    "fused decode GQA is specialized for cache_length=1024"
                )
            self.register_buffer(
                "_super_kernel_kv_positions",
                torch.arange(
                    int(cache_length),
                    device=parameter.device,
                    dtype=torch.int64,
                ),
                persistent=False,
            )
            if (
                self.optimization.ascendc_decode_gqa
                or self.optimization.ascendc_decode_gqa_mixed
                or self.optimization.ascendc_decode_gqa_mixed24
                or self.optimization.ascendc_packed_qkv_rope_gqa_mixed24
            ):
                self.register_buffer(
                    "_super_kernel_attention_mask_scratch",
                    torch.zeros(
                        (1, 1, 1, int(cache_length)),
                        device=parameter.device,
                        dtype=torch.bool,
                    ),
                    persistent=False,
                )
            torchair, _CompilerConfig = import_torchair()
            scope_module = importlib.import_module(
                f"{torchair.__name__}.scope"
            )
            self._super_kernel_scope = scope_module.super_kernel

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        rope_deltas: torch.Tensor,
        *flat_cache_tensors: torch.Tensor,
    ) -> torch.Tensor:
        if self.optimization.packed_kv_scatter:
            if len(flat_cache_tensors) != self.num_layers:
                raise ValueError("packed KV decode requires one cache per layer")
            batch_size = int(input_ids.shape[0])
            packed_kv_caches = flat_cache_tensors
            key_caches = tuple(cache[:batch_size] for cache in packed_kv_caches)
            value_caches = tuple(cache[batch_size:] for cache in packed_kv_caches)
        else:
            packed_kv_caches = None
            key_caches = flat_cache_tensors[: self.num_layers]
            value_caches = flat_cache_tensors[self.num_layers :]
        inputs_embeds = (
            decode_token_embedding(
                self.model.model.embed_tokens.weight,
                input_ids,
            )
            if self.optimization.ascendc_token_embedding
            else self.model.model.embed_tokens(input_ids)
        )
        hidden_states = run_text_decode_transformer(
            self.model.model,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            rope_deltas=rope_deltas,
            key_caches=key_caches,
            value_caches=value_caches,
            packed_kv_caches=packed_kv_caches,
            cache_length=int(key_caches[0].shape[2]),
            attention_mask=(
                self._super_kernel_attention_mask_scratch
                if (
                    self.optimization.ascendc_decode_gqa
                    or self.optimization.ascendc_decode_gqa_mixed
                    or self.optimization.ascendc_decode_gqa_mixed24
                    or self.optimization.ascendc_packed_qkv_rope_gqa_mixed24
                )
                else None
            ),
            static_kv_positions=(
                self._super_kernel_kv_positions
                if self.optimization.super_kernel_scope
                else None
            ),
            optimization=self.optimization,
        )
        output_head = getattr(self.model, "decode_lm_head", self.model.lm_head)
        logits = _linear_tokenwise(output_head, hidden_states[:, -1:, :])
        if hasattr(self.model, "decode_token_id_map"):
            compact_ids = torch.argmax(logits[:, -1, :].float(), dim=-1)
            return self.model.decode_token_id_map.index_select(
                0,
                compact_ids,
            ).view(-1, 1)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        rope_deltas: torch.Tensor,
        *flat_cache_tensors: torch.Tensor,
    ) -> torch.Tensor:
        if not self.optimization.super_kernel_scope:
            return self._forward_impl(
                input_ids,
                cache_position,
                rope_deltas,
                *flat_cache_tensors,
            )

        if self._super_kernel_scope is None:
            raise RuntimeError("TorchAir SuperKernel scope was not initialized")
        # The packed AscendC ABI requires a rank-one scalar buffer. Normalize
        # it outside the strict SuperKernel scope so GE does not have to fuse
        # the view-only AsStrided node with device subkernels.
        if self.optimization.ascendc_packed_qkv_rope_gqa_mixed24:
            rope_deltas = rope_deltas.reshape(-1)
        with self._super_kernel_scope(
            f"{self.optimization.name}_scope",
            self.optimization.super_kernel_options,
        ):
            return self._forward_impl(
                input_ids,
                cache_position,
                rope_deltas,
                *flat_cache_tensors,
            )


def decode_attention_label(
    device: torch.device,
    optimization: DecodeOptimizationConfig | None = None,
) -> str:
    if device.type != "npu":
        return "manual"
    if optimization is not None and optimization.attention == "gqa_aiv":
        if optimization.ascendc_packed_qkv_rope_gqa_mixed24:
            return "paddle_decode_packed_qkv_rope_gqa_mixed24"
        if optimization.ascendc_decode_gqa_mixed24:
            return "paddle_decode_gqa_increfa_mixed24"
        if optimization.ascendc_decode_gqa_mixed:
            return "paddle_decode_gqa_increfa_mixed"
        if optimization.ascendc_decode_gqa:
            return "paddle_decode_gqa_increfa_aiv"
        if optimization.ascendc_decode_gqa_attention:
            return "paddle_decode_gqa_attention_aiv"
        if optimization.ascendc_decode_gqa_attention_mixed24:
            return "paddle_decode_gqa_attention_mixed24"
        return "paddle_gqa_increfa_aiv"
    return DECODE_ATTENTION


def decode_cache_update_label(
    device: torch.device,
    optimization: DecodeOptimizationConfig | None = None,
) -> str:
    if (
        device.type == "npu"
        and optimization is not None
        and optimization.ascendc_kv_scatter_query
    ):
        return "paddle_decode_kv_scatter_query_v4"
    if (
        device.type == "npu"
        and optimization is not None
        and (
            optimization.ascendc_decode_gqa
            or optimization.ascendc_decode_gqa_mixed
            or optimization.ascendc_decode_gqa_mixed24
            or optimization.ascendc_packed_qkv_rope_gqa_mixed24
        )
    ):
        if optimization.ascendc_packed_qkv_rope_gqa_mixed24:
            return "paddle_decode_packed_qkv_rope_gqa_mixed24"
        if optimization.ascendc_decode_gqa_mixed24:
            return "paddle_decode_gqa_increfa_mixed24"
        if optimization.ascendc_decode_gqa_mixed:
            return "paddle_decode_gqa_increfa_mixed"
        return "paddle_decode_gqa_increfa_aiv"
    return DECODE_CACHE_UPDATE if device.type == "npu" else "per_row_copy"


def decode_source_hash() -> str:
    here = Path(__file__).resolve().parent
    digest = hashlib.sha1()
    # Decode owns its graph, while the shared text layer methods it calls are
    # defined by the prefill stage.
    for name in (
        "text_prefill_and_decode.py",
        "text_prefill_and_decode.py",
        "_support/model/gqa_increfa_aiv.py",
        "_support/model/decode_gqa_increfa_aiv.py",
        "_support/model/decode_gqa_increfa_mixed.py",
        "_support/model/decode_packed_qkv_rope_gqa_mixed24.py",
        "_support/model/decode_token_embedding.py",
        "_support/model/decode_linear_matmul_v3.py",
        "_support/model/decode_qkv_split.py",
        "_support/model/decode_kv_scatter_query.py",
        "_support/model/decode_gqa_attention_aiv.py",
        "_support/model/decode_gqa_attention_mixed24.py",
        "_support/model/decode_swiglu.py",
        "_support/model/decode_position_add.py",
        "_support/model/decode_rope_lookup.py",
        "_support/model/decode_kv_scatter.py",
    ):
        path = here / name
        digest.update(name.encode("utf-8"))
        digest.update(short_file_hash(path).encode("utf-8"))
    return digest.hexdigest()[:12]


def torchair_cache_dir_for_shape(
    cache_root: Path,
    *,
    batch_size: int,
    cache_length: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    model_dir: Path | None = None,
    linear_weight_format: str = DECODE_LINEAR_WEIGHT_FORMAT,
    optimization: str | DecodeOptimizationConfig = "baseline",
) -> Path:
    optimization = resolve_decode_optimization(optimization)
    model_hash = (
        short_file_hash(model_dir / "config.json")
        if model_dir is not None
        else "model_unknown"
    )
    shape_key = "_".join(
        [
            linear_weight_format,
            DECODE_ATTENTION,
            DECODE_CACHE_UPDATE,
            f"opt{cache_key_part(optimization.name)}",
            f"mode{cache_key_part(TORCHAIR_EXECUTION_MODE)}",
            f"dtype{cache_key_part(dtype or 'unknown')}",
            f"bs{int(batch_size)}",
            f"cache{int(cache_length)}",
            f"model{model_hash}",
            f"torch{cache_key_part(torch.__version__)}",
            f"torchnpu{torch_npu_version_label(device or torch.device('cpu'))}",
            f"torchair{torchair_version_label(device or torch.device('cpu'))}",
            f"src{decode_source_hash()}",
        ]
    )
    return cache_root.expanduser().resolve() / shape_key


def compile_text_decode_stage(
    stage: TextDecodeStage,
    *,
    backend_name: str,
    device: torch.device,
    cache_root: Path,
    batch_size: int,
    cache_length: int,
    dtype: torch.dtype | None = None,
    model_dir: Path | None = None,
    linear_weight_format: str = DECODE_LINEAR_WEIGHT_FORMAT,
    optimization: str | DecodeOptimizationConfig = "baseline",
) -> tuple[Any, dict[str, Any]]:
    optimization = resolve_decode_optimization(optimization)
    if optimization.attention == "gqa_aiv":
        if backend_name != "torchair":
            raise ValueError("gqa_aiv is an independent TorchAir-only operator")
        if batch_size != 1:
            raise ValueError("gqa_aiv currently supports only batch_size=1")
    if optimization.super_kernel_scope:
        if backend_name != "torchair":
            raise ValueError(
                "the Paddle decoder SuperKernel is a TorchAir-only path"
            )
        if batch_size != 1:
            raise ValueError(
                "the Paddle decoder SuperKernel currently supports only "
                "batch_size=1"
            )
        if optimization.ascendc_rope_lookup and cache_length != 1024:
            raise ValueError(
                "the specialized Paddle decoder RoPE lookup requires "
                "cache_length=1024"
            )
        if optimization.ascendc_rope_lookup and dtype != torch.float16:
            raise ValueError(
                "the specialized Paddle decoder RoPE lookup requires FP16"
            )
        if (
            optimization.ascendc_packed_qkv_rope_gqa_mixed24
            and (cache_length != 1024 or dtype != torch.float16)
        ):
            raise ValueError(
                "packed QKV/RoPE GQA requires FP16 with cache_length=1024"
            )
    common_metadata = {
        "backend": backend_name,
        "enabled": backend_name != "raw_eager",
        "boundary": "token_embedding_text_transformer_lm_head_static_step",
        "linear_weight_format": linear_weight_format,
        "decode_attention": decode_attention_label(device, optimization),
        "decode_cache_update": decode_cache_update_label(device, optimization),
        "decode_optimization": optimization.name,
        "decode_optimization_config": {
            "hoist_mrope": optimization.hoist_mrope,
            "packed_qkv": optimization.packed_qkv,
            "rms_norm": optimization.rms_norm,
            "rotary": optimization.rotary,
            "rotary_factors": optimization.rotary_factors,
            "packed_mlp": optimization.packed_mlp,
            "npu_swiglu": optimization.npu_swiglu,
            "add_rms_norm": optimization.add_rms_norm,
            "attention": optimization.attention,
            "increfa_length_mode": optimization.increfa_length_mode,
            "increfa_inner_precise": optimization.increfa_inner_precise,
            "stage_aware_weight_prefetch": (
                optimization.stage_aware_weight_prefetch
            ),
            "post_scatter_kv_prefetch": (
                optimization.post_scatter_kv_prefetch
            ),
            "weight_prefetch_timing": optimization.weight_prefetch_timing,
            "complete_layer_prefetch_ahead": (
                optimization.complete_layer_prefetch_ahead
            ),
            "prefetch_next_iteration": optimization.prefetch_next_iteration,
            "zero_residual_first_rms_norm": (
                optimization.zero_residual_first_rms_norm
            ),
            "packed_kv_scatter": optimization.packed_kv_scatter,
            "vector_add_rms_norm": optimization.vector_add_rms_norm,
            "gqa_aiv_vector_core_count": (
                optimization.gqa_aiv_vector_core_count
            ),
            "super_kernel_scope": optimization.super_kernel_scope,
            "super_kernel_options": optimization.super_kernel_options,
            "ascendc_token_embedding": optimization.ascendc_token_embedding,
            "ascendc_linear": optimization.ascendc_linear,
            "ascendc_qkv_split": optimization.ascendc_qkv_split,
            "ascendc_position_add": optimization.ascendc_position_add,
            "ascendc_rope_lookup": optimization.ascendc_rope_lookup,
            "ascendc_kv_scatter": optimization.ascendc_kv_scatter,
            "ascendc_kv_scatter_query": (
                optimization.ascendc_kv_scatter_query
            ),
            "ascendc_decode_gqa": optimization.ascendc_decode_gqa,
            "ascendc_decode_gqa_mixed": (
                optimization.ascendc_decode_gqa_mixed
            ),
            "ascendc_decode_gqa_mixed24": (
                optimization.ascendc_decode_gqa_mixed24
            ),
            "ascendc_packed_qkv_rope_gqa_mixed24": (
                optimization.ascendc_packed_qkv_rope_gqa_mixed24
            ),
            "ascendc_decode_gqa_attention": (
                optimization.ascendc_decode_gqa_attention
            ),
            "ascendc_decode_gqa_attention_mixed24": (
                optimization.ascendc_decode_gqa_attention_mixed24
            ),
            "ascendc_swiglu": optimization.ascendc_swiglu,
        },
    }
    if backend_name == "raw_eager":
        return stage, {**common_metadata, "compile_api": "none"}

    if backend_name == "torchair":
        if device.type != "npu":
            raise ValueError("--backend torchair requires an NPU device.")
        if optimization.vector_add_rms_norm:
            _register_vector_add_rms_norm_converter()
        torchair, CompilerConfig = import_torchair()
        if (
            optimization.attention == "gqa_aiv"
            and not optimization.ascendc_decode_gqa
            and not optimization.ascendc_decode_gqa_mixed
            and not optimization.ascendc_decode_gqa_mixed24
            and not optimization.ascendc_packed_qkv_rope_gqa_mixed24
            and not optimization.ascendc_decode_gqa_attention
            and not optimization.ascendc_decode_gqa_attention_mixed24
        ):
            register_gqa_increfa_aiv_converter()
        if optimization.ascendc_decode_gqa:
            register_decode_gqa_increfa_aiv_converter()
        if optimization.ascendc_decode_gqa_mixed:
            register_decode_gqa_increfa_mixed_converter()
        if optimization.ascendc_decode_gqa_mixed24:
            register_decode_gqa_increfa_mixed24_converter()
        if optimization.ascendc_packed_qkv_rope_gqa_mixed24:
            register_decode_packed_qkv_rope_gqa_mixed24_converter()
        if optimization.ascendc_decode_gqa_attention:
            register_decode_gqa_attention_aiv_converter()
        if optimization.ascendc_decode_gqa_attention_mixed24:
            register_decode_gqa_attention_mixed24_converter()
        if optimization.ascendc_token_embedding:
            register_decode_token_embedding_converter()
        if optimization.ascendc_linear:
            register_decode_linear_matmul_v3_converter()
        if optimization.ascendc_qkv_split:
            register_decode_qkv_split_converter()
        if optimization.ascendc_position_add:
            register_decode_position_add_converter()
        if optimization.ascendc_rope_lookup:
            register_decode_rope_lookup_converter()
        if optimization.ascendc_kv_scatter:
            register_decode_kv_scatter_converter()
        if optimization.ascendc_kv_scatter_query:
            register_decode_kv_scatter_query_converter()
        if optimization.ascendc_swiglu:
            register_decode_swiglu_converter()
        shape_cache_dir = torchair_cache_dir_for_shape(
            cache_root,
            batch_size=batch_size,
            cache_length=cache_length,
            dtype=dtype,
            device=device,
            model_dir=model_dir,
            linear_weight_format=linear_weight_format,
            optimization=optimization,
        )
        shape_cache_dir.mkdir(parents=True, exist_ok=True)
        # Multiple resident decode shapes must not share a Dynamo code object.
        # Otherwise constructing B1 after B8 invalidates B8's cached code and
        # TorchAir repeatedly rejects/scrubs the cache on subsequent calls.
        # This mirrors unique_spec_verify_forward; model math is unchanged.
        original = stage.forward.__func__
        name = f"text_decode_b{int(batch_size)}_kv{int(cache_length)}"
        function = types.FunctionType(
            original.__code__.replace(co_name=name), original.__globals__,
            name, original.__defaults__, original.__closure__,
        )
        function.__annotations__ = dict(original.__annotations__)
        function.__kwdefaults__ = original.__kwdefaults__
        entrypoint = types.MethodType(function, stage)
        compiled_decode = torchair.inference.cache_compile(
            entrypoint,
            config=CompilerConfig(),
            dynamic=False,
            cache_dir=str(shape_cache_dir),
            ge_cache=True,
        )
        return compiled_decode, {
            **common_metadata,
            "torchair_cache_dir": str(shape_cache_dir),
            "torchair_ge_cache": True,
            "compile_api": "torchair.inference.cache_compile",
            "cache_key_fields": {
                "batch_size": int(batch_size),
                "cache_length": int(cache_length),
                "dtype": str(dtype),
                "model_config_hash": (
                    short_file_hash(model_dir / "config.json")
                    if model_dir is not None
                    else None
                ),
                "torch": str(torch.__version__),
                "torch_npu": torch_npu_version_label(device),
                "torchair": torchair_version_label(device),
                "decode_source_hash": decode_source_hash(),
                "linear_weight_format": linear_weight_format,
                "decode_attention": decode_attention_label(
                    device, optimization
                ),
                "decode_cache_update": decode_cache_update_label(
                    device, optimization
                ),
                "execution_mode": TORCHAIR_EXECUTION_MODE,
                "decode_optimization": optimization.name,
            },
        }

    backend = compile_backend(backend_name)
    compile_kwargs = {"fullgraph": True, "dynamic": False}
    if backend is not None:
        compile_kwargs["backend"] = backend
    return torch.compile(stage, **compile_kwargs), {
        **common_metadata,
        "compile_api": "torch.compile",
    }


class TextDecodeRuntime:
    """Own the shared decode stage, its execution wrapper, and warm arena."""

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        *,
        backend: str,
        device: torch.device,
        cache_root: Path,
        batch_size: int,
        cache_length: int,
        dtype: torch.dtype,
        model_dir: Path,
        linear_weight_format: str,
        optimization: str | DecodeOptimizationConfig = "baseline",
    ):
        self.optimization = prepare_decode_optimization_modules(
            model,
            optimization,
        )
        prepare_decode_rope_factor_lut(
            model,
            self.optimization,
            cache_length=cache_length,
            dtype=dtype,
        )
        prepare_decode_weight_prefetch(model, self.optimization)
        self.stage = TextDecodeStage(
            model,
            optimization=self.optimization,
            cache_length=cache_length,
        ).eval()
        self.cache_num_key_value_heads = (
            int(model.config.text_config.num_attention_heads)
            if self.optimization.attention == "mha_cache"
            else int(model.config.text_config.num_key_value_heads)
        )
        synchronize(device)
        started = time.perf_counter()
        self.fn, self.metadata = compile_text_decode_stage(
            self.stage,
            backend_name=backend,
            device=device,
            cache_root=cache_root,
            batch_size=batch_size,
            cache_length=cache_length,
            dtype=dtype,
            model_dir=model_dir,
            linear_weight_format=linear_weight_format,
            optimization=self.optimization,
        )
        synchronize(device)
        compile_wrapper_s = time.perf_counter() - started

        self.warm_cache: LocalPaddleOCRVLStaticCache = model.allocate_static_cache(
            batch_size=batch_size,
            cache_length=cache_length,
            device=device,
            dtype=dtype,
            init_mode="zeros",
            num_key_value_heads=self.cache_num_key_value_heads,
            packed_kv=self.optimization.packed_kv_scatter,
        )
        self.metadata["cache_num_key_value_heads"] = (
            self.cache_num_key_value_heads
        )
        self.metadata["cache_allocated_bytes"] = sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in self.warm_cache.flat_tensors()
        )
        warm_input = torch.zeros((batch_size, 1), device=device, dtype=torch.int64)
        warm_position = torch.ones((batch_size,), device=device, dtype=torch.int64)
        warm_rope = torch.zeros((batch_size, 1), device=device, dtype=torch.int64)
        synchronize(device)
        started = time.perf_counter()
        self.fn(
            warm_input,
            warm_position,
            warm_rope,
            *self.warm_cache.flat_tensors(),
        )
        synchronize(device)
        compile_first_call_s = time.perf_counter() - started
        del warm_input, warm_position, warm_rope
        self.setup_timing_s = {
            "compile_wrapper": float(compile_wrapper_s),
            "compile_first_call": float(compile_first_call_s),
        }


# ---- Relocated text-prefill implementation (unchanged computation) ----

"""Unified eager and compiled model execution for text prefill.

Token embedding, multimodal embedding scatter, the LM head, and greedy argmax
remain outside this stage. This module prepares exact-shape or bucket-padded
multimodal embeddings and runs the same text transformer plus in-place KV-cache
population either eagerly or through TorchAir. It returns the hidden state at
the last real prompt token, so padded query rows never become observable.
"""

# Future annotations are enabled at the combined module's start.

import os
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from _support.model.compile_utils import TORCHAIR_EXECUTION_MODE, cache_key_part, import_torchair, short_file_hash, torch_npu_version_label, torchair_version_label
from _support.model.config import PaddleOCRTextConfig
pass  # Definition is now in this same module.
from _support.utils.timing import synchronize

if TYPE_CHECKING:
    from paddle_ocr_vl_1_6_modeling import LocalPaddleOCRVLForConditionalGeneration


DEFAULT_TEXT_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048)
TEXT_BACKEND_CHOICES = ("raw_eager", "torchair")
TEXT_PADDING_CHOICES = ("auto", "none", "bucket")
TEXT_SOFTMAX_DTYPE_ENV = "PADDLE_OCR_VL_TEXT_SOFTMAX_DTYPE"
SOFTMAX_DTYPE_CHOICES = ("fp32", "model")


def get_text_softmax_dtype_mode() -> str:
    mode = (
        os.environ.get(TEXT_SOFTMAX_DTYPE_ENV, "fp32").strip().lower()
        or "fp32"
    )
    if mode not in SOFTMAX_DTYPE_CHOICES:
        raise ValueError(
            f"{TEXT_SOFTMAX_DTYPE_ENV} must be one of "
            f"{SOFTMAX_DTYPE_CHOICES}, got {mode!r}"
        )
    return mode


def _activation(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "silu":
        return F.silu(x)
    if name == "gelu_pytorch_tanh":
        return F.gelu(x, approximate="tanh")
    if name == "gelu":
        return F.gelu(x)
    raise ValueError(f"unsupported activation: {name!r}")


def _prefill_linear_tokenwise(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Apply a Linear through a compiler-safe 2-D token matrix."""
    leading_shape = x.shape[:-1]
    output = linear(x.reshape(-1, x.shape[-1]))
    return output.reshape(*leading_shape, output.shape[-1])


def attention_softmax(
    scores: torch.Tensor,
    *,
    dim: int,
    output_dtype: torch.dtype,
    mode: str,
) -> torch.Tensor:
    if mode == "fp32":
        return F.softmax(scores, dim=dim, dtype=torch.float32).to(output_dtype)
    if mode == "model":
        return F.softmax(scores, dim=dim, dtype=output_dtype).to(output_dtype)
    raise ValueError(f"unsupported attention softmax dtype mode: {mode!r}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        n_rep,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(
        batch, num_key_value_heads * n_rep, seq_len, head_dim
    )


def apply_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    mrope_section = [int(value) for value in mrope_section] * 2
    cos = torch.cat(
        [
            part[i % 3]
            for i, part in enumerate(cos.split(mrope_section, dim=-1))
        ],
        dim=-1,
    )
    sin = torch.cat(
        [
            part[i % 3]
            for i, part in enumerate(sin.split(mrope_section, dim=-1))
        ],
        dim=-1,
    )
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (
        (q * cos) + (rotate_half(q) * sin),
        (k * cos) + (rotate_half(k) * sin),
    )


def build_causal_mask(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_length: int = 0,
) -> torch.Tensor:
    batch_size, query_length = inputs_embeds.shape[:2]
    if attention_mask is None:
        kv_length = int(past_length + query_length)
        attention_mask = torch.ones(
            batch_size,
            kv_length,
            device=inputs_embeds.device,
            dtype=torch.long,
        )
    else:
        kv_length = int(attention_mask.shape[-1])
    kv_positions = torch.arange(
        kv_length,
        device=inputs_embeds.device,
        dtype=cache_position.dtype,
    )
    allowed = kv_positions.unsqueeze(0) <= cache_position.reshape(-1, 1)
    allowed = allowed.reshape(1, 1, query_length, kv_length).expand(
        batch_size, 1, query_length, kv_length
    )
    padding_allowed = attention_mask[:, None, None, :kv_length].to(
        device=inputs_embeds.device, dtype=torch.bool
    )
    allowed = allowed & padding_allowed
    mask = torch.zeros(
        (batch_size, 1, query_length, kv_length),
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )
    return mask.masked_fill(
        ~allowed, torch.finfo(inputs_embeds.dtype).min
    )


def update_prefill_kv_cache_(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> None:
    sequence_length = int(key_states.shape[2])
    key_cache[:, :, :sequence_length, :].copy_(key_states.contiguous())
    value_cache[:, :, :sequence_length, :].copy_(value_states.contiguous())


class PaddleOCRRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return self.weight * hidden_states.to(input_dtype)


class PaddleOCRRotaryEmbedding(nn.Module):
    def __init__(self, config: PaddleOCRTextConfig):
        super().__init__()
        rope = config.rope_parameters or {}
        self.base = float(rope.get("rope_theta", 500000.0))
        self.dim = int(config.head_dim)
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)
        self.attention_scaling = 1.0

    def _compute_inv_freq(self) -> torch.Tensor:
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )

    def reset_inv_freq(self, device: torch.device | None = None) -> None:
        self.register_buffer(
            "inv_freq",
            self._compute_inv_freq().to(device=device),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, None, :, None].float().expand(
            3, position_ids.shape[1], -1, 1
        )
        position_ids = position_ids[:, :, None, :].float()
        freqs = (inv_freq * position_ids).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class PaddleOCRMLP(nn.Module):
    def __init__(self, config: PaddleOCRTextConfig):
        super().__init__()
        self.hidden_act = config.hidden_act
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = _prefill_linear_tokenwise(self.gate_proj, x)
        up = _prefill_linear_tokenwise(self.up_proj, x)
        return _prefill_linear_tokenwise(
            self.down_proj, _activation(self.hidden_act, gate) * up
        )


class PaddleOCRAttention(nn.Module):
    def __init__(self, config: PaddleOCRTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = config.head_dim**-0.5
        self.mrope_section = list(
            (config.rope_parameters or {})["mrope_section"]
        )
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.use_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.use_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.use_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.use_bias,
        )

    def project_qkv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, query_length, _hidden = hidden_states.shape
        query_states = _prefill_linear_tokenwise(
            self.q_proj, hidden_states
        ).view(
            batch, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = _prefill_linear_tokenwise(
            self.k_proj, hidden_states
        ).view(
            batch,
            query_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value_states = _prefill_linear_tokenwise(
            self.v_proj, hidden_states
        ).view(
            batch,
            query_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        return query_states, key_states, value_states

    def apply_rotary(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            position_embeddings[0],
            position_embeddings[1],
            self.mrope_section,
        )

    def attend(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, _heads, query_length, _dim = query_states.shape
        key_for_attn = repeat_kv(
            key_states, self.num_key_value_groups
        )
        value_for_attn = repeat_kv(
            value_states, self.num_key_value_groups
        )
        scores = (
            torch.matmul(query_states, key_for_attn.transpose(2, 3))
            * self.scaling
        )
        if attention_mask is not None:
            scores = scores + attention_mask[
                :, :, :, : key_for_attn.shape[-2]
            ]
        probs = attention_softmax(
            scores,
            dim=-1,
            output_dtype=query_states.dtype,
            mode=get_text_softmax_dtype_mode(),
        )
        attention_output = torch.matmul(probs, value_for_attn)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .reshape(batch, query_length, -1)
        )
        return _prefill_linear_tokenwise(self.o_proj, attention_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        query_states, key_states, value_states = self.project_qkv(hidden_states)
        query_states, key_states = self.apply_rotary(
            query_states, key_states, position_embeddings
        )
        if past_key_values is not None:
            past_key, past_value = past_key_values[self.layer_idx]
            key_states = torch.cat((past_key, key_states), dim=2)
            value_states = torch.cat((past_value, value_states), dim=2)
        new_past = (key_states, value_states) if use_cache else None
        return (
            self.attend(
                query_states, key_states, value_states, attention_mask
            ),
            new_past,
        )

    def forward_prefill_static(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states, key_states, value_states = self.project_qkv(hidden_states)
        query_states, key_states = self.apply_rotary(
            query_states, key_states, position_embeddings
        )
        return (
            self.attend(
                query_states, key_states, value_states, attention_mask
            ),
            key_states,
            value_states,
        )


class PaddleOCRDecoderLayer(nn.Module):
    def __init__(self, config: PaddleOCRTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.self_attn = PaddleOCRAttention(config, layer_idx)
        self.mlp = PaddleOCRMLP(config)
        self.input_layernorm = PaddleOCRRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = PaddleOCRRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_past = self.self_attn(
            hidden_states,
            attention_mask,
            position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states, new_past

    def apply_blocks(
        self,
        residual: torch.Tensor,
        attention_output: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def forward_prefill_static(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: LocalPaddleOCRVLStaticCache | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output, key_states, value_states = (
            self.self_attn.forward_prefill_static(
                hidden_states,
                attention_mask,
                position_embeddings,
            )
        )
        if cache is not None:
            key_cache, value_cache = cache.layer(self.layer_idx)
            update_prefill_kv_cache_(
                key_cache,
                value_cache,
                key_states,
                value_states,
            )
        return self.apply_blocks(residual, attention_output)


class PaddleOCRTextModel(nn.Module):
    def __init__(self, config: PaddleOCRTextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [
                PaddleOCRDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = PaddleOCRRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.rotary_emb = PaddleOCRRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]] | None,
    ]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            inputs_embeds = self.embed_tokens(input_ids)
        past_length = (
            0
            if past_key_values is None
            else int(past_key_values[0][0].shape[2])
        )
        cache_position = torch.arange(
            past_length,
            past_length + inputs_embeds.shape[1],
            device=inputs_embeds.device,
            dtype=torch.long,
        )
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(
                3, inputs_embeds.shape[0], -1
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, -1, -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        causal_mask = build_causal_mask(
            inputs_embeds,
            attention_mask,
            cache_position,
            past_length=past_length,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        new_past_key_values = [] if use_cache else None
        for layer in self.layers:
            hidden_states, new_past = layer(
                hidden_states,
                causal_mask,
                position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            if use_cache:
                new_past_key_values.append(new_past)
        return self.norm(hidden_states), new_past_key_values

    def forward_prefill_static(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        cache: LocalPaddleOCRVLStaticCache | None = None,
    ) -> torch.Tensor:
        cache_position = torch.arange(
            inputs_embeds.shape[1],
            device=inputs_embeds.device,
            dtype=torch.int64,
        )
        causal_mask = build_causal_mask(
            inputs_embeds, attention_mask, cache_position
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer.forward_prefill_static(
                hidden_states,
                causal_mask,
                position_embeddings,
                cache=cache,
            )
        return self.norm(hidden_states)


def parse_text_buckets(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        if not pieces:
            raise ValueError("text buckets cannot be empty")
        try:
            buckets = tuple(int(piece) for piece in pieces)
        except ValueError as exc:
            raise ValueError(f"invalid text buckets: {value!r}") from exc
    else:
        buckets = tuple(int(item) for item in value)
    if not buckets:
        raise ValueError("text buckets cannot be empty")
    if any(bucket <= 0 for bucket in buckets):
        raise ValueError("every text bucket must be positive")
    if tuple(sorted(set(buckets))) != buckets:
        raise ValueError("text buckets must be unique and strictly increasing")
    return buckets


def select_text_bucket(real_seq_len: int, buckets: Iterable[int]) -> int | None:
    real_seq_len = int(real_seq_len)
    if real_seq_len <= 0:
        raise ValueError("real text sequence length must be positive")
    for bucket in buckets:
        if real_seq_len <= int(bucket):
            return int(bucket)
    return None


class TextPrefillStage(torch.nn.Module):
    """Text prefill with flat mutable cache inputs for eager or compiled use."""

    def __init__(self, model: LocalPaddleOCRVLForConditionalGeneration):
        super().__init__()
        self.text_model = model.model
        self.num_layers = int(model.config.text_config.num_hidden_layers)
        self.softmax_dtype_mode = get_text_softmax_dtype_mode()

    def _attention(
        self,
        attention: torch.nn.Module,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        query_states, key_states, value_states = attention.project_qkv(hidden_states)
        query_states, key_states = attention.apply_rotary(
            query_states,
            key_states,
            position_embeddings,
        )
        update_prefill_kv_cache_(
            key_cache,
            value_cache,
            key_states,
            value_states,
        )
        key_for_attn = repeat_kv(key_states, int(attention.num_key_value_groups))
        value_for_attn = repeat_kv(value_states, int(attention.num_key_value_groups))
        batch, num_heads, seq_length, head_dim = query_states.shape

        # GE mis-infers the broadcast axes of the stock 4-D matmul. Flattening
        # B and H produces the same arithmetic while presenting two ordinary
        # 3-D batched matrix multiplications to the compiler.
        query_bh = query_states.reshape(batch * num_heads, seq_length, head_dim)
        key_bh = key_for_attn.reshape(batch * num_heads, seq_length, head_dim)
        value_bh = value_for_attn.reshape(batch * num_heads, seq_length, head_dim)
        scores = torch.bmm(query_bh, key_bh.transpose(1, 2)).view(
            batch,
            num_heads,
            seq_length,
            seq_length,
        ) * attention.scaling
        scores = scores + causal_mask
        probabilities = attention_softmax(
            scores,
            dim=-1,
            output_dtype=query_states.dtype,
            mode=self.softmax_dtype_mode,
        )
        attention_output = torch.bmm(
            probabilities.reshape(batch * num_heads, seq_length, seq_length),
            value_bh,
        ).view(batch, num_heads, seq_length, head_dim)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch,
            seq_length,
            num_heads * head_dim,
        )
        return _prefill_linear_tokenwise(attention.o_proj, attention_output)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        last_token_index: torch.Tensor,
        *flat_cache_tensors: torch.Tensor,
    ) -> torch.Tensor:
        key_caches = tuple(flat_cache_tensors[: self.num_layers])
        value_caches = tuple(flat_cache_tensors[self.num_layers :])
        cache_position = torch.arange(
            inputs_embeds.shape[1],
            device=inputs_embeds.device,
            dtype=torch.int64,
        )
        causal_mask = build_causal_mask(
            inputs_embeds,
            attention_mask,
            cache_position,
        )
        position_embeddings = self.text_model.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.text_model.layers):
            residual = hidden_states
            attention_input = layer.input_layernorm(hidden_states)
            attention_output = self._attention(
                layer.self_attn,
                attention_input,
                causal_mask,
                position_embeddings,
                key_caches[layer_idx],
                value_caches[layer_idx],
            )
            hidden_states = layer.apply_blocks(residual, attention_output)
        hidden_states = self.text_model.norm(hidden_states)
        return torch.index_select(hidden_states, 1, last_token_index)


def unique_bucket_forward(
    module: TextPrefillStage,
    bucket: int,
) -> Callable[..., torch.Tensor]:
    """Give each static bucket a distinct Dynamo code object."""

    original = module.forward.__func__
    name = f"text_prefill_bucket_{int(bucket)}"
    code = original.__code__.replace(co_name=name)
    function = types.FunctionType(
        code,
        original.__globals__,
        name,
        original.__defaults__,
        original.__closure__,
    )
    function.__annotations__ = dict(original.__annotations__)
    function.__kwdefaults__ = original.__kwdefaults__
    return types.MethodType(function, module)


@dataclass(frozen=True)
class PreparedTextPrefill:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    last_token_index: torch.Tensor
    real_seq_len: int
    physical_seq_len: int
    execution: str


def prepare_text_prefill(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    physical_seq_len: int,
    execution: str,
) -> PreparedTextPrefill:
    if inputs_embeds.ndim != 3 or int(inputs_embeds.shape[0]) != 1:
        raise ValueError(
            "text prefill expects B=1 embeddings shaped [1, S, H], "
            f"got {tuple(inputs_embeds.shape)}"
        )
    real_seq_len = int(inputs_embeds.shape[1])
    physical_seq_len = int(physical_seq_len)
    if real_seq_len > physical_seq_len:
        raise ValueError(
            f"real text sequence {real_seq_len} exceeds bucket {physical_seq_len}"
        )
    if tuple(attention_mask.shape) != (1, real_seq_len):
        raise ValueError(
            f"attention_mask must have shape {(1, real_seq_len)}, "
            f"got {tuple(attention_mask.shape)}"
        )
    if tuple(position_ids.shape) != (3, 1, real_seq_len):
        raise ValueError(
            f"position_ids must have shape {(3, 1, real_seq_len)}, "
            f"got {tuple(position_ids.shape)}"
        )

    pad_tokens = physical_seq_len - real_seq_len
    padded_embeds = F.pad(inputs_embeds, (0, 0, 0, pad_tokens)).contiguous()
    padded_mask = F.pad(attention_mask, (0, pad_tokens), value=0).contiguous()
    # get_rope_index uses position 1 for masked/padded rows. The padded query
    # results are discarded, but preserving that convention keeps the graph's
    # unused rows well defined.
    padded_positions = F.pad(position_ids, (0, pad_tokens), value=1).contiguous()
    last_token_index = torch.tensor(
        [real_seq_len - 1],
        device=inputs_embeds.device,
        dtype=torch.int64,
    )
    return PreparedTextPrefill(
        inputs_embeds=padded_embeds,
        attention_mask=padded_mask,
        position_ids=padded_positions,
        last_token_index=last_token_index,
        real_seq_len=real_seq_len,
        physical_seq_len=physical_seq_len,
        execution=str(execution),
    )


def text_source_hash() -> str:
    return short_file_hash(Path(__file__).resolve())


def text_cache_dir_for_bucket(
    cache_root: Path,
    *,
    bucket: int,
    cache_length: int,
    dtype: torch.dtype,
    device: torch.device,
    model_dir: Path,
    linear_weight_format: str,
) -> Path:
    key = "_".join(
        [
            "text_transformer_prefill",
            f"mode{cache_key_part(TORCHAIR_EXECUTION_MODE)}",
            f"softmax{cache_key_part(get_text_softmax_dtype_mode())}",
            "bs1",
            f"seq{int(bucket)}",
            f"cache{int(cache_length)}",
            f"weights{cache_key_part(linear_weight_format)}",
            f"dtype{cache_key_part(dtype)}",
            f"model{short_file_hash(model_dir / 'config.json')}",
            f"torch{cache_key_part(torch.__version__)}",
            f"torchnpu{torch_npu_version_label(device)}",
            f"torchair{torchair_version_label(device)}",
            f"src{text_source_hash()}",
        ]
    )
    return cache_root.expanduser().resolve() / key


class TextPrefillRuntime:
    """Run one text-prefill stage eagerly or through static bucket graphs."""

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        *,
        backend: str,
        buckets: Iterable[int],
        cache_root: Path,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
        model_dir: Path,
        linear_weight_format: str,
        padding: str = "auto",
    ):
        self.model = model
        self.backend = str(backend)
        self.buckets = parse_text_buckets(buckets)
        self.requested_padding = str(padding)
        if self.requested_padding not in TEXT_PADDING_CHOICES:
            raise ValueError(
                f"text padding must be one of {TEXT_PADDING_CHOICES}, got {padding!r}"
            )
        self.padding = (
            "bucket"
            if self.requested_padding == "auto" and self.backend == "torchair"
            else "none"
            if self.requested_padding == "auto"
            else self.requested_padding
        )
        if self.backend == "torchair" and self.padding != "bucket":
            raise ValueError("compiled text prefill requires bucket padding")
        self.cache_root = cache_root.expanduser().resolve()
        self.cache_length = int(cache_length)
        self.device = device
        self.dtype = dtype
        self.compiled: dict[int, Callable[..., torch.Tensor]] = {}
        self.entrypoints: dict[int, Callable[..., torch.Tensor]] = {}
        self.eager_stage = TextPrefillStage(model).eval()
        self.modules: dict[int, TextPrefillStage] = {}
        self.metadata: dict[str, Any] = {
            "backend": self.backend,
            "enabled": self.backend == "torchair",
            "boundary": "text_transformer_plus_in_place_prefill_kv_writes",
            "buckets": list(self.buckets),
            "requested_padding": self.requested_padding,
            "padding": self.padding,
            "overflow": (
                "eager_same_stage_unpadded"
                if self.padding == "bucket"
                else None
            ),
        }
        if self.backend not in TEXT_BACKEND_CHOICES:
            raise ValueError(
                f"text backend must be one of {TEXT_BACKEND_CHOICES}, got {backend!r}"
            )
        if self.padding == "bucket" and self.buckets[-1] > self.cache_length:
            raise ValueError(
                f"largest text bucket {self.buckets[-1]} exceeds cache length "
                f"{self.cache_length}"
            )
        if self.backend == "raw_eager":
            return
        if self.device.type != "npu":
            raise ValueError("compiled text backend torchair requires an NPU device")

        torchair, CompilerConfig = import_torchair()
        hidden_size = int(model.config.text_config.hidden_size)
        per_bucket: dict[str, Any] = {}
        wrapper_total_s = 0.0
        first_call_total_s = 0.0
        for bucket in self.buckets:
            module = TextPrefillStage(model).eval()
            cache_dir = text_cache_dir_for_bucket(
                self.cache_root,
                bucket=bucket,
                cache_length=self.cache_length,
                dtype=self.dtype,
                device=self.device,
                model_dir=model_dir,
                linear_weight_format=linear_weight_format,
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            config = CompilerConfig()
            entrypoint = unique_bucket_forward(module, bucket)
            synchronize(self.device)
            started = time.perf_counter()
            compiled = torchair.inference.cache_compile(
                entrypoint,
                config=config,
                dynamic=False,
                cache_dir=str(cache_dir),
                ge_cache=True,
            )
            synchronize(self.device)
            wrapper_s = time.perf_counter() - started

            warm_inputs = torch.zeros(
                (1, bucket, hidden_size),
                device=self.device,
                dtype=self.dtype,
            )
            warm_mask = torch.ones(
                (1, bucket),
                device=self.device,
                dtype=torch.int64,
            )
            warm_positions = torch.zeros(
                (3, 1, bucket),
                device=self.device,
                dtype=torch.int64,
            )
            warm_last_index = torch.tensor(
                [bucket - 1],
                device=self.device,
                dtype=torch.int64,
            )
            warm_cache = model.allocate_static_cache(
                batch_size=1,
                cache_length=self.cache_length,
                device=self.device,
                dtype=self.dtype,
                init_mode="zeros",
            )
            synchronize(self.device)
            started = time.perf_counter()
            warm_output = compiled(
                warm_inputs,
                warm_mask,
                warm_positions,
                warm_last_index,
                *warm_cache.flat_tensors(),
            )
            synchronize(self.device)
            first_call_s = time.perf_counter() - started
            del warm_output, warm_inputs, warm_mask, warm_positions, warm_last_index, warm_cache

            self.modules[bucket] = module
            self.entrypoints[bucket] = entrypoint
            self.compiled[bucket] = compiled
            wrapper_total_s += wrapper_s
            first_call_total_s += first_call_s
            per_bucket[str(bucket)] = {
                "compile_wrapper_s": float(wrapper_s),
                "compile_first_call_s": float(first_call_s),
                "torchair_cache_dir": str(cache_dir),
            }
        self.metadata.update(
            {
                "compile_api": "torchair.inference.cache_compile",
                "dynamic": False,
                "fullgraph": True,
                "torchair_ge_cache": True,
                "compile_wrapper_total_s": float(wrapper_total_s),
                "compile_first_call_total_s": float(first_call_total_s),
                "per_bucket": per_bucket,
                "cache_key_fields": {
                    "cache_length": self.cache_length,
                    "dtype": str(dtype),
                    "linear_weight_format": linear_weight_format,
                    "model_config_hash": short_file_hash(model_dir / "config.json"),
                    "torch": str(torch.__version__),
                    "torch_npu": torch_npu_version_label(device),
                    "torchair": torchair_version_label(device),
                    "text_source_hash": text_source_hash(),
                    "attention": "manual_causal",
                    "softmax_dtype": get_text_softmax_dtype_mode(),
                    "execution_mode": TORCHAIR_EXECUTION_MODE,
                },
            }
        )

    def route(self, real_seq_len: int) -> dict[str, Any]:
        real_seq_len = int(real_seq_len)
        bucket = (
            select_text_bucket(real_seq_len, self.buckets)
            if self.padding == "bucket"
            else None
        )
        if bucket is None:
            return {
                "execution": (
                    "eager_overflow"
                    if self.padding == "bucket"
                    else "eager"
                ),
                "real_text_tokens": real_seq_len,
                "physical_text_tokens": real_seq_len,
                "padding_text_tokens": 0,
                "useful_token_fraction": 1.0,
                "bucket": None,
            }
        return {
            "execution": (
                "compiled" if self.backend == "torchair" else "eager_padded"
            ),
            "real_text_tokens": real_seq_len,
            "physical_text_tokens": bucket,
            "padding_text_tokens": bucket - real_seq_len,
            "useful_token_fraction": float(real_seq_len) / float(bucket),
            "bucket": bucket,
        }

    def prepare(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        route: dict[str, Any],
    ) -> PreparedTextPrefill:
        return prepare_text_prefill(
            inputs_embeds,
            attention_mask,
            position_ids,
            physical_seq_len=int(route["physical_text_tokens"]),
            execution=str(route["execution"]),
        )

    def run_prepared(
        self,
        prepared: PreparedTextPrefill,
        cache: LocalPaddleOCRVLStaticCache,
    ) -> torch.Tensor:
        run = (
            self.compiled[prepared.physical_seq_len]
            if prepared.execution == "compiled"
            else self.eager_stage
        )
        return run(
            prepared.inputs_embeds,
            prepared.attention_mask,
            prepared.position_ids,
            prepared.last_token_index,
            *cache.flat_tensors(),
        )
