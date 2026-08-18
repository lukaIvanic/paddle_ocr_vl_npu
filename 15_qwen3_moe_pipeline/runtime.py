#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from checkpoint import load_pipeline_stage
from modeling_qwen3_moe_pipeline import Qwen3MoeConfig, Qwen3MoePipelineStage


def stage_log(stage_name: str, message: str) -> None:
    print(f"[{stage_name}] {message}", flush=True)


def memory_snapshot(device: torch.device) -> dict[str, float | str]:
    torch.npu.synchronize(device)
    with torch.npu.device(device):
        free_bytes, total_bytes = torch.npu.mem_get_info()
        return {
            "device": str(device),
            "allocated_gib": torch.npu.memory_allocated(device) / 1024**3,
            "reserved_gib": torch.npu.memory_reserved(device) / 1024**3,
            "free_gib": free_bytes / 1024**3,
            "total_gib": total_bytes / 1024**3,
        }


def build_stage(
    config: Qwen3MoeConfig,
    model_dir: str | Path,
    *,
    layer_start: int,
    layer_end: int,
    with_embedding: bool,
    with_lm_head: bool,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    name: str,
    cache_length: int,
    expert_impl: str = "selected_bmm",
) -> tuple[Qwen3MoePipelineStage, dict[str, object]]:
    started = time.perf_counter()
    with torch.device("meta"):
        stage = Qwen3MoePipelineStage(
            config,
            layer_start=layer_start,
            layer_end=layer_end,
            with_embedding=with_embedding,
            with_lm_head=with_lm_head,
            expert_impl=expert_impl,
        )
    stage = stage.to(dtype=dtype)
    stage.to_empty(device=device)
    allocated = memory_snapshot(device)
    stage_log(name, "allocated empty stage: " + json.dumps(allocated, sort_keys=True))
    load_pipeline_stage(
        stage,
        model_dir,
        device=device,
        progress=lambda message: stage_log(name, message),
    )
    stage.prepare_decode(cache_length=cache_length)
    stage.eval()
    loaded = memory_snapshot(device)
    metadata = {
        "name": name,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "device": str(device),
        "expert_impl": expert_impl,
        "allocated_memory": allocated,
        "loaded_memory": loaded,
        "load_elapsed_sec": time.perf_counter() - started,
    }
    stage_log(name, "ready: " + json.dumps(metadata, sort_keys=True))
    return stage, metadata
