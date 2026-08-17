#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch

from checkpoint import SafeTensorIndex, copy_parameter
from modeling_qwen3_moe_tp2 import Qwen3MoeTPStage, shard_bounds


def load_tp_stage_checkpoint(
    stage: Qwen3MoeTPStage,
    model_dir: str | Path,
    *,
    device: torch.device,
    progress: Callable[[str], None] | None = None,
) -> None:
    log = progress or (lambda _message: None)
    config = stage.config
    q_start, q_end = shard_bounds(
        config.num_attention_heads * config.head_dim,
        stage.tp_rank,
        stage.tp_size,
    )
    kv_start, kv_end = shard_bounds(
        config.num_key_value_heads * config.head_dim,
        stage.tp_rank,
        stage.tp_size,
    )
    intermediate_start, intermediate_end = shard_bounds(
        config.moe_intermediate_size,
        stage.tp_rank,
        stage.tp_size,
    )
    vocab_start, vocab_end = shard_bounds(
        config.vocab_size,
        stage.tp_rank,
        stage.tp_size,
    )

    with SafeTensorIndex(model_dir) as checkpoint, torch.no_grad():
        if stage.embed_tokens is not None:
            copy_parameter(
                stage.embed_tokens.weight,
                checkpoint.tensor("model.embed_tokens.weight")[
                    vocab_start:vocab_end
                ],
                device=device,
            )
            log("loaded vocabulary-sharded token embedding")
        for local_index, layer in enumerate(stage.layers):
            global_index = stage.layer_start + local_index
            prefix = f"model.layers.{global_index}"
            q_weight = checkpoint.tensor(f"{prefix}.self_attn.q_proj.weight")[
                q_start:q_end
            ]
            k_weight = checkpoint.tensor(f"{prefix}.self_attn.k_proj.weight")[
                kv_start:kv_end
            ]
            v_weight = checkpoint.tensor(f"{prefix}.self_attn.v_proj.weight")[
                kv_start:kv_end
            ]
            copy_parameter(
                layer.self_attn.qkv_proj.weight,
                torch.cat((q_weight, k_weight, v_weight), dim=0),
                device=device,
            )
            copy_parameter(
                layer.self_attn.o_proj.weight,
                checkpoint.tensor(f"{prefix}.self_attn.o_proj.weight")[
                    :, q_start:q_end
                ],
                device=device,
            )
            for name in ("q_norm", "k_norm"):
                copy_parameter(
                    getattr(layer.self_attn, name).weight,
                    checkpoint.tensor(f"{prefix}.self_attn.{name}.weight"),
                    device=device,
                )
            copy_parameter(
                layer.input_layernorm.weight,
                checkpoint.tensor(f"{prefix}.input_layernorm.weight"),
                device=device,
            )
            copy_parameter(
                layer.post_attention_layernorm.weight,
                checkpoint.tensor(f"{prefix}.post_attention_layernorm.weight"),
                device=device,
            )
            copy_parameter(
                layer.mlp.gate.weight,
                checkpoint.tensor(f"{prefix}.mlp.gate.weight"),
                device=device,
            )

            local_intermediate = intermediate_end - intermediate_start
            gate_up_cpu = torch.empty(
                (
                    config.num_experts,
                    2 * local_intermediate,
                    config.hidden_size,
                ),
                dtype=checkpoint.tensor(
                    f"{prefix}.mlp.experts.0.gate_proj.weight"
                ).dtype,
            )
            down_cpu = torch.empty(
                (
                    config.num_experts,
                    config.hidden_size,
                    local_intermediate,
                ),
                dtype=checkpoint.tensor(
                    f"{prefix}.mlp.experts.0.down_proj.weight"
                ).dtype,
            )
            for expert_index in range(config.num_experts):
                expert_prefix = f"{prefix}.mlp.experts.{expert_index}"
                gate_up_cpu[expert_index, :local_intermediate].copy_(
                    checkpoint.tensor(f"{expert_prefix}.gate_proj.weight")[
                        intermediate_start:intermediate_end
                    ]
                )
                gate_up_cpu[expert_index, local_intermediate:].copy_(
                    checkpoint.tensor(f"{expert_prefix}.up_proj.weight")[
                        intermediate_start:intermediate_end
                    ]
                )
                down_cpu[expert_index].copy_(
                    checkpoint.tensor(f"{expert_prefix}.down_proj.weight")[
                        :, intermediate_start:intermediate_end
                    ]
                )
            copy_parameter(layer.mlp.gate_up_proj, gate_up_cpu, device=device)
            copy_parameter(layer.mlp.down_proj, down_cpu, device=device)
            del gate_up_cpu, down_cpu
            log(
                f"loaded global layer {global_index} "
                f"({local_index + 1}/{stage.num_layers})"
            )

        if stage.norm is not None:
            copy_parameter(
                stage.norm.weight,
                checkpoint.tensor("model.norm.weight"),
                device=device,
            )
        if stage.lm_head is not None:
            copy_parameter(
                stage.lm_head.weight,
                checkpoint.tensor("lm_head.weight")[vocab_start:vocab_end],
                device=device,
            )
            log("loaded vocabulary-sharded LM head")
