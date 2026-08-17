#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open

from modeling_qwen3_tp2 import Qwen3TPForCausalLM, shard_bounds


class SafeTensorIndex:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with index_path.open() as handle:
                self.weight_map = json.load(handle)["weight_map"]
        else:
            files = sorted(self.model_dir.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(
                    f"No safetensors checkpoint found under {self.model_dir}"
                )
            self.weight_map = {}
            for path in files:
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        if key in self.weight_map:
                            raise RuntimeError(f"Duplicate checkpoint tensor: {key}")
                        self.weight_map[key] = path.name
        self._stack = contextlib.ExitStack()
        self._handles = {}

    def __enter__(self) -> "SafeTensorIndex":
        for filename in sorted(set(self.weight_map.values())):
            self._handles[filename] = self._stack.enter_context(
                safe_open(
                    str(self.model_dir / filename), framework="pt", device="cpu"
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stack.close()

    def tensor(self, name: str) -> torch.Tensor:
        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(f"Checkpoint tensor is missing: {name}")
        return self._handles[filename].get_tensor(name)


def _copy_parameter(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    device: torch.device,
) -> None:
    if tuple(target.shape) != tuple(source.shape):
        raise RuntimeError(
            f"Weight shape mismatch: target={tuple(target.shape)} "
            f"source={tuple(source.shape)}"
        )
    staged = source.contiguous().to(device=device, dtype=target.dtype)
    target.copy_(staged)


def load_tp_checkpoint(
    model: Qwen3TPForCausalLM,
    model_dir: str | Path,
    *,
    device: torch.device,
    progress: Callable[[str], None] | None = None,
) -> None:
    log = progress or (lambda _message: None)
    config = model.config
    q_start, q_end = shard_bounds(
        config.num_attention_heads * config.head_dim,
        model.tp_rank,
        model.tp_size,
    )
    kv_start, kv_end = shard_bounds(
        config.num_key_value_heads * config.head_dim,
        model.tp_rank,
        model.tp_size,
    )
    intermediate_start, intermediate_end = shard_bounds(
        config.intermediate_size,
        model.tp_rank,
        model.tp_size,
    )
    vocab_start, vocab_end = shard_bounds(
        config.vocab_size,
        model.tp_rank,
        model.tp_size,
    )

    with SafeTensorIndex(model_dir) as checkpoint, torch.no_grad():
        log("loading vocabulary-sharded embedding")
        _copy_parameter(
            model.embed_tokens.weight,
            checkpoint.tensor("model.embed_tokens.weight")[vocab_start:vocab_end],
            device=device,
        )

        for layer_index, layer in enumerate(model.layers):
            prefix = f"model.layers.{layer_index}"
            q_weight = checkpoint.tensor(f"{prefix}.self_attn.q_proj.weight")[
                q_start:q_end
            ]
            k_weight = checkpoint.tensor(f"{prefix}.self_attn.k_proj.weight")[
                kv_start:kv_end
            ]
            v_weight = checkpoint.tensor(f"{prefix}.self_attn.v_proj.weight")[
                kv_start:kv_end
            ]
            _copy_parameter(
                layer.self_attn.qkv_proj.weight,
                torch.cat((q_weight, k_weight, v_weight), dim=0),
                device=device,
            )
            _copy_parameter(
                layer.self_attn.o_proj.weight,
                checkpoint.tensor(f"{prefix}.self_attn.o_proj.weight")[
                    :, q_start:q_end
                ],
                device=device,
            )
            _copy_parameter(
                layer.self_attn.q_norm.weight,
                checkpoint.tensor(f"{prefix}.self_attn.q_norm.weight"),
                device=device,
            )
            _copy_parameter(
                layer.self_attn.k_norm.weight,
                checkpoint.tensor(f"{prefix}.self_attn.k_norm.weight"),
                device=device,
            )

            gate_weight = checkpoint.tensor(f"{prefix}.mlp.gate_proj.weight")[
                intermediate_start:intermediate_end
            ]
            up_weight = checkpoint.tensor(f"{prefix}.mlp.up_proj.weight")[
                intermediate_start:intermediate_end
            ]
            _copy_parameter(
                layer.mlp.gate_up_proj.weight,
                torch.cat((gate_weight, up_weight), dim=0),
                device=device,
            )
            _copy_parameter(
                layer.mlp.down_proj.weight,
                checkpoint.tensor(f"{prefix}.mlp.down_proj.weight")[
                    :, intermediate_start:intermediate_end
                ],
                device=device,
            )
            _copy_parameter(
                layer.input_layernorm.weight,
                checkpoint.tensor(f"{prefix}.input_layernorm.weight"),
                device=device,
            )
            _copy_parameter(
                layer.post_attention_layernorm.weight,
                checkpoint.tensor(f"{prefix}.post_attention_layernorm.weight"),
                device=device,
            )
            log(f"loaded layer {layer_index + 1}/{len(model.layers)}")

        _copy_parameter(
            model.norm.weight,
            checkpoint.tensor("model.norm.weight"),
            device=device,
        )
        if config.tie_word_embeddings:
            model.lm_head.weight = model.embed_tokens.weight
        else:
            _copy_parameter(
                model.lm_head.weight,
                checkpoint.tensor("lm_head.weight")[vocab_start:vocab_end],
                device=device,
            )
        log("loaded vocabulary-sharded LM head")
