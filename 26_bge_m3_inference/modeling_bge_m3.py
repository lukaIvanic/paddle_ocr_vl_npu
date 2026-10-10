# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0:
# https://www.apache.org/licenses/LICENSE-2.0
# Simplified from Transformers' modeling_xlm_roberta.py; see README.md.
"""Inference-only BGE-M3 dense encoder. No Transformers model dependency."""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int
    type_vocab_size: int
    pad_token_id: int
    layer_norm_eps: float

    @classmethod
    def from_dict(cls, raw: dict) -> Config:
        if raw.get("model_type") != "xlm-roberta" or raw.get("hidden_act") != "gelu":
            raise ValueError("Expected an XLM-RoBERTa encoder with GELU")
        if raw.get("is_decoder", False) or raw.get("add_cross_attention", False):
            raise ValueError("Decoder/cross-attention checkpoints are unsupported")
        if raw.get("position_embedding_type", "absolute") != "absolute":
            raise ValueError("Only absolute position embeddings are supported")
        config = cls(**{f.name: raw[f.name] for f in fields(cls)})
        if config.hidden_size % config.num_attention_heads or config.type_vocab_size != 1:
            raise ValueError("Invalid attention dimensions or non-BGE token types")
        return config

    @classmethod
    def from_directory(cls, path: str | Path) -> Config:
        return cls.from_dict(json.loads((Path(path) / "config.json").read_text()))


class Embeddings(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.padding_idx = c.pad_token_id
        self.word_embeddings = nn.Embedding(c.vocab_size, c.hidden_size, padding_idx=c.pad_token_id)
        self.token_type_embeddings = nn.Embedding(c.type_vocab_size, c.hidden_size)
        self.position_embeddings = nn.Embedding(c.max_position_embeddings, c.hidden_size, padding_idx=c.pad_token_id)
        self.LayerNorm = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(self.padding_idx).int()
        positions = (mask.cumsum(dim=1).to(mask.dtype) * mask).long() + self.padding_idx
        # Preserve Transformers' addition order, including the learned type-0 vector.
        hidden = self.word_embeddings(input_ids) + self.token_type_embeddings(torch.zeros_like(input_ids))
        return self.LayerNorm(hidden + self.position_embeddings(positions))


class SelfAttention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.heads = c.num_attention_heads
        self.head_dim = c.hidden_size // self.heads
        self.query = nn.Linear(c.hidden_size, c.hidden_size)
        self.key = nn.Linear(c.hidden_size, c.hidden_size)
        self.value = nn.Linear(c.hidden_size, c.hidden_size)
        self.qkv = None

    def forward(self, hidden: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, s, h = hidden.shape
        q, k, v = ((self.query(hidden), self.key(hidden), self.value(hidden))
                   if self.qkv is None else self.qkv(hidden))
        q = q.view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.heads, self.head_dim).transpose(1, 2)
        # Explicit head batches avoid GE's broadcast-matmul shape inference.
        q = q.reshape(b * self.heads, s, self.head_dim)
        k = k.reshape(b * self.heads, s, self.head_dim)
        v = v.reshape(b * self.heads, s, self.head_dim)
        scores = torch.bmm(q, k.transpose(1, 2)) * (self.head_dim ** -0.5)
        scores = scores.view(b, self.heads, s, s) + bias
        probs = torch.softmax(scores, dim=-1).view(b * self.heads, s, s)
        context = torch.bmm(probs, v).view(b, self.heads, s, self.head_dim)
        return context.transpose(1, 2).contiguous().view(b, s, h)


class ResidualOutput(nn.Module):
    def __init__(self, c: Config, input_size: int):
        super().__init__()
        self.dense = nn.Linear(input_size, c.hidden_size)
        self.LayerNorm = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)

    def forward(self, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.dense(hidden) + residual)


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.self = SelfAttention(c)
        self.output = ResidualOutput(c, c.hidden_size)

    def forward(self, hidden: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return self.output(self.self(hidden, bias), hidden)


class Intermediate(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.dense = nn.Linear(c.hidden_size, c.intermediate_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.dense(hidden))


class Layer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.attention = Attention(c)
        self.intermediate = Intermediate(c)
        self.output = ResidualOutput(c, c.intermediate_size)

    def forward(self, hidden: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        hidden = self.attention(hidden, bias)
        return self.output(self.intermediate(hidden), hidden)


class Encoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.layer = nn.ModuleList([Layer(c) for _ in range(c.num_hidden_layers)])

    def forward(self, hidden: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        for layer in self.layer:
            hidden = layer(hidden, bias)
        return hidden


class BGEM3(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)

    def forward_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embeddings(input_ids)
        bias = (1.0 - attention_mask[:, None, None, :].to(hidden.dtype)) * torch.finfo(hidden.dtype).min
        return self.encoder(hidden, bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_hidden_states(input_ids, attention_mask)
        return F.normalize(hidden[:, 0], p=2, dim=-1)


def load_model(directory: str | Path, device: torch.device, dtype: torch.dtype = torch.float16) -> BGEM3:
    directory = Path(directory)
    config = Config.from_directory(directory)
    # The pinned BAAI release uses pytorch_model.bin. Keep the exact HF key names.
    state = torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True)
    # XLMRobertaModel's optional tanh pooler is unused by BGE's CLS pooling.
    for key in ("pooler.dense.weight", "pooler.dense.bias"):
        state.pop(key, None)
    with torch.device("meta"):
        model = BGEM3(config)
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(device=device, dtype=dtype).eval()
