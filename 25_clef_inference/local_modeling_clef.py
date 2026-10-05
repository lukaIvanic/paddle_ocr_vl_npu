# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Licensed under Apache-2.0; see LICENSE.apache-2.0 and README.md.
"""Faithful eager Clef-flash text model, without Transformers.

Backbone, joint decision head and checkpoint loading live in this file.
B1 unpadded text, BF16 parameters, no generation or cache reuse. The backbone
preserves Transformers 5.17.0 arithmetic; the head preserves the pinned
Cloudflare release's computation. Source provenance is recorded in README.md.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
from types import SimpleNamespace

from safetensors import safe_open
from safetensors.torch import load_file
import torch
from torch import nn
from torch.nn import functional as F

from download_model import verify


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.float())).to(x.dtype)


class GatedRMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x, gate):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        # The cast before the learned weight is intentional (different from RMSNorm).
        y = self.weight * y.to(x.dtype)
        return (y * F.silu(gate.float())).to(x.dtype)


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64):
    """Reference FP32 chunk scan, starting from an empty recurrent state."""
    dtype = query.dtype
    batch, length, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    query, key, value, beta, g = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]
    query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    query = query * key_dim**-0.5
    padding = (-length) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, padding)) for x in (query, key, value))
    beta, g = (F.pad(x, (0, padding)) for x in (beta, g))
    v_beta, k_beta = value * beta.unsqueeze(-1), key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [
        x.reshape(batch, heads, -1, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    decay = g.reshape(batch, heads, -1, chunk_size).cumsum(dim=3)
    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    pairwise = (decay.unsqueeze(4) - decay.unsqueeze(3)).masked_fill(upper, -float("inf")).exp()
    system = (k_beta @ key.transpose(-1, -2)) * pairwise
    attention = (query @ key.transpose(-1, -2)) * pairwise
    decayed_keys = k_beta * decay.exp().unsqueeze(-1)
    values = torch.linalg.solve_triangular(system, v_beta, upper=False, unitriangular=True)
    keys = torch.linalg.solve_triangular(system, decayed_keys, upper=False, unitriangular=True)
    state = torch.zeros(batch, heads, key_dim, value_dim, dtype=values.dtype, device=values.device)
    output = torch.zeros_like(values)
    query = query * decay.exp().unsqueeze(-1)
    key = key * (decay[..., -1:] - decay).exp().unsqueeze(-1)
    chunk_decay = decay[..., -1].exp()[..., None, None]
    for i in range((length + padding) // chunk_size):
        delta = values[:, :, i] - keys[:, :, i] @ state
        output[:, :, i] = query[:, :, i] @ state + attention[:, :, i] @ delta
        state = state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ delta
    output = output.reshape(batch, heads, -1, value_dim)[:, :, :length]
    return output.transpose(1, 2).to(dtype, memory_format=torch.contiguous_format)


class GatedDeltaNet(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.key_heads = c.linear_num_key_heads
        self.value_heads = c.linear_num_value_heads
        self.key_dim = c.linear_key_head_dim
        self.value_dim = c.linear_value_head_dim
        self.keys = self.key_heads * self.key_dim
        self.values = self.value_heads * self.value_dim
        channels = 2 * self.keys + self.values
        self.conv1d = nn.Conv1d(channels, channels, c.linear_conv_kernel_dim,
                               groups=channels, padding=c.linear_conv_kernel_dim - 1, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.value_heads))
        self.A_log = nn.Parameter(torch.empty(self.value_heads))
        self.in_proj_qkv = nn.Linear(c.hidden_size, channels, bias=False)
        self.in_proj_z = nn.Linear(c.hidden_size, self.values, bias=False)
        self.in_proj_b = nn.Linear(c.hidden_size, self.value_heads, bias=False)
        self.in_proj_a = nn.Linear(c.hidden_size, self.value_heads, bias=False)
        self.norm = GatedRMSNorm(self.value_dim, c.rms_norm_eps)
        self.out_proj = nn.Linear(self.values, c.hidden_size, bias=False)

    def forward(self, x):
        batch, length, _ = x.shape
        qkv = self.in_proj_qkv(x).transpose(1, 2)
        z = self.in_proj_z(x).reshape(batch, length, -1, self.value_dim)
        b, a = self.in_proj_b(x), self.in_proj_a(x)
        qkv = F.silu(self.conv1d(qkv)[:, :, :length]).transpose(1, 2)
        q, k, v = torch.split(qkv, [self.keys, self.keys, self.values], dim=-1)
        q, k = (t.reshape(batch, length, -1, self.key_dim) for t in (q, k))
        v = v.reshape(batch, length, -1, self.value_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        repeats = self.value_heads // self.key_heads
        q, k = (t.repeat_interleave(repeats, dim=2) for t in (q, k))
        y = chunk_gated_delta_rule(q, k, v, g, beta)
        y = self.norm(y.reshape(-1, self.value_dim), z.reshape(-1, self.value_dim))
        return self.out_proj(y.reshape(batch, length, -1))


def apply_rope(x, cos, sin):
    size = cos.shape[-1]
    rotary, rest = x[..., :size], x[..., size:]
    half = size // 2
    rotated = torch.cat((-rotary[..., half:], rotary[..., :half]), dim=-1)
    return torch.cat((rotary * cos[:, None] + rotated * sin[:, None], rest), dim=-1)


class FullAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.head_dim = c.head_dim
        self.groups = c.num_attention_heads // c.num_key_value_heads
        self.q_proj = nn.Linear(c.hidden_size, c.num_attention_heads * c.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.num_attention_heads * c.head_dim, c.hidden_size, bias=False)
        self.q_norm = RMSNorm(c.head_dim, c.rms_norm_eps)
        self.k_norm = RMSNorm(c.head_dim, c.rms_norm_eps)

    def forward(self, x, positions, mask):
        batch, length, _ = x.shape
        shape = (batch, length, -1, self.head_dim)
        q, gate = self.q_proj(x).view(batch, length, -1, self.head_dim * 2).chunk(2, -1)
        q = self.q_norm(q.reshape(shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(shape)).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q, k = (apply_rope(t, *positions) for t in (q, k))
        k, v = (t[:, :, None].expand(batch, t.shape[1], self.groups, length, self.head_dim)
                .reshape(batch, -1, length, self.head_dim) for t in (k, v))
        weights = (q @ k.transpose(2, 3)) * self.head_dim**-0.5 + mask
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
        y = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, -1)
        return self.o_proj(y * gate.reshape(batch, length, -1).sigmoid())


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, c, kind):
        super().__init__()
        self.kind = kind
        if kind == "linear_attention":
            self.linear_attn = GatedDeltaNet(c)
        else:
            self.self_attn = FullAttention(c)
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.mlp = MLP(c)

    def forward(self, x, positions, mask):
        y = self.input_layernorm(x)
        y = self.linear_attn(y) if self.kind == "linear_attention" else self.self_attn(y, positions, mask)
        x = x + y
        return x + self.mlp(self.post_attention_layernorm(x))


class TextBackbone(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(c, kind) for kind in c.layer_types)
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        dim = int(c.head_dim * c.rope_parameters["partial_rotary_factor"])
        # Construct on CPU in FP32 even when parameters are constructed on meta.
        inv = 1.0 / (c.rope_parameters["rope_theta"] ** (torch.arange(0, dim, 2, device="cpu").float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, input_ids):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("Only one nonempty, unpadded text sequence is supported")
        x = self.embed_tokens(input_ids)
        length = input_ids.shape[1]
        # All three MRoPE axes are identical for text. Preserve the FP32 matmul.
        positions = torch.arange(length, device=x.device).view(1, 1, -1).expand(3, 1, -1)
        inv = self.inv_freq[None, None, :, None].float().expand(3, 1, -1, 1)
        freqs = (inv @ positions[:, :, None, :].float()).transpose(2, 3)
        cos, sin = (torch.cat((f[0], f[0]), -1).to(x.dtype) for f in (freqs.cos(), freqs.sin()))
        mask = torch.full((length, length), torch.finfo(x.dtype).min, dtype=x.dtype, device=x.device).triu(1)
        for layer in self.layers:
            x = layer(x, (cos, sin), mask[None, None])
        return self.norm(x)


class EvidenceRoutingLayer(torch.nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(width)
        self.memory_norm = torch.nn.LayerNorm(width)
        self.attention = torch.nn.MultiheadAttention(
            width,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = torch.nn.Dropout(dropout)
        self.feedforward_norm = torch.nn.LayerNorm(width)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(width, feedforward),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(feedforward, width),
            torch.nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_queries = self.query_norm(queries)
        routed, _ = self.attention(
            normalized_queries,
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        queries = queries + self.attention_dropout(routed)
        return queries + self.feedforward(self.feedforward_norm(queries))


class JointSchemaHead(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        width: int,
        routing_layers: int,
        layers: int,
        heads: int,
        feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_norm = torch.nn.LayerNorm(hidden_size)
        self.memory_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.question_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.global_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = torch.nn.Embedding(3, width)
        self.evidence_layers = torch.nn.ModuleList(
            [
                EvidenceRoutingLayer(
                    width=width,
                    heads=heads,
                    feedforward=feedforward,
                    dropout=dropout,
                )
                for _ in range(routing_layers)
            ]
        )
        self.option_summary_norm = torch.nn.LayerNorm(width)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.TransformerDecoderLayer(
                    d_model=width,
                    nhead=heads,
                    dim_feedforward=feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.field_norm = torch.nn.LayerNorm(width)
        self.option_norm = torch.nn.LayerNorm(width)
        self.residual_scorer = torch.nn.Sequential(
            torch.nn.Linear(width * 4, width),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(width, 1),
        )
        self.prior_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.residual_gate = torch.nn.Parameter(torch.zeros(()))

    @staticmethod
    def _mean_span(values: torch.Tensor, span: tuple[int, int]) -> torch.Tensor:
        start, end = span
        return values[start:end].mean(dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        records: list,
        output_embedding_weight: torch.Tensor,
    ) -> list[list[torch.Tensor]]:
        results: list[list[torch.Tensor]] = []
        normalized_hidden = self.hidden_norm(hidden_states)
        for batch_index, record in enumerate(records):
            sequence_length = int(attention_mask[batch_index].sum().item())
            sequence_hidden = normalized_hidden[batch_index, :sequence_length]
            memory = self.memory_projection(sequence_hidden).unsqueeze(0)
            global_vector = sequence_hidden[-1]
            question_vectors = torch.stack(
                [
                    self._mean_span(sequence_hidden, question.question_span)
                    for question in record.questions
                ]
            )
            type_ids = torch.tensor(
                [question.question_type for question in record.questions],
                device=hidden_states.device,
            )
            option_contexts: list[torch.Tensor] = []
            lexical_options: list[torch.Tensor] = []
            option_counts = []
            for question in record.questions:
                context_vectors = torch.stack(
                    [
                        self._mean_span(sequence_hidden, span)
                        for span in question.option_spans
                    ]
                )
                lexical_vectors = []
                for start, end in question.option_spans:
                    token_ids = input_ids[batch_index, start:end]
                    lexical_vectors.append(output_embedding_weight[token_ids].mean(dim=0))
                lexical = torch.stack(lexical_vectors)
                option_contexts.append(context_vectors)
                lexical_options.append(lexical)
                option_counts.append(len(question.option_spans))

            option_queries = []
            for question_index, (context_vectors, lexical) in enumerate(
                zip(option_contexts, lexical_options)
            ):
                option_queries.append(
                    self.option_context_projection(context_vectors)
                    + self.option_lexical_projection(lexical)
                    + self.option_question_projection(
                        question_vectors[question_index]
                    ).unsqueeze(0)
                )
            routed_options = torch.cat(option_queries, dim=0).unsqueeze(0)
            for layer in self.evidence_layers:
                routed_options = layer(routed_options, memory)
            routed_options = routed_options[0]
            split_options = list(torch.split(routed_options, option_counts, dim=0))

            base_fields = self.question_projection(question_vectors)
            option_summaries = []
            for field, options in zip(base_fields, split_options):
                routing_weights = torch.softmax(
                    torch.matmul(options, field) / math.sqrt(options.shape[-1]),
                    dim=0,
                )
                option_summaries.append(
                    torch.sum(routing_weights.unsqueeze(-1) * options, dim=0)
                )
            fields = (
                base_fields
                + self.option_summary_norm(torch.stack(option_summaries))
                + self.global_projection(global_vector).unsqueeze(0)
                + self.type_embedding(type_ids)
            )
            fields = fields.unsqueeze(0)
            for layer in self.layers:
                fields = layer(fields, memory)
            fields = self.field_norm(fields[0])

            record_logits: list[torch.Tensor] = []
            for field, question, lexical, routed in zip(
                fields,
                record.questions,
                lexical_options,
                split_options,
            ):
                anchor = F.normalize(
                    question_vectors[len(record_logits)] + global_vector,
                    dim=-1,
                )
                lexical_anchor = F.normalize(lexical, dim=-1)
                prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
                prior = prior_scale * torch.matmul(lexical_anchor, anchor)
                options = self.option_norm(routed)
                repeated_field = field.unsqueeze(0).expand_as(options)
                cosine = F.cosine_similarity(repeated_field, options, dim=-1)
                features = torch.cat(
                    [
                        repeated_field,
                        options,
                        repeated_field * options,
                        torch.abs(repeated_field - options),
                    ],
                    dim=-1,
                )
                residual = self.residual_scorer(features).squeeze(-1)
                joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
                joint = joint_scale * cosine + residual
                record_logits.append(
                    prior + torch.sigmoid(self.residual_gate) * joint
                )
            results.append(record_logits)
        return results


class ClefTextModel(nn.Module):
    def __init__(self, config, head_config):
        super().__init__()
        self.backbone = TextBackbone(config)
        self.lm_head = nn.Embedding(config.vocab_size, config.hidden_size)
        self.head = JointSchemaHead(**head_config)

    def forward(self, input_ids, record):
        if input_ids.shape != (1, len(record.input_ids)):
            raise ValueError("Expected one unpadded sequence matching the encoded record")
        hidden = self.backbone(input_ids)
        return self.head(hidden, input_ids, torch.ones_like(input_ids), [record], self.lm_head.weight)[0]


def load_model(directory, device, progress=lambda name: None):
    """Load only the pinned BF16 text checkpoint; verify before using its contents."""
    directory = Path(directory)
    release = json.loads(Path(__file__).with_name("release.json").read_text())
    index_name = "model.safetensors.index.json"
    needed = ["config.json", "joint_head_config.json", index_name, "joint_head.safetensors", "tokenizer.json"]
    needed += [name for name in release["files"] if name.startswith("model-")]
    for name in needed:
        if not verify(directory / name, release["files"][name]):
            raise ValueError(f"Pinned release verification failed: {name}")
        progress("verified:" + name)
    config = SimpleNamespace(**json.loads((directory / "config.json").read_text())["text_config"])
    head_config = json.loads((directory / "joint_head_config.json").read_text())
    # The verified config fixes the architecture; there is no general-model API.
    with torch.device("meta"):
        model = ClefTextModel(config, head_config)
    weight_map = json.loads((directory / index_name).read_text())["weight_map"]
    expected = {name: parameter for name, parameter in model.named_parameters() if not name.startswith("head.")}
    selected = {}
    for source in weight_map:
        if source.startswith("model.language_model."):
            selected["backbone." + source.removeprefix("model.language_model.")] = source
        elif source == "lm_head.weight":
            selected[source] = source
        elif not source.startswith("model.visual."):
            raise ValueError(f"Unexpected checkpoint tensor: {source}")
    if selected.keys() != expected.keys():
        raise ValueError(f"Checkpoint keys differ: {selected.keys() ^ expected.keys()}")
    shards = defaultdict(list)
    for name, source in selected.items():
        shards[weight_map[source]].append((name, source))
    for shard, entries in shards.items():
        state = {}
        with safe_open(directory / shard, framework="pt", device="cpu") as tensors:
            for name, source in entries:
                value = tensors.get_tensor(source)
                if value.shape != expected[name].shape or value.dtype != torch.bfloat16:
                    raise ValueError(f"Unexpected shape/dtype for {source}: {value.shape}, {value.dtype}")
                state[name] = value.to(device)
        model.load_state_dict(state, strict=False, assign=True)
        progress("loaded:" + shard)
    head_state = {k: v.to(device=device, dtype=torch.bfloat16)
                  for k, v in load_file(directory / "joint_head.safetensors").items()}
    model.head.load_state_dict(head_state, strict=True, assign=True)
    model.backbone.inv_freq = model.backbone.inv_freq.to(device)
    if any(p.is_meta or p.dtype != torch.bfloat16 or p.device != torch.device(device)
           for p in model.parameters()):
        raise RuntimeError("Incomplete BF16 checkpoint load")
    return model.eval()
