# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Licensed under Apache-2.0; see LICENSE.apache-2.0.
"""Text-only Qwen3.5 forward used by Clef-flash. Plain eager PyTorch, no cache.

Adapted from Transformers 5.17.0 modeling_qwen3_5.py (Apache-2.0).
Preserves the reference operation order and FP32 recurrent/norm arithmetic.
See THIRD_PARTY.md. This is a correctness baseline, not an optimized runtime.
"""
import torch
from torch import nn
from torch.nn import functional as F


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
