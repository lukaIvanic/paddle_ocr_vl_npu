# Qwen3-VL calculations adapted from Transformers 4.57.1 modeling_qwen3_vl.py.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. Apache-2.0.
# https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
"""Owned eager Ops-Colqwen3 image/text forward; no Transformers dependency.

Correctness baseline: no generation, cache, fused attention, compilation or
weight-format conversion. Inputs are the unchanged checkpoint processor tensors.
Only still images (T=1) and text are supported; unsupported inputs fail loudly.
"""
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from config import ColQwenConfig


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def interleave_mrope(freqs, sections):
    """Same strided axis selection as HF, without device indexed writes."""
    columns = torch.arange(freqs.shape[-1], device=freqs.device)
    temporal = freqs[0]
    for axis in (1, 2):
        select = ((columns % 3 == axis) & (columns < sections[axis]*3))
        temporal = torch.where(select.view(1, 1, -1).expand_as(temporal),
                               freqs[axis], temporal)
    return temporal


def dense_image_features(hidden, image_mask, features):
    """Scatter image rows in row-major order; avoid NonZero/IndexPut on 310P."""
    return torch.zeros_like(hidden).masked_scatter(
        image_mask.unsqueeze(-1).expand_as(hidden), features.to(hidden.dtype))


def add_image_features(hidden, image_mask, features):
    dense = dense_image_features(hidden, image_mask, features)
    # Preserve unselected values exactly, including signed zero; no masked read.
    return torch.where(image_mask.unsqueeze(-1).expand_as(hidden), hidden+dense, hidden)


def causal_attention_bias(valid, dtype):
    length = valid.shape[1]
    seq = torch.arange(length, device=valid.device)
    allowed = (seq[:, None] >= seq[None, :])[None, None] & valid[:, None, None, :].bool()
    return torch.where(allowed, torch.zeros_like(allowed, dtype=dtype),
                       torch.full_like(allowed, torch.finfo(dtype).min, dtype=dtype))


def attention(q, k, v, scale, mask=None):
    groups = q.shape[1] // k.shape[1]
    if groups != 1:
        b, h, s, d = k.shape
        k = k[:, :, None].expand(b, h, groups, s, d).reshape(b, h * groups, s, d)
        v = v[:, :, None].expand(b, h, groups, s, d).reshape(b, h * groups, s, d)
    weights = torch.matmul(q, k.transpose(2, 3)) * scale
    if mask is not None:
        weights = weights + mask
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(weights, v).transpose(1, 2).contiguous()


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * y.to(x.dtype)


class VisionPatchEmbed(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        kernel = (c.temporal_patch_size, c.patch_size, c.patch_size)
        self.proj = nn.Conv3d(c.in_channels, c.hidden_size, kernel, stride=kernel, bias=True)

    def forward(self, x):
        c = self.c
        x = x.view(-1, c.in_channels, c.temporal_patch_size, c.patch_size, c.patch_size)
        return self.proj(x.to(self.proj.weight.dtype)).view(-1, c.hidden_size)


class VisionMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.linear_fc1 = nn.Linear(c.hidden_size, c.intermediate_size)
        self.linear_fc2 = nn.Linear(c.intermediate_size, c.hidden_size)

    def forward(self, x):
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate='tanh'))


class VisionAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.num_heads
        self.scale = (c.hidden_size // c.num_heads) ** -0.5
        self.qkv = nn.Linear(c.hidden_size, c.hidden_size * 3)
        self.proj = nn.Linear(c.hidden_size, c.hidden_size)

    def forward(self, x, lengths, cos, sin):
        n = x.shape[0]
        q, k, v = self.qkv(x).reshape(n, 3, self.heads, -1).permute(1, 0, 2, 3).unbind(0)
        cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        q = (q.float() * cos + rotate_half(q.float()) * sin).to(q.dtype)
        k = (k.float() * cos + rotate_half(k.float()) * sin).to(k.dtype)
        q, k, v = [a.transpose(0, 1).unsqueeze(0) for a in (q, k, v)]
        chunks = [torch.split(a, lengths, dim=2) for a in (q, k, v)]
        out = torch.cat([attention(a, b, c, self.scale) for a, b, c in zip(*chunks)], dim=1)
        return self.proj(out.reshape(n, -1).contiguous())


class VisionBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(c.hidden_size, eps=1e-6)
        self.attn = VisionAttention(c)
        self.mlp = VisionMLP(c)

    def forward(self, x, lengths, cos, sin):
        x = x + self.attn(self.norm1(x), lengths, cos, sin)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    def __init__(self, c, postshuffle=False):
        super().__init__()
        self.size = c.hidden_size * c.spatial_merge_size ** 2
        self.postshuffle = postshuffle
        self.norm = nn.LayerNorm(self.size if postshuffle else c.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.size, self.size)
        self.linear_fc2 = nn.Linear(self.size, c.out_hidden_size)

    def forward(self, x):
        x = self.norm(x.view(-1, self.size) if self.postshuffle else x).view(-1, self.size)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class VisionModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.patch_embed = VisionPatchEmbed(c)
        self.pos_embed = nn.Embedding(c.num_position_embeddings, c.hidden_size)
        self.blocks = nn.ModuleList([VisionBlock(c) for _ in range(c.depth)])
        self.merger = PatchMerger(c)
        self.deepstack_merger_list = nn.ModuleList([PatchMerger(c, True) for _ in c.deepstack_visual_indexes])

    def position_embeddings(self, grids):
        # Keep the reference's four-term interpolation order and weight dtype.
        side = math.isqrt(self.c.num_position_embeddings)
        indices, weights, coords = [[] for _ in range(4)], [[] for _ in range(4)], []
        m = self.c.spatial_merge_size
        for t, h, w in grids:
            hi, wi = torch.linspace(0, side - 1, h), torch.linspace(0, side - 1, w)
            hf, wf = hi.int(), wi.int()
            hc, wc = (hf + 1).clamp(max=side - 1), (wf + 1).clamp(max=side - 1)
            dh, dw = hi - hf, wi - wf
            ids = [(hf[:, None] * side + wf).flatten(), (hf[:, None] * side + wc).flatten(),
                   (hc[:, None] * side + wf).flatten(), (hc[:, None] * side + wc).flatten()]
            ws = [((1-dh)[:, None] * (1-dw)).flatten(), ((1-dh)[:, None] * dw).flatten(),
                  (dh[:, None] * (1-dw)).flatten(), (dh[:, None] * dw).flatten()]
            for j in range(4):
                indices[j].extend(ids[j].tolist())
                weights[j].extend(ws[j].tolist())
            row = (torch.arange(h // m)[:, None, None, None] * m + torch.arange(m)[None, None, :, None])
            col = (torch.arange(w // m)[None, :, None, None] * m + torch.arange(m)[None, None, None, :])
            coords.append(torch.stack([row.expand(h//m, w//m, m, m).reshape(-1),
                                       col.expand(h//m, w//m, m, m).reshape(-1)], dim=-1).repeat(t, 1))
        device, dtype = self.pos_embed.weight.device, self.pos_embed.weight.dtype
        ids = torch.tensor(indices, device=device, dtype=torch.long)
        ws = torch.tensor(weights, device=device, dtype=dtype)
        terms = self.pos_embed(ids) * ws[:, :, None]
        absolute = terms[0] + terms[1] + terms[2] + terms[3]
        absolute = absolute.split([h*w for _, h, w in grids])
        absolute = torch.cat([a.repeat(t, 1).view(t, h//m, m, w//m, m, -1)
                             .permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
                             for a, (t, h, w) in zip(absolute, grids)])
        dim = (self.c.hidden_size // self.c.num_heads) // 2
        # HF initializes these non-persistent buffers on CPU in FP32. Computing
        # the powers on the NPU instead changes rounding before every RoPE.
        inv = (1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))).to(device)
        freq = torch.outer(torch.arange(max(max(h, w) for _, h, w in grids), device=device, dtype=inv.dtype), inv)
        # Table lookup through Embedding rather than AICPU IndexByTensor.
        rotary = F.embedding(torch.cat(coords).to(device), freq).flatten(1)
        emb = torch.cat((rotary, rotary), dim=-1)
        return absolute, emb.cos(), emb.sin()

    def forward(self, pixels, grids):
        x = self.patch_embed(pixels)
        absolute, cos, sin = self.position_embeddings(grids)
        x = x + absolute
        lengths = [h*w for t, h, w in grids for _ in range(t)]
        deep = []
        for index, block in enumerate(self.blocks):
            x = block(x, lengths, cos, sin)
            if index in self.c.deepstack_visual_indexes:
                deep.append(self.deepstack_merger_list[self.c.deepstack_visual_indexes.index(index)](x))
        return self.merger(x), deep


class TextAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.q_proj = nn.Linear(c.hidden_size, c.num_attention_heads*c.head_dim, bias=c.attention_bias)
        self.k_proj = nn.Linear(c.hidden_size, c.num_key_value_heads*c.head_dim, bias=c.attention_bias)
        self.v_proj = nn.Linear(c.hidden_size, c.num_key_value_heads*c.head_dim, bias=c.attention_bias)
        self.o_proj = nn.Linear(c.num_attention_heads*c.head_dim, c.hidden_size, bias=c.attention_bias)
        self.q_norm = RMSNorm(c.head_dim, c.rms_norm_eps)
        self.k_norm = RMSNorm(c.head_dim, c.rms_norm_eps)

    def forward(self, x, mask, cos, sin):
        shape = (*x.shape[:-1], -1, self.c.head_dim)
        q = self.q_norm(self.q_proj(x).view(shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(shape)).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        q, k = q*cos + rotate_half(q)*sin, k*cos + rotate_half(k)*sin
        out = attention(q, k, v, self.c.head_dim**-0.5, mask)
        return self.o_proj(out.reshape(*x.shape[:-1], -1).contiguous())


class TextMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TextLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = TextAttention(c)
        self.mlp = TextMLP(c)
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)

    def forward(self, x, mask, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), mask, cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class TextModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([TextLayer(c) for _ in range(c.num_hidden_layers)])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)

    def forward(self, x, padding_mask, positions, image_mask, deep, sections):
        mask = causal_attention_bias(padding_mask, x.dtype)
        inv = (1.0 / (self.c.rope_theta ** (torch.arange(0, self.c.head_dim, 2, dtype=torch.float32) / self.c.head_dim))).to(x.device)
        inv = inv[None, None, :, None].expand(3, x.shape[0], -1, 1)
        freqs = (inv.float() @ positions[:, :, None, :].float()).transpose(2, 3)
        temporal = interleave_mrope(freqs, sections)
        emb = torch.cat((temporal, temporal), dim=-1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        for index, layer in enumerate(self.layers):
            x = layer(x, mask, cos, sin)
            if index < len(deep):
                x = add_image_features(x, image_mask, deep[index])
        return self.norm(x)


def image_positions(input_ids, padding_mask, grids, c):
    """HF image-only MRoPE indexing, including left/right padding positions."""
    if not grids:
        pos = padding_mask.long().cumsum(-1) - 1
        return pos.masked_fill(padding_mask == 0, 1).unsqueeze(0).expand(3, -1, -1)
    ids_cpu, valid_cpu = input_ids.cpu(), padding_mask.cpu().bool()
    result = torch.ones((3, *input_ids.shape), dtype=input_ids.dtype)
    grid_index = 0
    for b, row in enumerate(ids_cpu):
        ids = row[valid_cpu[b]].tolist()
        starts = [i+1 for i, token in enumerate(ids[:-1])
                  if token == c.vision_start_token_id and ids[i+1] == c.image_token_id]
        parts, start, next_pos = [], 0, 0
        for end in starts:
            t, h, w = grids[grid_index]
            grid_index += 1
            h, w = h // c.vision_config.spatial_merge_size, w // c.vision_config.spatial_merge_size
            length = end-start
            parts.append(torch.arange(length).view(1, -1).expand(3, -1) + next_pos)
            ti = torch.arange(t).view(-1, 1).expand(-1, h*w).flatten()
            hi = torch.arange(h).view(1, -1, 1).expand(t, -1, w).flatten()
            wi = torch.arange(w).view(1, 1, -1).expand(t, h, -1).flatten()
            vision = torch.stack([ti, hi, wi]) + length + next_pos
            parts.append(vision)
            next_pos = int(vision.max()) + 1
            start = end + t*h*w
            if ids[end:start] != [c.image_token_id] * (t*h*w):
                raise ValueError('Image placeholder run does not match its grid')
        if start < len(ids):
            parts.append(torch.arange(len(ids)-start).view(1, -1).expand(3, -1) + next_pos)
        result[:, b, valid_cpu[b]] = torch.cat(parts, dim=1)
    if grid_index != len(grids):
        raise ValueError('Image grids and placeholder runs disagree')
    return result.to(input_ids.device)


class LocalColQwen3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual = VisionModel(config.vision_config)
        self.language_model = TextModel(config.text_config)
        self.custom_text_proj = nn.Linear(config.text_config.hidden_size, config.dims)

    @classmethod
    def from_pretrained(cls, directory, *, device='npu:0', dtype=torch.float16):
        from safetensors.torch import load_file
        directory = Path(directory)
        config = ColQwenConfig.from_model_dir(directory)
        with torch.device('meta'):
            model = cls(config)
        state = {}
        shards = sorted(directory.glob('*.safetensors'))
        if not shards:
            raise FileNotFoundError('No safetensors checkpoint shards')
        for shard in shards:
            for key, value in load_file(shard).items():
                # Current checkpoint stores visual.*, language_model.* directly.
                if key in state:
                    raise ValueError(f'Duplicate checkpoint tensor: {key}')
                state[key] = value.to(dtype)
        model.load_state_dict(state, strict=True, assign=True)
        return model.to(device).eval()

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None):
        if input_ids.ndim != 2 or bool((input_ids == self.config.video_token_id).any()):
            raise ValueError('Only 2D text/still-image token batches are supported')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape or not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError('Expected a binary 2D attention mask')
        if not bool(attention_mask.bool().any(-1).all()):
            raise ValueError('Empty input rows are unsupported')
        grids, deep = [], []
        image_mask = input_ids == self.config.image_token_id
        x = self.language_model.embed_tokens(input_ids)
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError('Image grid required with pixels')
            grids = image_grid_thw.cpu().tolist()
            m = self.config.vision_config.spatial_merge_size
            if any(t != 1 or h <= 0 or w <= 0 or h % m or w % m for t, h, w in grids):
                raise ValueError('Expected still-image, positive merge-aligned grids')
            if pixel_values.ndim != 3 or len(pixel_values) != len(grids):
                raise ValueError('Expected processor pixels [images, padded_patches, patch_features]')
            if any(t*h*w > pixel_values.shape[1] for t, h, w in grids):
                raise ValueError('Grid exceeds available pixel patches')
            pixels = torch.cat([p[:t*h*w] for p, (t, h, w) in zip(pixel_values, grids)])
            features, deep = self.visual(pixels, grids)
            if int(image_mask.sum()) != len(features):
                raise ValueError('Image token count and merged vision features disagree')
            x = x.masked_scatter(image_mask.unsqueeze(-1).expand_as(x), features.to(x.dtype))
        elif image_grid_thw is not None or bool(image_mask.any()):
            raise ValueError('Image placeholders/grids require pixels')
        positions = image_positions(input_ids, attention_mask, grids, self.config)
        x = self.language_model(x, attention_mask, positions, image_mask, deep, self.config.mrope_section)
        projected = self.custom_text_proj(x)
        projected = projected / projected.norm(dim=-1, keepdim=True)
        projected = projected * attention_mask.unsqueeze(-1)
        if pixel_values is not None and self.config.mask_non_image_embeddings:
            projected = projected * image_mask.unsqueeze(-1)
        return projected
