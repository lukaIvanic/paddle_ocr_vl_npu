"""Tensor-only ColQwen transformer stages with eager preparation and finish.

Exact-shape B1 still images/text only. No bucketing, attention-kernel replacement,
KV cache, or changes to the independently validated local eager reference.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import time
import types

import torch
from torch import nn
import torch.nn.functional as F

from local_modeling_colqwen3 import image_positions, rotate_half


def linear(module, x):
    shape = x.shape[:-1]
    return module(x.reshape(-1, x.shape[-1])).reshape(*shape, -1)


def bmm_attention(q, k, v, scale, mask):
    batch, heads, length, dim = q.shape
    groups = heads // k.shape[1]
    if groups != 1:
        kv_heads = k.shape[1]
        k = k[:, :, None].expand(batch, kv_heads, groups, length, dim).reshape(batch, heads, length, dim)
        v = v[:, :, None].expand(batch, kv_heads, groups, length, dim).reshape(batch, heads, length, dim)
    q = q.reshape(batch*heads, length, dim)
    k = k.reshape(batch*heads, length, dim)
    v = v.reshape(batch*heads, length, dim)
    scores = torch.bmm(q, k.transpose(1, 2)).reshape(batch, heads, length, length) * scale
    scores = scores + mask
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.bmm(probs.reshape(batch*heads, length, length), v)
    return out.reshape(batch, heads, length, dim).transpose(1, 2).contiguous()


class PreparedVisionStage(nn.Module):
    """24 vision blocks -> final raw features and three unmerged DeepStack taps."""
    def __init__(self, model):
        super().__init__()
        self.blocks = model.visual.blocks
        self.tap_indices = model.config.vision_config.deepstack_visual_indexes

    def forward(self, hidden, cos, sin, mask):
        # All shapes are static. B1 means no variable-length chunk/split metadata.
        length = hidden.shape[0]
        cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        taps = []
        for index, block in enumerate(self.blocks):
            attn = block.attn
            normalized = block.norm1(hidden)
            q, k, v = linear(attn.qkv, normalized).reshape(length, 3, attn.heads, -1).permute(1, 0, 2, 3).unbind(0)
            q = (q.float()*cos + rotate_half(q.float())*sin).to(q.dtype)
            k = (k.float()*cos + rotate_half(k.float())*sin).to(k.dtype)
            q, k, v = [a.transpose(0, 1).unsqueeze(0).contiguous() for a in (q, k, v)]
            out = bmm_attention(q, k, v, attn.scale, mask).reshape(length, -1)
            hidden = hidden + linear(attn.proj, out)
            mlp = block.mlp
            hidden = hidden + linear(mlp.linear_fc2, F.gelu(linear(mlp.linear_fc1, block.norm2(hidden)), approximate='tanh'))
            if index in self.tap_indices:
                taps.append(hidden)
        return hidden, taps[0], taps[1], taps[2]


class PreparedTextStage(nn.Module):
    """36 text layers + final norm; fixed dense DeepStack injection, no KV writes."""
    def __init__(self, model):
        super().__init__()
        self.layers = model.language_model.layers
        self.norm = model.language_model.norm

    def forward(self, hidden, cos, sin, mask, deep0, deep1, deep2):
        deep = (deep0, deep1, deep2)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        for index, layer in enumerate(self.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(hidden)
            shape = (*x.shape[:-1], -1, attn.c.head_dim)
            q = attn.q_norm(linear(attn.q_proj, x).view(shape)).transpose(1, 2)
            k = attn.k_norm(linear(attn.k_proj, x).view(shape)).transpose(1, 2)
            v = linear(attn.v_proj, x).view(shape).transpose(1, 2)
            q, k = q*cos + rotate_half(q)*sin, k*cos + rotate_half(k)*sin
            out = bmm_attention(q, k, v, attn.c.head_dim**-0.5, mask)
            hidden = hidden + linear(attn.o_proj, out.reshape(*x.shape[:-1], -1))
            x = layer.post_attention_layernorm(hidden)
            mlp = layer.mlp
            hidden = hidden + linear(mlp.down_proj, F.silu(linear(mlp.gate_proj, x))*linear(mlp.up_proj, x))
            if index < 3:
                hidden = hidden + deep[index]
        return self.norm(hidden)


@dataclass
class PreparedInputs:
    inputs: dict
    grids: list
    positions: torch.Tensor
    vision_args: tuple | None


def prepare_inputs(model, inputs):
    ids, valid = inputs['input_ids'], inputs['attention_mask']
    if ids.ndim != 2 or ids.shape[0] != 1 or valid.shape != ids.shape:
        raise ValueError('Initial prepared path requires B1 input IDs and matching 2D mask')
    if not bool((valid == 1).all()):
        raise ValueError('Initial exact-shape path requires unpadded input; trim before preparing')
    if bool((ids == model.config.video_token_id).any()):
        raise ValueError('Video input is unsupported')
    pixels, grid = inputs.get('pixel_values'), inputs.get('image_grid_thw')
    grids, vision_args = [], None
    if pixels is not None:
        if grid is None or grid.shape != (1, 3) or pixels.ndim != 3 or pixels.shape[0] != 1:
            raise ValueError('Expected exactly one still-image grid and padded processor pixel sequence')
        grids = grid.cpu().tolist()
        t, h, w = grids[0]
        merge = model.config.vision_config.spatial_merge_size
        if t != 1 or h <= 0 or w <= 0 or h % merge or w % merge or h*w > pixels.shape[1]:
            raise ValueError('Invalid still-image grid')
        if int((ids == model.config.image_token_id).sum()) != h*w // merge**2:
            raise ValueError('Placeholder count does not match image grid')
        hidden = model.visual.patch_embed(pixels[0, :h*w].contiguous())
        absolute, cos, sin = model.visual.position_embeddings(grids)
        hidden = hidden + absolute
        mask = hidden.new_zeros((1, 1, h*w, h*w))
        vision_args = tuple(a.contiguous() for a in (hidden, cos, sin, mask))
    elif grid is not None or bool((ids == model.config.image_token_id).any()):
        raise ValueError('Image placeholders/grids require pixels')
    positions = image_positions(ids, valid, grids, model.config)
    return PreparedInputs(inputs, grids, positions, vision_args)


def prepare_text(model, prepared, vision_outputs):
    ids, valid = prepared.inputs['input_ids'], prepared.inputs['attention_mask']
    hidden = model.language_model.embed_tokens(ids)
    deep_dense = [torch.zeros_like(hidden) for _ in range(3)]
    if prepared.vision_args is not None:
        image_mask = ids == model.config.image_token_id
        features = model.visual.merger(vision_outputs[0])
        hidden = hidden.masked_scatter(image_mask.unsqueeze(-1).expand_as(hidden), features.to(hidden.dtype))
        for index in range(3):
            features = model.visual.deepstack_merger_list[index](vision_outputs[index+1])
            deep_dense[index][image_mask] = features.to(hidden.dtype)
    length = hidden.shape[1]
    seq = torch.arange(length, device=hidden.device)
    allowed = (seq[:, None] >= seq[None, :])[None, None] & valid[:, None, None, :].bool()
    mask = torch.where(allowed, 0.0, torch.finfo(hidden.dtype).min).to(hidden.dtype)
    c = model.config.text_config
    # Same CPU FP32 initialization as the validated reference. Never move pow to NPU.
    inv = (1.0 / (c.rope_theta ** (torch.arange(0, c.head_dim, 2, dtype=torch.float32) / c.head_dim))).to(hidden.device)
    inv = inv[None, None, :, None].expand(3, 1, -1, 1)
    freqs = (inv.float() @ prepared.positions[:, :, None, :].float()).transpose(2, 3)
    temporal = freqs[0].clone()
    for axis in (1, 2):
        temporal[..., axis:model.config.mrope_section[axis]*3:3] = freqs[axis, ..., axis:model.config.mrope_section[axis]*3:3]
    emb = torch.cat((temporal, temporal), dim=-1)
    cos, sin = emb.cos().to(hidden.dtype), emb.sin().to(hidden.dtype)
    return tuple(a.contiguous() for a in (hidden, cos, sin, mask, *deep_dense))


def finish_embeddings(model, prepared, hidden):
    projected = model.custom_text_proj(hidden)
    projected = projected / projected.norm(dim=-1, keepdim=True)
    projected = projected * prepared.inputs['attention_mask'].unsqueeze(-1)
    if prepared.vision_args is not None and model.config.mask_non_image_embeddings:
        projected = projected * (prepared.inputs['input_ids'] == model.config.image_token_id).unsqueeze(-1)
    return projected


def unique_forward(module, name):
    # Dynamo cache identity is a code object, not merely a bound module instance.
    original = module.forward.__func__
    fn = types.FunctionType(original.__code__.replace(co_name=name), original.__globals__,
                            name, original.__defaults__, original.__closure__)
    fn.__kwdefaults__ = original.__kwdefaults__
    return types.MethodType(fn, module)


class StageCompiler:
    """Persistent, signature-keyed fullgraph wrappers. No eager fallback."""
    def __init__(self, model_dir, cache_root, emit):
        import torch_npu
        try:
            import torchair
        except ImportError:
            from torch_npu.dynamo import torchair
        if not hasattr(torchair, 'inference'):
            torchair.inference = importlib.import_module(f'{torchair.__name__}.inference')
        self.torchair, self.cache_root, self.emit = torchair, Path(cache_root), emit
        self.compiled, self.records = {}, []
        directory = Path(model_dir).resolve()
        self.identity = {
            'model_dir': str(directory),
            'model_config': hashlib.sha256((directory/'config.json').read_bytes()).hexdigest(),
            'weight_files': {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(directory.glob('*.safetensors'))},
            'source': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ('prepared_prefill.py', 'local_modeling_colqwen3.py', 'config.py')},
            'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
            'torchair': getattr(torchair, '__version__', 'unknown'),
            'device': torch.npu.get_device_name(),
            'contract': 'b1_exact_manual_bmm_fp32softmax_native_weights_v1',
        }

    def get(self, stage, module, args):
        signature = {'stage': stage, 'inputs': [{'shape': list(a.shape), 'stride': list(a.stride()),
                       'dtype': str(a.dtype)} for a in args], **self.identity}
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
        if digest in self.compiled:
            return self.compiled[digest]
        path = self.cache_root/f'{stage}_{digest}'
        warm = path.exists() and any(path.rglob('*.om'))
        path.mkdir(parents=True, exist_ok=True)
        (path/'signature.json').write_text(json.dumps(signature, indent=2)+'\n')
        self.emit('cache_wrapper_start', stage=stage, path=str(path), om_present_before=warm)
        start = time.perf_counter()
        call = self.torchair.inference.cache_compile(unique_forward(module, f'colqwen_{stage}_{digest}'),
                    config=self.torchair.CompilerConfig(), dynamic=False, fullgraph=True,
                    cache_dir=str(path), ge_cache=True)
        record = {'stage': stage, 'path': str(path), 'om_present_before': warm,
                  'signature': signature, 'wrapper_s': time.perf_counter()-start}
        self.records.append(record)
        self.compiled[digest] = call
        self.emit('cache_wrapper_finish', stage=stage, seconds=record['wrapper_s'])
        return call
