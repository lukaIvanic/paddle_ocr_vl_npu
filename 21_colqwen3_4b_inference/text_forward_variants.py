"""Isolated layout/rotary/SwiGLU candidates for the frozen text-forward lab.

Norms, checkpoint projections, causal masking and DeepStack stay identical to
OptimizedTextStage. Each native implementation follows an owned donor path.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from local_modeling_colqwen3 import rotate_half
from optimized_prefill import TextBlock, OptimizedTextStage


@dataclass(frozen=True)
class TextVariant:
    layout: str = 'BNSD'
    rotary: str = 'manual'
    swiglu: bool = False
    gqa: str = 'native'
    attention_alignment: int = 1
    norm: str = 'manual'


VARIANTS = {
    'baseline': TextVariant(layout='BSND', gqa='repeat', attention_alignment=128),
    'rms_fp16': TextVariant(layout='BSND', gqa='repeat', attention_alignment=128, norm='native_fp16'),
    'rms_fp16_round': TextVariant(layout='BSND', gqa='repeat', attention_alignment=128, norm='unit_fp16'),
    'rms_fp32_round': TextVariant(layout='BSND', gqa='repeat', attention_alignment=128, norm='unit_fp32'),
    'baseline_910b': TextVariant(),
    'bsnd': TextVariant(layout='BSND'),
    'rotary_bnsd': TextVariant(rotary='rotary_mul'),
    'apply_bnsd': TextVariant(rotary='apply'),
    'apply_bsnd': TextVariant(layout='BSND', rotary='apply'),
    'swiglu': TextVariant(swiglu=True),
    'apply_bsnd_swiglu': TextVariant(layout='BSND', rotary='apply', swiglu=True),
}


class NativeTextNorm(nn.Module):
    """Native norm candidates; unit-gamma modes retain the reference FP16 boundary.

    Atlas inference-series API permits FP16/FP16 and FP32/FP32 input/gamma.
    Both ColQwen widths (128 and 2560) exceed its 32-byte tail-axis minimum.
    Direct native FP16 follows the reranker donor. Unit gamma separates learned
    scaling so normalization is rounded to FP16 before the weight multiply.
    """
    def __init__(self, source, mode):
        super().__init__()
        self.weight, self.eps, self.mode = source.weight, source.eps, mode
        dtype = torch.float32 if mode == 'unit_fp32' else torch.float16
        self.register_buffer('unit_weight', torch.ones_like(self.weight, dtype=dtype))

    def forward(self, x):
        import torch_npu
        if self.mode == 'native_fp16':
            return torch_npu.npu_rms_norm(x, self.weight, self.eps)[0]
        value = x.float() if self.mode == 'unit_fp32' else x
        normalized = torch_npu.npu_rms_norm(value, self.unit_weight, self.eps)[0]
        return self.weight * normalized.to(x.dtype)


class NativeNormTextStage(OptimizedTextStage):
    def __init__(self, model, options, mode):
        super().__init__(model, options)
        for layer in self.layers:
            for name in ('norm1', 'norm2', 'q_norm', 'k_norm'):
                setattr(layer, name, NativeTextNorm(getattr(layer, name), mode))
        self.norm = NativeTextNorm(self.norm, mode)


def portable_variant(name):
    variant = VARIANTS[name]
    return variant.gqa == 'repeat' and variant.attention_alignment == 128


def build_text_stage(model, options, name):
    if name == 'baseline':
        return OptimizedTextStage(model, options)
    if portable_variant(name):
        return NativeNormTextStage(model, options, VARIANTS[name].norm)
    return VariantTextStage(model, options, name)


class VariantTextBlock(TextBlock):
    def __init__(self, source, options, variant):
        super().__init__(source, options)
        if not self.fused:
            raise ValueError('Variant lab preserves packed projections and native GQA')
        if variant.gqa != 'native' or variant.attention_alignment != 1:
            raise ValueError('Portable baseline uses OptimizedTextStage, not the diagnostic variants')
        self.variant = variant

    def forward(self, hidden, cos, sin, mask):
        import torch_npu

        x = self.norm1(hidden)
        q, k, v = self.qkv(x).split(self.qkv.sizes, -1)
        shape = (*x.shape[:-1], -1, self.dim)
        q = self.q_norm(q.reshape(shape))
        k = self.k_norm(k.reshape(shape))
        v = v.reshape(shape)
        # Joint ApplyRotary consumes BSND Q/K directly, before attention layout
        # conversion. ColQwen's existing half-layout MRoPE factors are reused.
        if self.variant.rotary == 'apply':
            q, k = torch_npu.npu_apply_rotary_pos_emb(
                q.contiguous(), k.contiguous(), cos.unsqueeze(2).contiguous(),
                sin.unsqueeze(2).contiguous(), layout='BSND', rotary_mode='half')
        if self.variant.layout == 'BNSD':
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            factors = cos.unsqueeze(1), sin.unsqueeze(1)
        else:
            factors = cos.unsqueeze(2), sin.unsqueeze(2)
        if self.variant.rotary == 'manual':
            c, s = factors
            q, k = q*c + rotate_half(q)*s, k*c + rotate_half(k)*s
        elif self.variant.rotary == 'rotary_mul':
            c, s = (a.contiguous() for a in factors)
            q = torch_npu.npu_rotary_mul(q.contiguous(), c, s, rotary_mode='half')
            k = torch_npu.npu_rotary_mul(k.contiguous(), c, s, rotary_mode='half')
        out = torch_npu.npu_prompt_flash_attention(
            q.contiguous(), k.contiguous(), v.contiguous(), atten_mask=mask,
            num_heads=self.qkv.sizes[0] // self.dim,
            num_key_value_heads=self.qkv.sizes[1] // self.dim,
            input_layout=self.variant.layout, scale_value=self.dim**-0.5,
            pre_tokens=2147483647, next_tokens=2147483647, sparse_mode=0)
        if self.variant.layout == 'BNSD':
            out = out.transpose(1, 2).contiguous()
        hidden = hidden + self.out(out.reshape(*x.shape[:-1], -1))
        x = self.norm2(hidden)
        packed = self.gate_up(x)
        if self.variant.swiglu:
            activated = torch_npu.npu_swiglu(packed, dim=-1)
        else:
            gate, up = packed.split(self.gate_up.sizes, -1)
            activated = F.silu(gate)*up
        return hidden + self.down(activated)


class VariantTextStage(nn.Module):
    def __init__(self, model, options, name):
        super().__init__()
        self.variant = VARIANTS[name]
        self.layers = nn.ModuleList([
            VariantTextBlock(layer, options, self.variant)
            for layer in model.language_model.layers])
        self.norm = model.language_model.norm

    def forward(self, hidden, cos, sin, mask, deep0, deep1, deep2):
        deep = (deep0, deep1, deep2)
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, cos, sin, mask)
            if index < 3:
                hidden = hidden + deep[index]
        return self.norm(hidden)


def variant_identity(name):
    return dict(name=name, **asdict(VARIANTS[name]))
