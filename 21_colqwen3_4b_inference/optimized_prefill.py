"""Opt-in Ascend prefill candidates; the local/prepared references stay untouched.

Uses the fixed-shape PromptFA contract exercised by the reranker on 310P,
not 910B-only fusion_attention or CPU actual-sequence-length inputs.
"""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from local_modeling_colqwen3 import rotate_half
from prepared_prefill import gelu_tanh


@dataclass(frozen=True)
class Options:
    fused_projections: bool = True
    weight_format: str = 'native'
    gqa: str = 'native'
    vision_norm: str = 'manual_fp32'

    def __post_init__(self):
        if self.weight_format not in ('native', 'fractal_nz'):
            raise ValueError('Unsupported weight format')
        if self.gqa not in ('native', 'repeat'):
            raise ValueError('Unsupported GQA contract')
        if self.vision_norm not in ('module', 'manual_fp32'):
            raise ValueError('Unsupported vision norm')


def format_code(value):
    """torch-npu versions expose integer, enum, or named format values."""
    try:
        return int(value)
    except (TypeError,ValueError):
        names={'ND':2,'FRACTAL_NZ':29}
        name=str(value).split('.')[-1]
        if name not in names:
            raise ValueError(f'Unrecognized NPU format: {value!r}')
        return names[name]


class Linear(nn.Module):
    """Own fused/formatted parameters without modifying source Linear modules."""
    def __init__(self, sources, weight_format):
        super().__init__()
        if not sources or len({m.in_features for m in sources}) != 1:
            raise ValueError('Projection inputs must agree')
        self.sizes = tuple(m.out_features for m in sources)
        weight = sources[0].weight.detach() if len(sources) == 1 else torch.cat(
            [m.weight.detach() for m in sources], dim=0)
        has_bias = [m.bias is not None for m in sources]
        if any(has_bias) and not all(has_bias):
            raise ValueError('Mixed bias projections unsupported')
        bias = None if not any(has_bias) else (
            sources[0].bias.detach() if len(sources) == 1 else torch.cat(
                [m.bias.detach() for m in sources]))
        if weight_format == 'fractal_nz':
            if weight.device.type != 'npu':
                raise ValueError('NZ requires NPU; no format fallback')
            import torch_npu
            weight = torch_npu.npu_format_cast(weight, 29)
            if format_code(torch_npu.get_npu_format(weight)) != 29:
                raise RuntimeError('NZ cast failed; refusing native fallback')
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return F.linear(x.reshape(-1, x.shape[-1]), self.weight, self.bias).reshape(*x.shape[:-1], -1)


def manual_layer_norm(module, hidden):
    x = hidden.float()
    centered = x - x.mean(-1, keepdim=True)
    # Match the established MinerU compile workaround: FP32 statistics, then
    # model-dtype normalization output and separate model-dtype affine ops.
    y = (centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + module.eps)).to(hidden.dtype)
    if module.weight is not None:
        y = y * module.weight
    if module.bias is not None:
        y = y + module.bias
    return y


def _promptfa(q, k, v, **kwargs):
    import torch_npu
    return torch_npu.npu_prompt_flash_attention(q, k, v, **kwargs)


def prompt_attention(q, k, v, scale, mask=None, gqa='native'):
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError('Portable PromptFA path requires FP16 Q/K/V')
    if q.shape[-1] not in (64, 128) or q.shape[2] != k.shape[2] or k.shape != v.shape:
        raise ValueError('Expected square D64/D128 prefill attention')
    if q.shape[1] % k.shape[1]:
        raise ValueError('Invalid GQA head counts')
    if gqa == 'repeat' and q.shape[1] != k.shape[1]:
        b,h,s,d = k.shape
        groups = q.shape[1] // h
        k = k[:, :, None].expand(b,h,groups,s,d).reshape(b,h*groups,s,d)
        v = v[:, :, None].expand(b,h,groups,s,d).reshape(b,h*groups,s,d)
    kwargs = dict(num_heads=q.shape[1], input_layout='BNSD', scale_value=scale,
                  pre_tokens=2147483647, next_tokens=2147483647, sparse_mode=0)
    if k.shape[1] != q.shape[1]:
        kwargs['num_key_value_heads'] = k.shape[1]
    if mask is not None:
        if mask.dtype != torch.bool or mask.shape != (q.shape[0],1,q.shape[2],k.shape[2]):
            raise ValueError('Expected a prepared square bool mask')
        kwargs['atten_mask'] = mask
    out = _promptfa(q.contiguous(), k.contiguous(), v.contiguous(), **kwargs)
    return out.transpose(1,2).contiguous()


class VisionBlock(nn.Module):
    def __init__(self, source, options):
        super().__init__()
        self.norm1, self.norm2 = source.norm1, source.norm2
        self.heads, self.scale = source.attn.heads, source.attn.scale
        self.norm_mode = options.vision_norm
        self.qkv = Linear([source.attn.qkv], options.weight_format)
        self.proj = Linear([source.attn.proj], options.weight_format)
        self.fc1 = Linear([source.mlp.linear_fc1], options.weight_format)
        self.fc2 = Linear([source.mlp.linear_fc2], options.weight_format)

    def norm(self, module, x):
        return manual_layer_norm(module,x) if self.norm_mode == 'manual_fp32' else module(x)

    def forward(self, hidden, cos, sin):
        length = hidden.shape[0]
        q,k,v = self.qkv(self.norm(self.norm1,hidden)).reshape(
            length,3,self.heads,-1).permute(1,0,2,3).unbind(0)
        q = (q.float()*cos + rotate_half(q.float())*sin).to(q.dtype)
        k = (k.float()*cos + rotate_half(k.float())*sin).to(k.dtype)
        q,k,v = [a.transpose(0,1).unsqueeze(0) for a in (q,k,v)]
        # Exact B1 image, no padding: full bidirectional attention without mask.
        out = prompt_attention(q,k,v,self.scale).reshape(length,-1)
        hidden = hidden + self.proj(out)
        return hidden + self.fc2(gelu_tanh(self.fc1(self.norm(self.norm2,hidden))))


class OptimizedVisionStage(nn.Module):
    def __init__(self, model, options):
        super().__init__()
        self.blocks = nn.ModuleList([VisionBlock(b,options) for b in model.visual.blocks])
        self.taps = model.config.vision_config.deepstack_visual_indexes

    def forward(self, hidden, cos, sin):
        cos,sin = cos.unsqueeze(-2).float(),sin.unsqueeze(-2).float()
        taps=[]
        for index,block in enumerate(self.blocks):
            hidden=block(hidden,cos,sin)
            if index in self.taps:
                taps.append(hidden)
        return hidden,taps[0],taps[1],taps[2]


class TextBlock(nn.Module):
    def __init__(self, source, options):
        super().__init__()
        a,m=source.self_attn,source.mlp
        self.norm1,self.norm2=source.input_layernorm,source.post_attention_layernorm
        self.q_norm,self.k_norm=a.q_norm,a.k_norm
        self.dim,self.gqa=a.c.head_dim,options.gqa
        self.fused=options.fused_projections
        if self.fused:
            self.qkv=Linear([a.q_proj,a.k_proj,a.v_proj],options.weight_format)
            self.gate_up=Linear([m.gate_proj,m.up_proj],options.weight_format)
        else:
            self.q,self.k,self.v=[Linear([p],options.weight_format) for p in (a.q_proj,a.k_proj,a.v_proj)]
            self.gate,self.up=[Linear([p],options.weight_format) for p in (m.gate_proj,m.up_proj)]
        self.out=Linear([a.o_proj],options.weight_format)
        self.down=Linear([m.down_proj],options.weight_format)

    def forward(self,hidden,cos,sin,mask):
        x=self.norm1(hidden)
        q,k,v=self.qkv(x).split(self.qkv.sizes,-1) if self.fused else (self.q(x),self.k(x),self.v(x))
        shape=(*x.shape[:-1],-1,self.dim)
        q=self.q_norm(q.reshape(shape)).transpose(1,2)
        k=self.k_norm(k.reshape(shape)).transpose(1,2)
        v=v.reshape(shape).transpose(1,2)
        q,k=q*cos+rotate_half(q)*sin,k*cos+rotate_half(k)*sin
        out=prompt_attention(q,k,v,self.dim**-0.5,mask,self.gqa)
        hidden=hidden+self.out(out.reshape(*x.shape[:-1],-1))
        x=self.norm2(hidden)
        gate,up=self.gate_up(x).split(self.gate_up.sizes,-1) if self.fused else (self.gate(x),self.up(x))
        return hidden+self.down(F.silu(gate)*up)


class OptimizedTextStage(nn.Module):
    def __init__(self,model,options):
        super().__init__()
        self.layers=nn.ModuleList([TextBlock(layer,options) for layer in model.language_model.layers])
        self.norm=model.language_model.norm

    def forward(self,hidden,cos,sin,mask,deep0,deep1,deep2):
        deep=(deep0,deep1,deep2)
        cos,sin=cos.unsqueeze(1),sin.unsqueeze(1)
        for index,layer in enumerate(self.layers):
            hidden=layer(hidden,cos,sin,mask)
            if index<3:
                hidden=hidden+deep[index]
        return self.norm(hidden)


def text_args_for_promptfa(args):
    # Outside the graph. True means blocked; every causal row includes itself.
    values=list(args)
    values[3]=(values[3]<0).contiguous()
    if not bool((~values[3]).any(-1).all()):
        raise ValueError('Fully masked attention rows are forbidden')
    return tuple(values)


def configure_compiler(compiler, options):
    compiler.identity['optimized_options']=asdict(options)
    compiler.identity['optimized_source']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def weight_formats(*modules):
    import torch_npu
    counts={}
    for module in modules:
        for child in module.modules():
            if isinstance(child,Linear):
                code=str(format_code(torch_npu.get_npu_format(child.weight)))
                counts[code]=counts.get(code,0)+1
    return counts
