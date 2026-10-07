#!/usr/bin/env python3
"""Locate portable text eager/GE divergence using the first real text layer."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from bench_prepared_prefill import compare
from local_modeling_colqwen3 import LocalColQwen3, rotate_half
from optimized_prefill import (Options, OptimizedTextStage, prepare_310p_text_inputs,
                               prompt_attention, configure_compiler)
from prepared_prefill import StageCompiler
from profile_warm_forward import emit
from profile_warm_text import identity, validate_snapshot_identity


class Attention(nn.Module):
    def __init__(self, layout, gqa):
        super().__init__()
        self.layout, self.gqa = layout, gqa

    def forward(self, q, k, v, mask):
        return prompt_attention(q, k, v, q.shape[-1]**-.5, mask,
                                self.gqa, layout=self.layout)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--anchor', type=Path, required=True)
    p.add_argument('--reference-snapshot', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True)
    args = p.parse_args()
    import torch_npu
    torch.npu.set_device('npu:0')
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = True
    torch.npu.matmul.allow_hf32 = False
    torch.set_num_threads(4)
    options = Options()
    saved = torch.load(args.reference_snapshot, map_location='cpu', weights_only=True)
    validate_snapshot_identity(saved['identity'], identity(args, options), reference_only=True)
    tensors = tuple(t.to('npu:0').contiguous() for t in saved['text_inputs'])
    hidden, cos, sin, mask, *_ = prepare_310p_text_inputs(*tensors)
    layer = OptimizedTextStage(LocalColQwen3.from_pretrained(args.model, device='npu:0'), options).layers[0]
    cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    x = layer.norm1(hidden)
    q, k, v = layer.qkv(x).split(layer.qkv.sizes, -1)
    shape = (*x.shape[:-1], -1, layer.dim)
    q, k = layer.q_norm(q.reshape(shape)), layer.k_norm(k.reshape(shape))
    v = v.reshape(shape)
    q, k = q*cos+rotate_half(q)*sin, k*cos+rotate_half(k)*sin
    compiler = StageCompiler(args.model, args.cache_root, emit)
    configure_compiler(compiler, options)
    compiler.identity['probe_source'] = Path(__file__).read_text()
    compiler.identity['internal_format'] = True
    for layout in ('BSND', 'BNSD'):
        inputs = (q, k, v) if layout == 'BSND' else tuple(a.transpose(1,2).contiguous() for a in (q,k,v))
        for gqa in ('repeat', 'native'):
            module = Attention(layout, gqa)
            call_args = (*inputs, mask)
            expected = module(*call_args)
            compiled = compiler.get(f'attention_{layout}_{gqa}', module, call_args)
            actual = compiled(*call_args)
            torch.npu.synchronize()
            print('ATTENTION_PARITY', layout, gqa, json.dumps(compare(actual, expected)), flush=True)
    call_args = (hidden, cos, sin, mask)
    expected = layer(*call_args)
    compiled = compiler.get('first_text_block', layer, call_args)
    actual = compiled(*call_args)
    torch.npu.synchronize()
    print('FIRST_BLOCK_PARITY', json.dumps(compare(actual, expected)), flush=True)


if __name__ == '__main__':
    main()
