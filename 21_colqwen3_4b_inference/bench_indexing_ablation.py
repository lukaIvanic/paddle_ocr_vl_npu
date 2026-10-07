"""910B-only indexing ablation using identical inputs and shared compiled graphs.

Replays the exact pre-fix preparation function from a pinned Git revision and
applies one checked source replacement per candidate. Historical unsafe indexing
is diagnostic only: never run this harness on 310P. No production defaults change.
Timings cover the complete warmed forward from NPU inputs to NPU embeddings;
loading, compilation, transfers, validation and benchmark checks are excluded.
"""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import textwrap
import types

import torch
import local_modeling_colqwen3 as local
import prepared_prefill as prepared
import profile_warm_forward as forward
from forward_profile_analysis import distribution
from optimized_prefill import (Options, OptimizedVisionStage, OptimizedTextStage,
                               configure_compiler, text_args_for_promptfa)
from patch_embedding import LinearPatchEmbed, prepare_linear_patch_inputs

OLD = '83bd2640931c98a5051fa17c97606bc7cc588a1c'
ROTARY = '''    temporal = freqs[0].clone()
    for axis in (1, 2):
        temporal[..., axis:model.config.mrope_section[axis]*3:3] = freqs[axis, ..., axis:model.config.mrope_section[axis]*3:3]'''
MASK = '''    length = hidden.shape[1]
    seq = torch.arange(length, device=hidden.device)
    allowed = (seq[:, None] >= seq[None, :])[None, None] & valid[:, None, None, :].bool()
    mask = torch.where(allowed, 0.0, torch.finfo(hidden.dtype).min).to(hidden.dtype)'''


def historical_function(filename, name, class_name=None):
    source = subprocess.check_output(['git', 'show', f'{OLD}:21_colqwen3_4b_inference/{filename}'], text=True)
    nodes = ast.parse(source).body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent(ast.get_source_segment(source, node))


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Historical source contract changed: {old}')
    return source.replace(old, new)


def function(source, name, namespace):
    scope = dict(namespace)
    exec(compile(source, '<indexing_ablation>', 'exec'), scope)
    return scope[name]


def variants():
    old = historical_function('prepared_prefill.py', 'prepare_text')
    fixed = replace_once(old, 'deep_dense[index][image_mask] = features.to(hidden.dtype)',
        'deep_dense[index] = deep_dense[index].masked_scatter(\n'
        '                image_mask.unsqueeze(-1).expand_as(hidden), features.to(hidden.dtype))')
    current = textwrap.dedent(inspect.getsource(prepared.prepare_text))
    rotary_call = '    temporal = interleave_mrope(freqs, model.config.mrope_section)'
    sources = dict(legacy=old, fix_only=fixed,
        fix_rotary=replace_once(fixed, ROTARY, rotary_call),
        fix_lookup=fixed,
        fix_mask=replace_once(fixed, MASK, '    mask = causal_attention_bias(valid, hidden.dtype)'),
        all_current=current,
        all_except_rotary=replace_once(current, rotary_call, ROTARY))
    lookup = historical_function('local_modeling_colqwen3.py', 'position_embeddings', 'VisionModel')
    old_lookup = function(lookup, 'position_embeddings', vars(local))
    return {name: dict(prepare=function(source, 'prepare_text', vars(prepared)),
                      lookup=local.VisionModel.position_embeddings if name in
                      ('fix_lookup', 'all_current', 'all_except_rotary') else old_lookup,
                      source_sha256=hashlib.sha256(source.encode()).hexdigest())
            for name, source in sources.items()}


@torch.inference_mode()
def run(args, result):
    import torch_npu
    from transformers.feature_extraction_utils import BatchFeature
    torch.serialization.add_safe_globals([BatchFeature])
    torch.npu.set_device('npu:0')
    if '910B' not in torch.npu.get_device_name() or torch.npu.device_count() != 1:
        raise RuntimeError('Historical indexing probes require exactly one visible 910B')
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = True
    torch.npu.matmul.allow_hf32 = False
    torch.set_num_threads(4)
    result.update(device=torch.npu.get_device_name(), torch=torch.__version__,
                  torch_npu=torch_npu.__version__, historical_revision=OLD)
    model = local.LocalColQwen3.from_pretrained(args.model, device='npu:0')
    patch = LinearPatchEmbed(model.visual.patch_embed).eval()
    options = Options()
    vision = OptimizedVisionStage(model, options).eval()
    text = OptimizedTextStage(model, options).eval()
    compiler = prepared.StageCompiler(args.model, args.cache_root, forward.emit)
    configure_compiler(compiler, options)
    compiler.identity['internal_format'] = True
    choices = variants()
    result['variant_sources'] = {k:v['source_sha256'] for k,v in choices.items()}
    original_prepare = forward.prepare_text
    original_lookup = model.visual.position_embeddings
    result['cases'] = []
    try:
        for anchor in args.anchors:
            saved = torch.load(anchor, map_location='cpu', weights_only=True)
            batch = {k:v.to('npu:0') for k,v in saved['inputs'].items()}
            expected = saved['eager'].to('npu:0')
            inputs = prepare_linear_patch_inputs(model, batch, patch)
            vc = compiler.get('optimized_vision', vision, inputs.vision_args[:3])
            visual = vc(*inputs.vision_args[:3])
            tensors = text_args_for_promptfa(prepared.prepare_text(model, inputs, visual))
            tc = compiler.get('optimized_text', text, tensors)
            tc(*tensors)
            fn = forward.Forward(model, batch, patch, vc, tc)
            row = dict(anchor=str(anchor), blocks=[], parity={}, summary={})
            result['cases'].append(row)
            samples = {k:[] for k in choices}
            devices = {k:[] for k in choices}
            def activate(name):
                forward.prepare_text = choices[name]['prepare']
                model.visual.position_embeddings = types.MethodType(choices[name]['lookup'], model.visual)
            for name in choices:
                activate(name)
                for _ in range(args.warmups):
                    output = fn()
                torch.npu.synchronize()
                row['parity'][name] = dict(exact=torch.equal(output, expected),
                    max_abs=float((output.float()-expected.float()).abs().max()))
                if not row['parity'][name]['exact']:
                    raise RuntimeError(f'{name} changed embeddings')
            names = list(choices)
            for cycle in range(args.cycles):
                shift = cycle % len(names)
                order = names[shift:]+names[:shift]
                for name in order+order[::-1]:
                    activate(name)
                    timing, output = forward.measure(fn, args.repeats)
                    samples[name].extend(timing['wall_samples_ms'])
                    devices[name].extend(timing['device_samples_ms'])
                    row['blocks'].append(dict(cycle=cycle, variant=name, **timing))
                    if not torch.equal(output, expected):
                        raise RuntimeError(f'{name} changed timed output')
                    forward.emit('indexing_ablation_block', case=anchor.name, cycle=cycle,
                                 variant=name, wall_ms=timing['wall_ms'])
            for name in choices:
                row['summary'][name] = dict(wall_ms=distribution(samples[name]),
                    device_ms=distribution(devices[name]), pages_per_second=1000/distribution(samples[name])['mean'])
            forward.emit('indexing_ablation_case', case=anchor.name, summary=row['summary'])
            forward.prepare_text = original_prepare
            model.visual.position_embeddings = original_lookup
            args.output.write_text(json.dumps(result, indent=2)+'\n')
    finally:
        forward.prepare_text = original_prepare
        model.visual.position_embeddings = original_lookup
    result['status'] = 'passed'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--anchors', type=Path, nargs='+', required=True)
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmups', type=int, default=5)
    p.add_argument('--cycles', type=int, default=3)
    p.add_argument('--repeats', type=int, default=10)
    args = p.parse_args()
    if min(args.warmups,args.cycles,args.repeats) < 1 or args.output.exists():
        p.error('Positive counts and a new output file required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status='started', scope=__doc__, physical_npu=os.getenv('ASCEND_RT_VISIBLE_DEVICES'),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    try:
        run(args,result)
    finally:
        args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
