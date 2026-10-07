#!/usr/bin/env python3
"""Repeat a validated text candidate against its control in alternating order.

Uses existing graph caches and the same frozen B1 inputs. No profiler is active;
compile/cache load, transfers and parity checks are outside all timed blocks.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import torch

from bench_prepared_prefill import compare
from forward_profile_analysis import distribution
from local_modeling_colqwen3 import LocalColQwen3
from optimized_prefill import Options, OptimizedTextStage, configure_compiler
from prepared_prefill import StageCompiler
from profile_warm_forward import emit, measure
from profile_warm_text import TextForward, identity
from run_hf_baseline import sha256
from text_forward_variants import VARIANTS, VariantTextStage, variant_identity


@torch.inference_mode()
def run(args, result):
    import torch_npu
    if not torch.npu.is_available():
        raise RuntimeError('Ascend NPU required; no fallback')
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = True
    torch.npu.matmul.allow_hf32 = False
    torch.set_num_threads(4)
    options = Options()
    saved = torch.load(args.frozen_inputs, map_location='cpu', weights_only=True)
    provenance = identity(args, options)
    if saved['identity'] != provenance:
        raise RuntimeError('Frozen input identity changed')
    tensors = tuple(t.to(args.device).contiguous() for t in saved['text_inputs'])
    if len(tensors) != 7 or tensors[0].shape[0] != 1:
        raise ValueError('Expected seven frozen B1 tensors')
    result.update(device=torch.npu.get_device_name(), torch=torch.__version__,
                  torch_npu=torch_npu.__version__, input_identity=provenance,
                  frozen_inputs_sha256=sha256(args.frozen_inputs),
                  text_input_shapes=[list(t.shape) for t in tensors],
                  variant_source_sha256=sha256(Path(__file__).with_name('text_forward_variants.py')),
                  runner_source_sha256=sha256(Path(__file__)), parity={}, cache_records={})
    model = LocalColQwen3.from_pretrained(args.model, device=args.device)
    modules = {'baseline': OptimizedTextStage(model, options).eval(),
               args.variant: VariantTextStage(model, options, args.variant).eval()}
    calls = {}
    for name, module in modules.items():
        eager = module(*tensors)
        eager_parity = compare(eager, saved['expected_hidden'])
        if not eager_parity['passed']:
            raise RuntimeError(f'{name} eager parity failed')
        compiler = StageCompiler(args.model, args.cache_root/name, emit)
        configure_compiler(compiler, options)
        compiler.identity['internal_format'] = True
        stage = 'optimized_text'
        if name != 'baseline':
            compiler.identity['text_variant'] = variant_identity(name)
            compiler.identity['variant_source_sha256'] = result['variant_source_sha256']
            stage = 'candidate_text'
        call = compiler.get(stage, module, tensors)
        output = call(*tensors)
        torch.npu.synchronize()
        compiled_parity = compare(output, saved['expected_hidden'])
        vs_eager = compare(output, eager)
        result['parity'][name] = dict(eager=eager_parity, compiled=compiled_parity, vs_eager=vs_eager)
        if not compiled_parity['passed'] or not vs_eager['passed']:
            raise RuntimeError(f'{name} compiled parity failed')
        if name == 'baseline' and not compiled_parity['exact']:
            raise RuntimeError('Control must remain bit-exact')
        calls[name] = TextForward(call, tensors)
        result['cache_records'][name] = compiler.records
    for _ in range(args.warmups):
        for fn in calls.values():
            fn()
    torch.npu.synchronize()
    samples = {name: [] for name in calls}
    device_samples = {name: [] for name in calls}
    outputs = {}
    result['blocks'] = []
    for cycle in range(args.cycles):
        order = ('baseline', args.variant, args.variant, 'baseline')
        for name in order:
            timing, outputs[name] = measure(calls[name], args.repeats)
            samples[name].extend(timing['wall_samples_ms'])
            device_samples[name].extend(timing['device_samples_ms'])
            result['blocks'].append(dict(cycle=cycle, variant=name, **timing))
            emit('abba_block', cycle=cycle, variant=name, wall_ms=timing['wall_ms'])
    result['final_parity'] = {name: compare(output, saved['expected_hidden'])
                              for name, output in outputs.items()}
    if not all(p['passed'] for p in result['final_parity'].values()):
        raise RuntimeError('Final replay parity failed')
    result['warm_wall_ms'] = {name: distribution(v) for name,v in samples.items()}
    result['warm_device_interval_ms'] = {name: distribution(v) for name,v in device_samples.items()}
    a,b = (result['warm_wall_ms'][name]['mean'] for name in ('baseline',args.variant))
    result.update(speedup=a/b, latency_reduction_percent=100*(1-b/a), status='completed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--anchor', type=Path, required=True)
    parser.add_argument('--frozen-inputs', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--variant', choices=[name for name in VARIANTS if name != 'baseline'], default='apply_bsnd')
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--cycles', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if not args.device.startswith('npu:') or min(args.warmups,args.cycles,args.repeats) < 1:
        parser.error('NPU and positive warmups/cycles/repeats required')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = dict(status='started', host=platform.node(), command=sys.argv,
                  commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                  physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                  variant=args.variant, warmups=args.warmups, cycles=args.cycles,
                  repeats=args.repeats, scope=__doc__)
    try:
        run(args,result)
    except Exception:
        result.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        (args.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
