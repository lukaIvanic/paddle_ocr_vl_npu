#!/usr/bin/env python3
"""Isolated warmed ColQwen text transformer forward on one real B1 page.

The 36 text layers, DeepStack additions and final RMSNorm are timed/profiled,
including the portable baseline's one-time sequence padding and output trim.
Hidden states, cos/sin, causal mask and three DeepStack tensors are prepared once
and resident on NPU. Vision, mergers, token embedding/image insertion, rotary
initialization, retrieval projection, transfers, compile and validation are outside
the measured window. Eager exports a frozen input/reference snapshot; compiled
loads that exact snapshot in a separate process. No generation or KV cache.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import torch

from bench_prepared_prefill import compare, load_case
from local_modeling_colqwen3 import LocalColQwen3
from optimized_prefill import (Options, OptimizedVisionStage, OptimizedTextStage,
                               configure_compiler, text_args_for_promptfa)
from patch_embedding import LinearPatchEmbed, prepare_linear_patch_inputs
from prepared_prefill import StageCompiler, prepare_text, finish_embeddings
from profile_warm_forward import capture, emit, measure
from run_hf_baseline import sha256
from text_forward_variants import VARIANTS, build_text_stage, portable_variant, variant_identity


class TextForward:
    profile_scope = 'text_forward'

    def __init__(self, text, tensors):
        self.text, self.tensors = text, tensors
        self.annotate = False

    def __call__(self):
        context = (torch.profiler.record_function('colqwen.text.transformer')
                   if self.annotate else nullcontext())
        with context:
            return self.text(*self.tensors)


def identity(args, options):
    model = Path(args.model).resolve()
    return dict(model=str(model), config_sha256=sha256(model/'config.json'),
                weights={p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                         for p in sorted(model.glob('*.safetensors'))},
                anchor_sha256=sha256(args.anchor), options=asdict(options),
                source={name: sha256(Path(__file__).with_name(name))
                        for name in ('local_modeling_colqwen3.py', 'optimized_prefill.py',
                                     'prepared_prefill.py', 'patch_embedding.py')})


def validate_snapshot_identity(saved, current, *, reference_only=False):
    # A cross-implementation comparison deliberately changes modeling source
    # and options, but must still use the same checkpoint and captured page.
    keys = ('model', 'config_sha256', 'weights', 'anchor_sha256') if reference_only else tuple(current)
    changed = [key for key in keys if saved.get(key) != current.get(key)]
    if changed:
        raise RuntimeError(f'Frozen text identity mismatch: {changed}')
    return [key for key in current if saved.get(key) != current.get(key)]


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
    result.update(device=torch.npu.get_device_name(), torch=torch.__version__,
                  torch_npu=torch_npu.__version__, internal_format=True)
    if '310P' in result['device'].upper() and not portable_variant(args.variant):
        raise ValueError('Historical native-GQA/rotary/SwiGLU variants are 910B diagnostics; use baseline on 310P')
    options = Options()
    provenance = identity(args, options)
    result['input_identity'] = provenance
    emit('text_model_load_start', execution=args.execution)
    model = LocalColQwen3.from_pretrained(args.model, device=args.device)
    text = build_text_stage(model, options, args.variant).eval()
    result['variant'] = variant_identity(args.variant)
    result['variant_source_sha256'] = sha256(Path(__file__).with_name('text_forward_variants.py'))
    result['runner_source_sha256'] = sha256(Path(__file__))
    input_snapshot = args.frozen_inputs or args.reference_snapshot
    if input_snapshot:
        saved = torch.load(input_snapshot, map_location='cpu', weights_only=True)
        changes = validate_snapshot_identity(saved['identity'], provenance,
                                             reference_only=bool(args.reference_snapshot))
        result['reference_identity'] = saved['identity']
        result['reference_identity_changes'] = changes
        tensors = tuple(t.to(args.device).contiguous() for t in saved['text_inputs'])
        expected = saved['expected_hidden']
        result['setup_validation'] = saved['setup_validation']
        snapshot = input_snapshot
    else:
        batch, hf_anchor = load_case(args.anchor, 0, args.device)
        if 'pixel_values' not in batch:
            raise ValueError('Expected a saved real B1 page anchor')
        reference = model(**batch)
        reference_vs_hf = compare(reference, hf_anchor)
        patch = LinearPatchEmbed(model.visual.patch_embed).eval()
        vision = OptimizedVisionStage(model, options).eval()
        prepared = prepare_linear_patch_inputs(model, batch, patch)
        visual = vision(*prepared.vision_args[:3])
        tensors = text_args_for_promptfa(prepare_text(model, prepared, visual))
        expected = text(*tensors)
        optimized_vs_reference = compare(finish_embeddings(model, prepared, expected), reference)
        result['setup_validation'] = dict(reference_vs_hf=reference_vs_hf,
                                         optimized_vs_reference=optimized_vs_reference)
        if not reference_vs_hf['passed']:
            raise RuntimeError('Owned reference differs from the saved HF anchor')
        # The established optimized path has known differences from the manual
        # full model (PromptFA/manual vision norm). Preserve that diagnostic;
        # this experiment compares the same optimized text module/input in both
        # lanes, rather than claiming new equivalence to the manual full model.
        snapshot = args.output_dir/'text_inputs.pt'
        torch.save(dict(identity=provenance, text_inputs=tuple(t.cpu() for t in tensors),
                        expected_hidden=expected.cpu(), setup_validation=result['setup_validation']), snapshot)
        # No setup tensors remain in the timed callable other than the seven text inputs.
        del reference, batch, hf_anchor, patch, vision, prepared, visual
    result.update(frozen_inputs=str(snapshot), frozen_inputs_sha256=sha256(snapshot),
                  text_tokens=int(tensors[0].shape[1]), batch_size=int(tensors[0].shape[0]),
                  text_layers=len(text.layers),
                  text_input_shapes=[list(t.shape) for t in tensors],
                  text_input_dtypes=[str(t.dtype) for t in tensors])
    result['attention_contract'] = dict(
        layout=result['variant']['layout'], gqa=result['variant']['gqa'],
        real_tokens=result['text_tokens'],
        physical_tokens=((result['text_tokens'] + result['variant']['attention_alignment'] - 1)
                         // result['variant']['attention_alignment'] * result['variant']['attention_alignment']),
        norms=result['variant']['norm'], scope='text_prefill_no_kv_cache')
    if len(tensors) != 7 or result['batch_size'] != 1:
        raise ValueError('Expected seven frozen B1 text tensors')
    candidate_eager = text(*tensors)
    torch.npu.synchronize()
    result['candidate_eager_vs_frozen'] = compare(candidate_eager, expected)
    candidate_expected = candidate_eager.cpu()
    del candidate_eager
    call = text
    if args.execution == 'torchair':
        compiler = StageCompiler(args.model, args.cache_root, emit)
        configure_compiler(compiler, options)
        compiler.identity['internal_format'] = True
        stage = 'optimized_text'
        if args.variant != 'baseline':
            compiler.identity['text_variant'] = result['variant']
            compiler.identity['variant_source_sha256'] = result['variant_source_sha256']
            stage = 'candidate_text'
        call = compiler.get(stage, text, tensors)
        call(*tensors)
        torch.npu.synchronize()
        result['cache_records'] = compiler.records
    fn = TextForward(call, tensors)
    for _ in range(args.warmups):
        fn()
    torch.npu.synchronize()
    result['before_profile'], output = measure(fn, args.repeats)
    result['vs_frozen_eager'] = compare(output, expected)
    result['vs_candidate_eager'] = compare(output, candidate_expected)
    result['adoption_eligible'] = (result['candidate_eager_vs_frozen']['passed'] and
                                   result['vs_frozen_eager']['passed'] and
                                   result['vs_candidate_eager']['passed'])
    if args.variant == 'baseline' and not args.reference_snapshot and not result['vs_frozen_eager']['exact']:
        raise RuntimeError('Isolated text forward differs from frozen eager reference')
    if not result['adoption_eligible'] and not args.diagnostic_parity:
        raise RuntimeError('Text variant failed existing atol/rtol=0.002 numerical gate')
    emit('text_variant_parity', variant=args.variant, adoption_eligible=result['adoption_eligible'],
         vs_frozen=result['vs_frozen_eager'], vs_candidate=result['vs_candidate_eager'])
    emit('text_warm_baseline', execution=args.execution, timing=result['before_profile'])
    result['profiles'] = {}
    for metric in args.metrics:
        emit('text_profile_start', execution=args.execution, metric=metric)
        result['profiles'][metric], replay = capture(fn, args, metric)
        parity = compare(replay, output)
        result['profiles'][metric]['replay_parity'] = parity
        if not parity['passed']:
            raise RuntimeError('Profiling changed text output')
        emit('text_profile_finish', execution=args.execution, metric=metric)
        (args.output_dir/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    result['after_profile'], replay = measure(fn, args.repeats)
    result['final_replay_parity'] = compare(replay, output)
    result['finite'] = bool(torch.isfinite(replay).all())
    if not result['finite'] or not result['final_replay_parity']['passed']:
        raise RuntimeError('Text validity/replay gate failed')
    torch.save(replay.cpu(), args.output_dir/'hidden.pt')
    result['status'] = 'completed'
    emit('text_warm_final', execution=args.execution, timing=result['after_profile'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--anchor', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--frozen-inputs', type=Path)
    inputs.add_argument('--reference-snapshot', type=Path,
                        help='Compare changed source/options against an older frozen reference; checkpoint and anchor must match')
    parser.add_argument('--execution', choices=('raw_eager', 'torchair'), required=True)
    parser.add_argument('--variant', choices=tuple(VARIANTS), default='baseline')
    parser.add_argument('--diagnostic-parity', action='store_true',
                        help='Profile failed candidates for diagnosis; never mark them adoption-eligible')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, default=Path('.runtime_cache/21_colqwen3/prepared'))
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--profile-steps', type=int, default=3)
    parser.add_argument('--metrics', nargs='+', choices=('pipe', 'memory'), default=['pipe', 'memory'])
    args = parser.parse_args()
    if not args.device.startswith('npu:') or min(args.warmups, args.repeats, args.profile_steps) < 1:
        parser.error('NPU and positive warmups/repeats/profile steps required')
    if args.execution == 'torchair' and not (args.frozen_inputs or args.reference_snapshot):
        parser.error('Compiled text must load the eager --frozen-inputs snapshot')
    if args.variant != 'baseline' and not (args.frozen_inputs or args.reference_snapshot):
        parser.error('Text variants must use the existing baseline --frozen-inputs snapshot')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = dict(status='started', command=sys.argv, host=platform.node(),
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), execution=args.execution,
                  scope=__doc__, warmups=args.warmups, repeats=args.repeats, profile_steps=args.profile_steps)
    try:
        run(args, result)
    except Exception:
        result.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        (args.output_dir/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        emit('text_finished', status=result['status'])


if __name__ == '__main__':
    main()
