#!/usr/bin/env python3
"""Warm B1 owned ColQwen page forward, with separate eager/TorchAir profiles.

Inputs are already processed and resident on NPU. The measured full forward
includes patch/position preparation, vision, mergers/text preparation, text and
retrieval projection/normalization. Compiled means the two existing static
transformer graphs; preparation/mergers/projection remain eager in both lanes.
First use, compile/cache load, preprocessing, input/output transfers, validation
and serialization are outside clean timings. Profiled times are diagnostic only.
Existing shape checks and metadata synchronization inside preparation are included.
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
import time
import traceback

import torch

from bench_prepared_prefill import compare, load_case
from forward_profile_analysis import distribution, summarize_kernel_csv
from local_modeling_colqwen3 import LocalColQwen3
from optimized_prefill import (Options, OptimizedVisionStage, OptimizedTextStage,
                               configure_compiler, text_args_for_promptfa)
from patch_embedding import LinearPatchEmbed, prepare_linear_patch_inputs
from prepared_prefill import StageCompiler, prepare_text, finish_embeddings
from run_hf_baseline import sha256


def emit(phase, **values):
    print('WARM_FORWARD '+json.dumps(dict(phase=phase, **values)), flush=True)


def measure(fn, steps):
    wall, device = [], []
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    for _ in range(steps):
        torch.npu.synchronize()
        before = time.perf_counter()
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter()-before)*1000)
        device.append(start.elapsed_time(end))
    return dict(wall_ms=distribution(wall), device_interval_ms=distribution(device),
                wall_samples_ms=wall, device_samples_ms=device), output


class Forward:
    def __init__(self, model, batch, patch, vision, text):
        self.model, self.batch, self.patch = model, batch, patch
        self.vision, self.text = vision, text
        self.annotate = False

    def section(self, name):
        return torch.profiler.record_function('colqwen.forward.'+name) if self.annotate else nullcontext()

    def __call__(self):
        with self.section('vision_prepare'):
            prepared = prepare_linear_patch_inputs(self.model, self.batch, self.patch)
        with self.section('vision_transformer'):
            visual = self.vision(*prepared.vision_args[:3])
        with self.section('text_prepare_mergers'):
            tensors = text_args_for_promptfa(prepare_text(self.model, prepared, visual))
        with self.section('text_transformer'):
            hidden = self.text(*tensors)
        with self.section('retrieval_projection'):
            return finish_embeddings(self.model, prepared, hidden)


class PageEndToEnd:
    """Serial file-to-CPU-embedding calls; model/processor/graphs already loaded."""
    def __init__(self, forward, processor, image, device, signature):
        self.forward, self.processor = forward, processor
        self.image, self.device, self.signature = image, device, signature

    def preprocess(self):
        from PIL import Image
        with Image.open(self.image) as original:
            image = original.convert('RGB')
        return self.processor.process_images([image])

    def __call__(self):
        inputs = self.preprocess()
        signature = {k: (tuple(v.shape), str(v.dtype)) for k,v in inputs.items()}
        if signature != self.signature:
            raise RuntimeError('Image preprocessing changed the warmed graph input contract')
        self.forward.batch = {k:v.to(self.device) for k,v in inputs.items()}
        return self.forward().cpu()


def capture(fn, args, metric):
    import torch_npu.profiler as prof
    metrics = dict(pipe=prof.AiCMetrics.PipeUtilization, memory=prof.AiCMetrics.Memory)
    destination = args.output_dir/'profiles'/metric
    destination.mkdir(parents=True, exist_ok=False)
    config = prof._ExperimentalConfig(profiler_level=prof.ProfilerLevel.Level1,
        aic_metrics=metrics[metric], export_type=prof.ExportType.Text)
    timings = []
    fn.annotate = True
    try:
        with prof.profile(activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
            schedule=prof.schedule(wait=0, warmup=0, active=args.profile_steps, repeat=1),
            experimental_config=config, record_shapes=True, profile_memory=False, with_stack=True,
            on_trace_ready=prof.tensorboard_trace_handler(str(destination/'raw'), analyse_flag=True)) as recording:
            for index in range(args.profile_steps):
                torch.npu.synchronize()
                before = time.perf_counter()
                scope = getattr(fn, 'profile_scope', 'forward')
                with torch.profiler.record_function(f'colqwen.{args.execution}.{scope}.step{index}'):
                    output = fn()
                    torch.npu.synchronize()
                timings.append((time.perf_counter()-before)*1000)
                recording.step()
    finally:
        fn.annotate = False
    parser = Path(__file__).resolve().parents[1]/'11_mineru_2_5_pro_inference/parse_npu_profile.py'
    command = [sys.executable, str(parser), '--profile-dir', str(destination/'raw'),
               '--topn', '40', '--out-json', str(destination/'parsed.json'),
               '--out-md', os.devnull, '--skip-trace']
    parsed = subprocess.run(command, check=True, capture_output=True, text=True)
    (destination/'parser.log').write_text(parsed.stdout+parsed.stderr)
    kernels = sorted((destination/'raw').glob('**/ASCEND_PROFILER_OUTPUT/kernel_details.csv'))
    if len(kernels) != 1:
        raise RuntimeError(f'Expected one analyzed kernel trace, got {len(kernels)}')
    accounting = summarize_kernel_csv(kernels[0], args.profile_steps)
    (destination/'kernel_accounting.json').write_text(json.dumps(accounting, indent=2)+'\n')
    return dict(profiled_wall_ms=distribution(timings), diagnostic_only=True,
                parser_command=command, parsed_json=str(destination/'parsed.json'),
                kernel_accounting=accounting), output


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
    model = LocalColQwen3.from_pretrained(args.model, device=args.device)
    batch, hf_anchor = load_case(args.anchor, 0, args.device)
    if 'pixel_values' not in batch:
        raise ValueError('Expected one real saved B1 page anchor')
    reference = model(**batch)
    result['reference_vs_hf'] = compare(reference, hf_anchor)
    if not result['reference_vs_hf']['passed']:
        raise RuntimeError('Owned reference differs from saved HF anchor')
    options = Options()
    result['options'] = asdict(options)
    patch = LinearPatchEmbed(model.visual.patch_embed).eval()
    vision, text = OptimizedVisionStage(model, options).eval(), OptimizedTextStage(model, options).eval()
    eager = Forward(model, batch, patch, vision, text)
    expected = eager()
    result['optimized_eager_vs_reference'] = compare(expected, reference)
    result['optimized_eager_vs_hf'] = compare(expected, hf_anchor)
    fn = eager
    if args.execution == 'torchair':
        compiler = StageCompiler(args.model, args.cache_root, emit)
        configure_compiler(compiler, options)
        compiler.identity['internal_format'] = True
        prepared = prepare_linear_patch_inputs(model, batch, patch)
        vision_args = prepared.vision_args[:3]
        vision_call = compiler.get('optimized_vision', vision, vision_args)
        visual = vision_call(*vision_args)
        text_args = text_args_for_promptfa(prepare_text(model, prepared, visual))
        text_call = compiler.get('optimized_text', text, text_args)
        text_call(*text_args)
        torch.npu.synchronize()
        result['cache_records'] = compiler.records
        fn = Forward(model, batch, patch, vision_call, text_call)
    result['input_shapes'] = {k:list(v.shape) for k,v in batch.items()}
    result['vision_tokens'] = int(batch['image_grid_thw'].prod().cpu())
    result['text_tokens'] = int(batch['input_ids'].shape[1])
    for _ in range(args.warmups):
        fn()
    torch.npu.synchronize()
    result['before_profile'], output = measure(fn, args.repeats)
    result['vs_optimized_eager'] = compare(output, expected)
    if not result['vs_optimized_eager']['passed']:
        raise RuntimeError('Compiled/eager forward parity failed')
    emit('warm_baseline', execution=args.execution, timing=result['before_profile'])
    result['profiles'] = {}
    for metric in args.metrics:
        emit('profile_start', execution=args.execution, metric=metric)
        result['profiles'][metric], replay = capture(fn, args, metric)
        parity = compare(replay, output)
        result['profiles'][metric]['replay_parity'] = parity
        if not parity['passed']:
            raise RuntimeError('Profiling changed the forward output')
        emit('profile_finish', execution=args.execution, metric=metric)
        (args.output_dir/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    result['after_profile'], replay = measure(fn, args.repeats)
    result['final_replay_parity'] = compare(replay, output)
    norms = replay.float().norm(dim=-1)
    active = norms > 0
    result['finite'] = bool(torch.isfinite(replay).all())
    result['max_unit_norm_error'] = float(torch.where(
        active, (norms-1).abs(), torch.zeros_like(norms)).max())
    if not bool(active.any()) or not result['final_replay_parity']['passed'] or not result['finite'] or result['max_unit_norm_error'] > .002:
        raise RuntimeError('Output validity/replay gate failed')
    torch.save(replay.cpu(), args.output_dir/'embeddings.pt')
    if args.e2e_image:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True,
                                                  local_files_only=True)
        signature = {k:(tuple(v.shape),str(v.dtype)) for k,v in batch.items()}
        pipeline = PageEndToEnd(fn, processor, args.e2e_image, args.device, signature)
        fresh = pipeline.preprocess()
        matches = {k: k in fresh and fresh[k].dtype == v.dtype and
                   torch.equal(fresh[k].cpu(),v.cpu()) for k,v in batch.items()}
        if set(fresh) != set(batch) or not all(matches.values()):
            raise RuntimeError(f'Fresh image preprocessing differs from saved benchmark input: {matches}')
        result['e2e'] = dict(
            scope='Serial B1: file read/decode, processor, CPU-to-NPU transfer, complete model '
                  'forward, NPU-to-CPU embedding transfer. Warm filesystem/model/processor/graphs. '
                  'No index insertion, retrieval scoring, or network serving.',
            image=str(args.e2e_image), image_sha256=sha256(args.e2e_image),
            batch_size=1, distinct_pages=1, preprocessing_exact_to_anchor=matches,
            processor_image_config=processor.image_processor.to_dict(), blocks=[])
        for _ in range(args.warmups):
            pipeline()
        for block in range(2):
            timing, embeddings = measure(pipeline, args.repeats)
            parity = compare(embeddings, output)
            if not parity['passed']:
                raise RuntimeError('File-to-embedding output differs from same optimized forward')
            result['e2e']['blocks'].append(dict(timing=timing, parity=parity))
        samples = [v for b in result['e2e']['blocks'] for v in b['timing']['wall_samples_ms']]
        result['e2e']['wall_ms'] = distribution(samples)
        result['e2e']['pages_per_second'] = 1000/result['e2e']['wall_ms']['mean']
        emit('page_e2e_finish', execution=args.execution, wall_ms=result['e2e']['wall_ms'],
             pages_per_second=result['e2e']['pages_per_second'])
    result['status'] = 'completed'
    emit('warm_final', execution=args.execution, timing=result['after_profile'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--anchor', type=Path, required=True)
    parser.add_argument('--execution', choices=('raw_eager','torchair'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, default=Path('.runtime_cache/21_colqwen3/prepared'))
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--profile-steps', type=int, default=3)
    parser.add_argument('--metrics', nargs='+', choices=('pipe','memory'), default=['pipe','memory'])
    parser.add_argument('--skip-profile', action='store_true',
                        help='Measure warmed forward timing and validation without capturing profiler traces')
    parser.add_argument('--e2e-image', type=Path,
                        help='Also time repeated file-to-CPU-embedding calls; processed inputs must exactly match anchor')
    args = parser.parse_args()
    if args.skip_profile:
        args.metrics = []
    if not args.device.startswith('npu:') or min(args.warmups,args.repeats,args.profile_steps) < 1:
        parser.error('NPU and positive warmups/repeats/profile steps required')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = dict(status='started', command=sys.argv, host=platform.node(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), execution=args.execution,
        anchor=str(args.anchor), anchor_sha256=sha256(args.anchor),
        scope=__doc__, warmups=args.warmups, repeats=args.repeats, profile_steps=args.profile_steps,
        profiles_enabled=bool(args.metrics))
    try:
        run(args, result)
    except Exception:
        result.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        (args.output_dir/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        emit('finished', status=result['status'])


if __name__ == '__main__':
    main()
