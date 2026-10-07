#!/usr/bin/env python3
"""Full 32-block vision ablations over tensors from actual images/production.

No production source or defaults are changed. Capture runs are NOT throughput
benchmarks. Replay compares all 32 real blocks; unpad uses eager block dispatch
and includes per-layer CPU metadata handling. No candidate outputs enter OCR.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import types
import warnings

import torch
import torch.nn.functional as F

from local_modeling_mineru import LocalMinerU2_5ForConditionalGeneration
from mineru_prefill_timing import PrefillDeviceTimeline
from vision_diagnostic_config import load_config, config_key, runtime_options
from profile_production_vision_routes import measure
from run_transformers_recognition_smoke import configure_npu, synchronize
from vision_prefill_compile import MinerUVisionPrefillRuntime, StaticMinerUVisionBlocks, _import_torchair

VARIANTS = ('baseline', 'pfa_d128', 'pfa_approx', 'pfa_d128_approx',
            'unpad_d80', 'unpad_d128', 'eager_pfa', 'pfa_nz_weights',
            'eager_pfa_nz_weights', 'unpad_d128_nz_weights',
            'pfa_d128_nz_weights', 'grouped_qkv_nz_weights',
            'grouped_qkv_mlp_fc1_nz_weights',
            'grouped_qkv', 'grouped_qkv_mlp_fc1', 'internal_format_opposite')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def phase(event, **kwargs):
    print('ATTENTION_LAB ' + json.dumps(dict(event=event, **kwargs)), flush=True)


def mask_segments(mask):
    """Validate EXACT contiguous block-diagonal full attention, including filler.

    Run on CPU outside timing. Never infer segments just from token counts, and
    never remove the mask unless every mask entry agrees with the length list.
    """
    if mask.dtype != torch.bool or mask.ndim != 4 or mask.shape[:2] != (1, 1):
        raise ValueError('expected bool [1,1,S,S] mask')
    square = mask[0, 0].cpu()
    size = square.shape[0]
    if square.shape[1] != size or not size:
        raise ValueError('expected nonempty square mask')
    lengths, begin = [], 0
    while begin < size:
        allowed = (~square[begin]).nonzero().flatten()
        if not allowed.numel():
            raise ValueError('fully masked query row')
        end = begin + allowed.numel()
        if not torch.equal(allowed, torch.arange(begin, end)):
            raise ValueError('mask is not contiguous full-attention components')
        if square[begin:end, begin:end].any():
            raise ValueError('mask has an internal forbidden pair')
        if begin and not square[begin:end, :begin].all():
            raise ValueError('mask permits an earlier component')
        if end < size and not square[begin:end, end:].all():
            raise ValueError('mask permits a later component')
        lengths.append(end - begin)
        begin = end
    return lengths


def capture(args):
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    command = shlex.split(args.reference_command.read_text())
    command[0] = sys.executable
    for flag, value in (('--limit', args.limit), ('--warmup-pages', 0),
                        ('--output-dir', root / 'diagnostic_pages_NOT_THROUGHPUT')):
        command[command.index(flag) + 1] = str(value)
    model = Path(command[command.index('--model') + 1])
    manifest = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    command=command, model=str(model),
                    model_hashes={n: sha(model / n) for n in ('config.json', 'model.safetensors')},
                    routes={}, diagnostic_page_throughput_valid=False)
    write_json(root / 'manifest.json', manifest)
    original_measure = PrefillDeviceTimeline.measure
    original_compile = MinerUVisionPrefillRuntime._compiled_for_bucket
    wanted = set(args.routes.split(','))
    current = []

    def instrument(timeline, name, fn, *, tags=None):
        current.append(tags if name == 'vision_transformer_blocks' else None)
        try:
            return original_measure(timeline, name, fn, tags=tags)
        finally:
            current.pop()

    def get_compiled(runtime, bucket):
        compiled = original_compile(runtime, bucket)
        if getattr(compiled, '_attention_capture', False):
            return compiled

        def call(*inputs):
            output = compiled(*inputs)
            tags = current[-1] if current else None
            route = tags.get('route') if tags else None
            if route in wanted and route not in manifest['routes']:
                synchronize()
                cpu_inputs = [v.detach().cpu().clone() for v in inputs]
                segments = mask_segments(cpu_inputs[3])
                payload = dict(inputs=cpu_inputs, expected=output.detach().cpu().clone())
                target = root / (route + '.pt')
                torch.save(payload, target)
                manifest['routes'][route] = dict(tags=tags, bucket=int(bucket),
                    segments=segments, file=target.name, sha256=sha(target),
                    baseline_cache=str(runtime._cache_dir(bucket)))
                write_json(root / 'manifest.json', manifest)
                phase('capture_saved', route=route, segments=segments)
            return output

        call._attention_capture = True
        runtime.compiled[bucket] = call
        return call

    try:
        PrefillDeviceTimeline.measure = instrument
        MinerUVisionPrefillRuntime._compiled_for_bucket = get_compiled
        sys.argv = command[1:]
        import run_official_transformers_omnidocbench as runner
        runner.main()
    finally:
        PrefillDeviceTimeline.measure = original_measure
        MinerUVisionPrefillRuntime._compiled_for_bucket = original_compile
    missing = wanted - manifest['routes'].keys()
    if missing:
        raise RuntimeError(f'production did not cover routes: {sorted(missing)}')
    phase('capture_complete', routes=sorted(manifest['routes']))


def unpad_attention(lengths):
    import torch_npu
    # CPU int32 lengths are required by this stock operator. The tensor is
    # prepared once; the operator's per-layer host consumption stays timed.
    seq_lengths = torch.tensor(lengths, dtype=torch.int32, device='cpu')

    def attention(q, k, v, *, num_heads, scale, atten_mask, sparse_mode):
        def tnd(value):
            return value[0].transpose(0, 1).contiguous()
        query, key, value = tnd(q), tnd(k), tnd(v)
        output = torch.empty_like(query)
        torch_npu._npu_flash_attention_unpad(query=query, key=key, value=value,
            seq_len=seq_lengths, scale_value=float(scale), num_heads=num_heads,
            num_kv_heads=num_heads, out=output)
        return output.transpose(0, 1).unsqueeze(0).contiguous()
    return attention


@torch.inference_mode()
def capture_crops(args):
    """Capture the actual full vision forward; never manufacture hidden states."""
    from PIL import Image
    from transformers import AutoProcessor
    from vision_prefill_compile import select_vision_bucket
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    configure_npu()
    import torch_npu
    cfg = load_config(args.config_json)
    torch.npu.config.allow_internal_format = cfg['allow_internal_format']
    if cfg['approximate_precision']:
        if not 200 <= int(torch_npu.npu.get_soc_version()) <= 205:
            raise RuntimeError('requested production precision mode4 is 310P-only')
        approximate_converter()
    model = LocalMinerU2_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16, device='npu:0').eval()
    model.set_vision_attention_impl(cfg['attention_impl'])
    processor = AutoProcessor.from_pretrained(args.model, use_fast=False, local_files_only=True)
    processor.image_processor.min_pixels = cfg['min_pixels']
    processor.image_processor.max_pixels = cfg['max_pixels']
    size = getattr(processor.image_processor, 'size', None)
    if size is not None:
        size['shortest_edge'], size['longest_edge'] = cfg['min_pixels'], cfg['max_pixels']
    manifest = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        model=str(args.model.resolve()), model_hashes={n:sha(args.model/n) for n in ('config.json','model.safetensors')},
        routes={}, scope='actual crops, patch embedding/positions -> full 32-block compiled vision -> merger',
        diagnostic_page_throughput_valid=False, vision_config=cfg,
        processor=dict(min_pixels=cfg['min_pixels'],max_pixels=cfg['max_pixels']),
        physical_device=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
    runtime = MinerUVisionPrefillRuntime(model.visual, buckets=cfg['buckets'],
        cache_root=args.cache_root / ('config_'+config_key(cfg)),
        model_dir=args.model, device=torch.device('npu:0'), dtype=torch.float16, **runtime_options(cfg))
    original_compile = runtime._compiled_for_bucket
    current = {}

    def get_compiled(bucket):
        compiled = original_compile(bucket)
        def call(*inputs):
            output = compiled(*inputs)
            synchronize()
            route = current['route']
            cpu_inputs = [v.detach().cpu().clone() for v in inputs]
            target = root/(route+'.pt')
            torch.save(dict(inputs=cpu_inputs,expected=output.detach().cpu().clone()), target)
            manifest['routes'][route] = dict(tags=current['tags'],bucket=bucket,
                segments=mask_segments(cpu_inputs[3]),file=target.name,sha256=sha(target),
                baseline_cache=str(runtime._cache_dir(bucket)),image=current['image'],
                image_sha256=current['image_sha256'])
            write_json(root/'manifest.json',manifest)
            return output
        return call

    runtime._compiled_for_bucket = get_compiled
    model.set_vision_prefill_runtime(runtime)
    for index,path in enumerate(args.image):
        with Image.open(path) as image:
            image = image.convert('RGB')
        inp = processor.image_processor(images=[image],return_tensors='pt')
        real = int(inp.pixel_values.shape[0])
        if real*196 > cfg['max_pixels']:
            raise ValueError('processor ignored the actual crop pixel cap')
        bucket = select_vision_bucket(real,cfg['buckets'])
        if bucket is None: raise ValueError('real crop exceeds supported buckets')
        route = f'crop_{index}_bucket_{bucket}'
        current.update(route=route,image=str(path.resolve()),image_sha256=sha(path),
            tags=dict(route=route,real_tokens=real,physical_tokens=bucket,members=1,member_lengths=[real],
                image_grid_thw=inp.image_grid_thw.tolist(),num_heads=16,head_dim=80,dtype='torch.float16'))
        phase('real_crop_capture',**current['tags'])
        pixel = inp.pixel_values.to('npu:0',dtype=torch.float16)
        grid = inp.image_grid_thw.to('npu:0')
        features = model.get_image_features(pixel,grid)
        synchronize()
        manifest['routes'][route]['merged_feature_shape'] = list(features.shape)
        write_json(root/'manifest.json',manifest)
    phase('capture_complete',routes=list(manifest['routes']))


def candidate_forward(module, route, variant, lengths):
    """Reuse the exact production forward bytecode, replacing only attention.

    A private globals dictionary avoids changing the production helper or a
    baseline graph's callable. D128 uses production's existing padding branch.
    """
    source = StaticMinerUVisionBlocks.forward
    namespace = dict(source.__globals__)
    if variant.startswith('unpad_'):
        namespace['vision_prompt_flash_attention_bnsd'] = unpad_attention(lengths)
    name = f'attention_lab_{route}_{variant}'
    fn = types.FunctionType(source.__code__.replace(co_name=name), namespace,
                            name, source.__defaults__, source.__closure__)
    return types.MethodType(fn, module)


def approximate_converter():
    # Reuse the already-owned Paddle lab's narrow process-local GE converter.
    # It has no model loading or NPU side effects merely from module import.
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / '09_persistent_page_engine'))
    from scripts.vision_matmul_lab import _register_promptfa_inner_precise_converter
    _register_promptfa_inner_precise_converter(4)


def differences(reference, actual):
    a, b = reference.float(), actual.float()
    delta = (a - b).abs()
    return dict(exact=bool(torch.equal(reference, actual)),
        max_abs=float(delta.max()), mean_abs=float(delta.mean()),
        relative_l2=float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(a).clamp_min(1e-12)),
        cosine=float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
        nonfinite=int((~torch.isfinite(b)).sum()),
        allclose_atol_005_rtol_005=bool(torch.allclose(a, b, atol=.05, rtol=.05)))


@torch.inference_mode()
def replay(args):
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.capture_dir / 'manifest.json').read_text())
    entry = manifest['routes'][args.route]
    tensor_path = args.capture_dir / entry['file']
    if sha(tensor_path) != entry['sha256']:
        raise ValueError('captured tensor hash mismatch')
    model_path = args.model or Path(manifest['model'])
    for name, expected in manifest['model_hashes'].items():
        if sha(model_path / name) != expected:
            raise ValueError(f'model hash mismatch: {name}')
    bundle = torch.load(tensor_path, map_location='cpu', weights_only=True)
    lengths = mask_segments(bundle['inputs'][3])
    if lengths != entry['segments']:
        raise ValueError('captured mask/segment mismatch')
    import torch_npu
    cfg = load_config(args.config_json, manifest.get('vision_config'))
    capture_cfg = load_config(inherited=manifest.get('vision_config'))
    if args.variant == 'internal_format_opposite':
        cfg['allow_internal_format'] = not cfg['allow_internal_format']
    if args.variant.startswith('grouped_'):
        cfg['projection_impl'] = args.variant.removesuffix('_nz_weights')
    if 'd128' in args.variant:
        cfg['promptfa_pad_head_dim_to'] = 128
    if 'approx' in args.variant:
        cfg['approximate_precision'] = True
    if args.variant.startswith('unpad_'):
        cfg['approximate_precision'] = False
    torch.npu.config.allow_internal_format = cfg['allow_internal_format']
    configure_npu()
    device_name = torch.npu.get_device_name(0)
    result = dict(variant=args.variant, route=args.route, tags=entry['tags'], segments=lengths,
        vision_config=cfg, capture_config=capture_cfg, profile_forwards=3,
        config_provenance='explicit manifest/CLI' if 'vision_config' in manifest or args.config_json else 'legacy 910B defaults; verify historical source before 310P use',
        exploratory_projection=args.variant.startswith('grouped_'),
        device=device_name, soc_version=int(torch_npu.npu.get_soc_version()),
        torch=torch.__version__, torch_npu=torch_npu.__version__,
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_capture_sha256=entry['sha256'], model_hashes=manifest['model_hashes'],
        execution='raw_eager' if args.variant.startswith(('unpad_', 'eager_pfa')) else 'torchair_fullgraph',
        boundary='all 32 production vision blocks; excludes capture/H2D/setup; includes attention layout and CPU metadata handling',
        accuracy_scope='same-input layer0 and propagated full-encoder features; NOT downstream OCR accuracy')
    if cfg['approximate_precision'] and not 200 <= result['soc_version'] <= 205:
        result.update(status='unsupported_on_this_device', reason='innerPrecise=4 is 310P-only')
        write_json(root / 'result.json', result)
        phase('unsupported_on_this_device', **result)
        return
    phase('model_load_start', variant=args.variant)
    model = LocalMinerU2_5ForConditionalGeneration.from_pretrained(model_path, dtype=torch.float16, device='npu:0')
    model.set_vision_attention_impl(cfg['attention_impl'])
    if args.variant.endswith('_nz_weights'):
        if not cfg['allow_internal_format']:
            result.update(status='unsupported_configuration',reason='NZ requested while internal formats disabled; no fallback')
            write_json(root/'result.json',result)
            return
        began = time.perf_counter()
        for block in model.visual.blocks:
            for linear in (block.attn.qkv,block.attn.proj,block.mlp.fc1,block.mlp.fc2):
                native = linear.weight.detach().cpu()
                formatted = torch_npu.npu_format_cast(linear.weight.detach(),29)
                if int(torch_npu.get_npu_format(formatted)) != 29:
                    raise RuntimeError('vision weight NZ descriptor not retained; no fallback')
                if not torch.equal(torch_npu.npu_format_cast(formatted,2).cpu(),native):
                    raise RuntimeError('vision weight values changed during NZ conversion')
                linear.weight = torch.nn.Parameter(formatted,requires_grad=False)
        synchronize()
        result['vision_weight_conversion_s'] = time.perf_counter()-began
    result['vision_weights'] = [dict(layer=index, projection=name,
        shape=list(linear.weight.shape), dtype=str(linear.weight.dtype),
        format=int(torch_npu.get_npu_format(linear.weight)))
        for index, block in enumerate(model.visual.blocks)
        for name, linear in (('qkv', block.attn.qkv), ('out', block.attn.proj),
                             ('fc1', block.mlp.fc1), ('fc2', block.mlp.fc2))]
    if args.variant.startswith('grouped_'):
        # Inspect the existing production helper's exact transformation outside
        # timing. Do not substitute a different GMM layout or prepack its work.
        result['grouped_weight_inputs'] = []
        for index, block in enumerate(model.visual.blocks):
            selected = [('qkv', block.attn.qkv)]
            if cfg['projection_impl'] == 'grouped_qkv_mlp_fc1':
                selected.append(('fc1', block.mlp.fc1))
            for name, linear in selected:
                prepared = linear.weight.transpose(0, 1).contiguous().unsqueeze(0)
                result['grouped_weight_inputs'].append(dict(layer=index, projection=name,
                    parameter_format=int(torch_npu.get_npu_format(linear.weight)),
                    operator_input_format=int(torch_npu.get_npu_format(prepared)),
                    operator_input_shape=list(prepared.shape)))
                del prepared
        synchronize()
        result['grouped_weight_scope'] = 'Original in-forward transpose/contiguous/unsqueeze preserved; profiler determines compiled input formats.'
    inputs = [v.to('npu:0') for v in bundle['inputs']]
    expected = bundle['expected'].to('npu:0')
    module = StaticMinerUVisionBlocks(model.visual, **runtime_options(cfg)).eval()
    fn = candidate_forward(module, args.route, args.variant, lengths)
    # First-block same-input comparison controls accumulated numerical drift.
    full_blocks = module.blocks
    module.blocks = torch.nn.ModuleList([full_blocks[0]])
    layer_candidate = fn(*inputs)
    reference_module = StaticMinerUVisionBlocks(model.visual, **runtime_options(capture_cfg)).eval()
    reference_blocks = reference_module.blocks
    reference_module.blocks = torch.nn.ModuleList([reference_blocks[0]])
    layer_reference = reference_module(*inputs)
    result['first_layer_eager_parity'] = differences(layer_reference, layer_candidate)
    result['first_layer_eager_parity_note'] = 'Approximate variants use mode1 here; mode4 is selected only during GE lowering.'
    module.blocks = full_blocks
    reference_module.blocks = reference_blocks
    if result['execution'] == 'torchair_fullgraph':
        if cfg['approximate_precision']:
            approximate_converter()
        if args.variant == 'baseline':
            if cfg != capture_cfg:
                raise RuntimeError('baseline config differs from capture; capture production config separately first')
            runtime = MinerUVisionPrefillRuntime(model.visual, buckets=(entry['bucket'],),
                cache_root=args.cache_root / ('config_'+config_key(cfg)) if 'vision_config' in manifest else args.cache_root,
                model_dir=model_path, device=torch.device('npu:0'), dtype=torch.float16, **runtime_options(cfg))
            cache = runtime._cache_dir(entry['bucket'])
            if not cache.is_dir() or not any(cache.iterdir()):
                raise RuntimeError(f'baseline cache missing; refusing cold rebuild: {cache}')
            fn = runtime._compiled_for_bucket(entry['bucket'])
        else:
            torchair, CompilerConfig = _import_torchair()
            identity = hashlib.sha256((sha(Path(__file__)) + sha(Path(__file__).with_name('vision_prefill_compile.py'))
                + json.dumps(manifest['model_hashes'], sort_keys=True) + torch.__version__ + torch_npu.__version__
                + device_name + json.dumps(cfg,sort_keys=True)).encode()).hexdigest()[:16]
            cache = args.cache_root / 'attention_ablation' / f'{args.route}_{args.variant}_{identity}'
            cache.mkdir(parents=True, exist_ok=True)
            phase('cache_wrapper_start', cache=str(cache))
            fn = torchair.inference.cache_compile(fn, config=CompilerConfig(), dynamic=False,
                cache_dir=str(cache), ge_cache=True, fullgraph=True)
        result['cache'] = str(cache)
    phase('first_call_start', variant=args.variant, route=args.route)
    start = time.perf_counter()
    candidate = fn(*inputs)
    synchronize()
    result['first_call_s'] = time.perf_counter() - start
    result['full_encoder_parity'] = differences(expected[:, :entry['tags']['real_tokens']], candidate[:, :entry['tags']['real_tokens']])
    write_json(root / 'result.json', result)
    phase('first_call_finish', elapsed_s=result['first_call_s'], parity=result['full_encoder_parity'])
    if result['full_encoder_parity']['nonfinite']:
        raise RuntimeError('nonfinite candidate features')
    if args.variant == 'baseline' and not result['full_encoder_parity']['exact']:
        raise RuntimeError('baseline differs from its own production capture; investigate before ablations')
    for _ in range(2):
        fn(*inputs)
    graphs_before = int(torch._dynamo.utils.counters['stats']['unique_graphs'])
    with warnings.catch_warnings(record=True) as caught:
        timing, output = measure(lambda: fn(*inputs), args.steps)
    result['timing_gate'] = dict(
        new_graphs=int(torch._dynamo.utils.counters['stats']['unique_graphs'])-graphs_before,
        recompile_warnings=sum('recompiled' in str(w.message) for w in caught))
    if any(result['timing_gate'].values()):
        result['status'] = 'invalid_timing_compile_in_measurement'
        write_json(root/'result.json',result)
        raise RuntimeError('compilation occurred inside warm vision timer')
    result['timing'] = timing
    result['repeat_parity'] = differences(candidate, output)
    if result['repeat_parity']['nonfinite'] or not result['repeat_parity']['exact']:
        write_json(root / 'result.json', result)
        raise RuntimeError('same-input candidate replay is nonfinite or nondeterministic')
    result['real_tok_s'] = entry['tags']['real_tokens'] * 1000 / timing['device_ms']['mean']
    result['wall_real_tok_s'] = entry['tags']['real_tokens'] * 1000 / timing['wall_ms']['mean']
    result['physical_tok_s'] = entry['tags']['physical_tokens'] * 1000 / timing['device_ms']['mean']
    result['status'] = 'completed'
    if args.profile:
        from profile_vision_prefill_lab import npu_profiler_config, _run_parser
        import torch_npu.profiler as prof
        phase('profile_start')
        with prof.profile(activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
            schedule=prof.schedule(wait=0, warmup=0, active=3, repeat=1),
            experimental_config=npu_profiler_config('pipe'), record_shapes=True,
            on_trace_ready=prof.tensorboard_trace_handler(str(root / 'profile'), analyse_flag=True)) as recording:
            for _ in range(3):
                fn(*inputs)
                synchronize()
                recording.step()
        result['profile'] = _run_parser(root / 'profile', root, topn=60)
        files = list((root / 'profile').rglob('kernel_details.csv'))
        if len(files) != 1:
            raise RuntimeError('missing or ambiguous kernel CSV for actual weight-format audit')
        rows = list(csv.DictReader(files[0].open()))
        from analyze_vision_diagnostics import bucket
        matmuls = [row for row in rows if bucket(row) == 'matmul']
        formats = collections.defaultdict(collections.Counter)
        for row in matmuls:
            formats[row['Type']][row.get('Input Formats', 'MISSING')] += 1
        result['weight_kernel_audit'] = dict(
            requested_nz=args.variant.endswith('_nz_weights'),
            profiled_forwards=3,
            matmul_kernel_count=len(matmuls),
            input_formats_by_type={key:dict(value) for key,value in formats.items()},
            every_matmul_reports_nz_input=bool(matmuls) and all(
                'FRACTAL_NZ' in row.get('Input Formats','') for row in matmuls),
            interpretation='Execution success alone is not proof of NZ consumption; inspect each projection type, especially GroupedMatmul.')
        phase('profile_finish')
    write_json(root / 'result.json', result)
    phase('complete', variant=args.variant, route=args.route, device_ms=timing['device_ms'], parity=result['full_encoder_parity'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    cap = sub.add_parser('capture')
    cap.add_argument('--reference-command', type=Path, required=True)
    cap.add_argument('--output-dir', type=Path, required=True)
    cap.add_argument('--routes', default='bucket_768,packed_768,bucket_5632')
    cap.add_argument('--limit', type=int, default=16)
    run = sub.add_parser('replay')
    run.add_argument('--capture-dir', type=Path, required=True)
    run.add_argument('--route', required=True)
    run.add_argument('--variant', choices=VARIANTS, required=True)
    run.add_argument('--model', type=Path)
    run.add_argument('--cache-root', type=Path, required=True)
    run.add_argument('--output-dir', type=Path, required=True)
    run.add_argument('--steps', type=int, default=10)
    run.add_argument('--profile', action='store_true')
    run.add_argument('--config-json',type=Path,help='Complete config; baseline requires a matching own capture')
    crops = sub.add_parser('capture-crops')
    crops.add_argument('--model',type=Path,required=True)
    crops.add_argument('--config-json',type=Path,help='Complete production/replay config; recorded in manifest')
    crops.add_argument('--image',type=Path,action='append',required=True)
    crops.add_argument('--cache-root',type=Path,required=True)
    crops.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    if args.mode != 'capture-crops' and (args.limit if args.mode == 'capture' else args.steps) <= 0:
        parser.error('counts must be positive')
    if args.mode == 'capture': capture(args)
    elif args.mode == 'capture-crops': capture_crops(args)
    else: replay(args)


if __name__ == '__main__':
    main()
