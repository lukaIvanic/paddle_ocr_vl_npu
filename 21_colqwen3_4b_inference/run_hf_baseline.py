#!/usr/bin/env python3
"""Unmodified checkpoint HF model/processor on NPU; no vLLM or torch.compile."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--images', type=Path, nargs='+', default=[
        ROOT / 'crops/crop_01_text_block_en.png',
        ROOT / 'crops/crop_05_table_rwkv_dims.png'])
    p.add_argument('--queries', nargs='+', default=[
        'What does the text say?', 'What dimensions are listed in the table?'])
    p.add_argument('--dtype', choices=['fp16', 'bf16'], default='fp16')
    p.add_argument('--device', default='npu:0')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--hash-weights', action='store_true')
    args = p.parse_args(argv)
    if not args.device.startswith('npu:'):
        p.error('This baseline requires an NPU; no CPU/CUDA fallback.')
    if args.repeats < 1:
        p.error('--repeats must be positive')
    return args


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def model_manifest(path, hash_weights):
    records = {}
    for f in sorted(path.rglob('*')):
        if not f.is_file() or not (f.suffix in {'.json', '.py', '.safetensors'}):
            continue
        if '__pycache__' in f.parts or '.cache' in f.parts:
            continue
        records[str(f.relative_to(path))] = {
            'bytes': f.stat().st_size,
            'sha256': sha256(f) if hash_weights or f.suffix != '.safetensors' else None}
    return records


def phase(name, **kwargs):
    print('HF_BASELINE ' + json.dumps({'phase': name, **kwargs}), flush=True)


def verify_checkpoint_keys(model, path):
    # The checkpoint's from_pretrained override assumes a model return value,
    # so output_loading_info=True breaks it. Inspect tensor headers instead;
    # leave the supplied loader and model code unchanged.
    from safetensors import safe_open
    expected = {}
    mapping = model._checkpoint_conversion_mapping
    for shard in sorted(path.glob('*.safetensors')):
        with safe_open(str(shard), framework='pt', device='cpu') as source:
            for key in source.keys():
                mapped = key
                for pattern, replacement in mapping.items():
                    mapped = re.sub(pattern, replacement, mapped)
                if mapped in expected:
                    raise RuntimeError(f'Duplicate mapped checkpoint key: {mapped}')
                expected[mapped] = tuple(source.get_slice(key).get_shape())
    actual = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    return {'missing_keys': sorted(actual.keys() - expected.keys()),
            'unexpected_keys': sorted(expected.keys() - actual.keys()),
            'mismatched_keys': [k for k in actual.keys() & expected.keys()
                                if actual[k] != expected[k]],
            'checkpoint_tensor_count': len(expected)}


def run(args, result):
    import torch
    import torch_npu  # noqa: F401; explicitly registers the NPU backend
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    if not torch.npu.is_available():
        raise RuntimeError('NPU unavailable; no fallback permitted')
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16}[args.dtype]
    result['device_name'] = torch.npu.get_device_name(args.device)
    phase('model_load_start')
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        str(args.model), trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        str(args.model), trust_remote_code=True, local_files_only=True,
        dtype=dtype, attn_implementation='eager')
    result['loading_info'] = verify_checkpoint_keys(model, args.model)
    if any(result['loading_info'][k] for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys')):
        raise RuntimeError('Checkpoint loading is not exact; inspect loading_info')
    model = model.to(args.device).eval()
    torch.npu.synchronize()
    result['model_load_s'] = time.perf_counter() - started
    result['model_class'] = type(model).__name__
    result['processor_class'] = type(processor).__name__
    result['config'] = model.config.to_dict()
    result['processor_image_config'] = processor.image_processor.to_dict()
    result['parameter_dtypes'] = sorted({str(p.dtype) for p in model.parameters()})
    phase('model_load_finish', seconds=result['model_load_s'])

    def encode(name, inputs):
        phase('encode_start', input=name)
        cpu_inputs = {k: v.detach().cpu() for k, v in inputs.items()}
        batch = {k: v.to(args.device) for k, v in inputs.items()}
        durations = []
        reference = None
        max_repeat_diff = 0.0
        with torch.inference_mode():
            for iteration in range(args.repeats + 1):
                torch.npu.synchronize()
                begin = time.perf_counter()
                output = model(**batch)
                torch.npu.synchronize()
                durations.append(time.perf_counter() - begin)
                actual = output.detach().cpu()
                if reference is None:
                    reference = actual
                else:
                    max_repeat_diff = max(max_repeat_diff, float((reference.float() - actual.float()).abs().max()))
                phase('encode_iteration', input=name, iteration=iteration, seconds=durations[-1])
        if reference.ndim != 3 or reference.shape[:2] != cpu_inputs['input_ids'].shape:
            raise RuntimeError(f'Unexpected embedding shape: {reference.shape}')
        if not bool(torch.isfinite(reference).all()):
            raise RuntimeError(f'Nonfinite embeddings: {name}')
        norms = reference.float().norm(dim=-1)
        active = norms > 0
        if not bool(active.any(dim=1).all()):
            raise RuntimeError(f'Empty embedding item: {name}')
        norm_error = float((norms[active] - 1).abs().max())
        if norm_error > 0.02:
            raise RuntimeError(f'Non-unit embedding rows: {name}: {norm_error}')
        if max_repeat_diff > 0.001:
            raise RuntimeError(f'Unstable repeated embeddings: {name}: {max_repeat_diff}')
        torch.save({'inputs': cpu_inputs, 'embeddings': reference}, args.output_dir / f'{name}.pt')
        metadata = {
            'name': name, 'input_shapes': {k: list(v.shape) for k, v in cpu_inputs.items()},
            'image_grid_thw': cpu_inputs.get('image_grid_thw', torch.empty(0)).tolist(),
            'embedding_shape': list(reference.shape), 'dtype': str(reference.dtype),
            'active_rows_per_item': active.sum(dim=1).tolist(),
            'finite': True, 'max_unit_norm_error': norm_error,
            'repeat_max_abs': max_repeat_diff, 'repeat_exact': max_repeat_diff == 0,
            'first_forward_s': durations[0], 'warm_forward_s': durations[1:],
            'warm_mean_s': sum(durations[1:]) / args.repeats}
        result.setdefault('encodings', []).append(metadata)
        phase('encode_finish', **metadata)
        return reference

    query_embeddings = encode('queries', processor.process_queries(args.queries))
    documents = []
    for i, path in enumerate(args.images):
        with Image.open(path) as original:
            image = original.convert('RGB')
        documents.extend(encode(f'image_{i:02d}', processor.process_images([image])))
    phase('score_start')
    with torch.inference_mode():
        scores = processor.score_multi_vector(list(query_embeddings), documents, device=args.device)
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError('Nonfinite MaxSim scores')
    result['maxsim_scores'] = scores.tolist()
    result['status'] = 'passed'
    phase('complete', scores=result['maxsim_scores'])


def main(argv=None):
    args = parse_args(argv)
    args.model = args.model.resolve()
    args.images = [p.resolve() for p in args.images]
    if not (args.model / 'config.json').is_file():
        raise FileNotFoundError(args.model / 'config.json')
    for path in args.images:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ.setdefault('HF_MODULES_CACHE', str(args.output_dir.resolve() / 'hf_modules'))
    result = {
        'status': 'started', 'experiment': '21_colqwen3_4b_hf_baseline',
        'model': str(args.model), 'queries': args.queries,
        'dtype': args.dtype, 'attention': 'eager', 'execution': 'raw_eager',
        'scope': 'HF embedding smoke; not retrieval accuracy or production throughput',
        'python': sys.version, 'host': platform.node(),
        'physical_npu': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        'logical_device': args.device,
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'model_files': model_manifest(args.model, args.hash_weights),
        'images': [{'path': str(p), 'sha256': sha256(p)} for p in args.images],
        'versions': {}}
    for package in ('torch', 'torch-npu', 'transformers', 'tokenizers', 'Pillow', 'qwen-vl-utils'):
        try:
            result['versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result['versions'][package] = None
    (args.output_dir / 'command.json').write_text(json.dumps(sys.argv, indent=2) + '\n')
    try:
        run(args, result)
    except Exception:
        result['status'] = 'failed'
        result['error'] = traceback.format_exc()
        raise
    finally:
        (args.output_dir / 'result.json').write_text(json.dumps(result, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
