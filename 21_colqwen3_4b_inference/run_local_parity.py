"""910B/NPU HF-vs-owned-model parity, using identical saved processor inputs."""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import torch

from local_modeling_colqwen3 import LocalColQwen3, image_positions
from run_hf_baseline import ROOT, model_manifest, sha256, verify_checkpoint_keys


def emit(event, **data):
    print('LOCAL_PARITY ' + json.dumps({'phase': event, **data}), flush=True)


def difference(actual, expected, *, atol=0.002, rtol=0.002):
    if actual.shape != expected.shape:
        return {'passed': False, 'actual_shape': list(actual.shape), 'expected_shape': list(expected.shape)}
    a, b = actual.float().flatten(), expected.float().flatten()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    delta = (a-b).abs()
    return {'shape': list(actual.shape), 'finite': finite, 'exact': torch.equal(actual, expected),
            'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()),
            'rmse': float((a-b).square().mean().sqrt()),
            'cosine': float(torch.nn.functional.cosine_similarity(a[None], b[None])),
            'atol': atol, 'rtol': rtol,
            'passed': finite and bool(torch.allclose(a, b, atol=atol, rtol=rtol))}


def trace_modules(model, hf):
    base = 'qwen3vl.' if hf else ''
    selected = ['visual.patch_embed', 'visual.merger', 'language_model.norm', 'custom_text_proj']
    selected += [f'visual.blocks.{i}' for i in (0, 5, 11, 17, 23)]
    selected += [f'visual.deepstack_merger_list.{i}' for i in range(3)]
    selected += [f'language_model.layers.{i}' for i in (0, 1, 2, 17, 35)]
    modules = dict(model.named_modules())
    captured, handles = {}, []
    for name in selected:
        key = name if name == 'custom_text_proj' else base + name
        def hook(module, inputs, output, label=name):
            if isinstance(output, tuple):
                output = output[0]
            # Copy now: DeepStack mutates text-layer output in place afterwards.
            captured[label] = output.detach().cpu().clone()
        handles.append(modules[key].register_forward_hook(hook))
    return captured, handles


def forward(model, inputs, hf):
    traced, handles = trace_modules(model, hf)
    try:
        with torch.inference_mode():
            torch.npu.synchronize()
            start = time.perf_counter()
            out = model(**inputs)
            torch.npu.synchronize()
            elapsed = time.perf_counter() - start
            out = out.detach().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return out, traced, elapsed


def run(args, result):
    import torch_npu  # noqa: F401
    from transformers import AutoModel, AutoProcessor
    from PIL import Image

    if importlib.metadata.version('transformers') != '4.57.1':
        raise RuntimeError('Use the pinned Transformers 4.57.1 baseline environment')
    if not torch.npu.is_available():
        raise RuntimeError('No NPU: no CPU fallback')
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    result['device_name'] = torch.npu.get_device_name(args.device)
    emit('local_model_load_start')
    local = LocalColQwen3.from_pretrained(args.model, device=args.device)
    result['local_tensor_count'] = len(local.state_dict())
    emit('hf_model_load_start')
    hf = AutoModel.from_pretrained(args.model, trust_remote_code=True, local_files_only=True,
                                  dtype=torch.float16, attn_implementation='eager').to(args.device).eval()
    result['hf_loading_info'] = verify_checkpoint_keys(hf, Path(args.model))
    if any(result['hf_loading_info'][k] for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys')):
        raise RuntimeError('HF weight loading mismatch')
    result['hf_buffer_dtypes'] = {k: str(v.dtype) for k, v in hf.named_buffers()}
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    cases = []
    result['anchor_files'] = []
    for group, directory in enumerate(args.reference_dirs):
        for path in sorted(directory.glob('*.pt')):
            saved = torch.load(path, map_location='cpu', weights_only=True)
            cases.append((f'anchor_{group}_{path.stem}', saved['inputs'], saved['embeddings']))
            result['anchor_files'].append({'path': str(path), 'sha256': sha256(path)})
    if not cases:
        raise RuntimeError('No saved HF tensor anchors found')
    for side in ('left', 'right'):
        processor.tokenizer.padding_side = side
        cases.append((f'queries_{side}_padding', dict(processor.process_queries(
            ['Short query?', 'Find the document explaining how to read the dimensions in a table.'])), None))
        images = []
        for path in [ROOT/'crops/crop_01_text_block_en.png', ROOT/'crops/crop_05_table_rwkv_dims.png']:
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        cases.append((f'images_{side}_padding', dict(processor.process_images(images)), None))
    result['cases'] = []
    outputs = {}
    for name, cpu_inputs, anchor in cases:
        emit('case_start', name=name)
        batch = {k: v.to(args.device) for k, v in cpu_inputs.items()}
        expected, hf_trace, hf_s = forward(hf, batch, True)
        actual, local_trace, local_s = forward(local, batch, False)
        with torch.inference_mode():
            repeated = local(**batch).detach().cpu()
            expected_pos, _ = hf.qwen3vl.get_rope_index(batch['input_ids'], batch.get('image_grid_thw'),
                                                       attention_mask=batch['attention_mask'])
            grids = batch['image_grid_thw'].cpu().tolist() if 'image_grid_thw' in batch else []
            actual_pos = image_positions(batch['input_ids'], batch['attention_mask'], grids, local.config)
        row = {'name': name, 'input_shapes': {k: list(v.shape) for k, v in cpu_inputs.items()},
               'embeddings': difference(actual, expected),
               'repeat': difference(repeated, actual, atol=0, rtol=0),
               'position_ids_exact': torch.equal(actual_pos, expected_pos),
               'trace_keys_match': local_trace.keys() == hf_trace.keys(),
               'layers': {k: difference(local_trace[k], v) for k, v in hf_trace.items() if k in local_trace},
               'instrumented_wall_s': {'hf': hf_s, 'local': local_s}}
        if anchor is not None:
            row['hf_vs_saved_anchor'] = difference(expected, anchor, atol=0, rtol=0)
            row['local_vs_saved_anchor'] = difference(actual, anchor)
        norms = actual.float().norm(dim=-1)
        active = cpu_inputs['attention_mask'].bool()
        row['max_unit_norm_error'] = float((norms[active]-1).abs().max())
        row['padding_zero'] = bool((actual[~active] == 0).all())
        row['passed'] = (row['embeddings']['passed'] and row['repeat']['passed']
                         and row['position_ids_exact'] and row['trace_keys_match']
                         and all(x['passed'] for x in row['layers'].values())
                         and row.get('hf_vs_saved_anchor', {'passed': True})['passed']
                         and row.get('local_vs_saved_anchor', {'passed': True})['passed']
                         and row['max_unit_norm_error'] < 0.002 and row['padding_zero'])
        result['cases'].append(row)
        torch.save({'inputs': cpu_inputs, 'hf': expected, 'local': actual}, args.output_dir/f'{name}.pt')
        outputs[name] = (expected, actual)
        emit('case_finish', name=name, passed=row['passed'], embeddings=row['embeddings'],
             first_bad_layer=next((k for k, v in row['layers'].items() if not v['passed']), None))
        del hf_trace, local_trace, batch
        gc.collect()
    result['maxsim'] = []
    for group in range(len(args.reference_dirs)):
        q = outputs[f'anchor_{group}_queries']
        docs = [outputs[k] for k in sorted(outputs) if k.startswith(f'anchor_{group}_image_')]
        with torch.inference_mode():
            reference_scores = processor.score_multi_vector(list(q[0]), [d[0][0] for d in docs], device=args.device)
            local_scores = processor.score_multi_vector(list(q[1]), [d[1][0] for d in docs], device=args.device)
        row = difference(local_scores.cpu(), reference_scores.cpu(), atol=0.02, rtol=0.001)
        row.update(group=group, hf=reference_scores.tolist(), local=local_scores.tolist())
        result['maxsim'].append(row)
    result['status'] = 'passed' if all(c['passed'] for c in result['cases']+result['maxsim']) else 'failed'
    if result['status'] != 'passed':
        raise RuntimeError('Parity gate failed; see result.json for first divergent stages')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--reference-dirs', type=Path, nargs='+', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='npu:0')
    args = p.parse_args()
    if not args.device.startswith('npu:'):
        p.error('NPU validation only')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ['HF_HUB_OFFLINE'] = '1'
    result = {'status': 'started', 'scope': 'eager modeling parity, not throughput or retrieval accuracy',
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'host': platform.node(), 'physical_npu': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
              'command': sys.argv, 'model_files': model_manifest(Path(args.model), False),
              'versions': {k: importlib.metadata.version(k) for k in ['torch', 'torch-npu', 'transformers']}}
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
