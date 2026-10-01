"""NPU parity/timing ladder: reference eager -> prepared eager -> TorchAir."""
import argparse
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

from local_modeling_colqwen3 import LocalColQwen3
from prepared_prefill import (PreparedVisionStage, PreparedTextStage, prepare_inputs,
                              prepare_text, finish_embeddings, StageCompiler)
from run_hf_baseline import sha256
from run_local_parity import difference


def emit(event, **data):
    print('PREFILL ' + json.dumps({'phase': event, **data}), flush=True)


def timed(call, args):
    torch.npu.synchronize()
    start = time.perf_counter()
    output = call(*args)
    torch.npu.synchronize()
    return output, time.perf_counter()-start


def cpu(value):
    if isinstance(value, tuple):
        return tuple(v.detach().cpu() for v in value)
    return value.detach().cpu()


def compare(value, reference):
    if isinstance(reference, tuple):
        parts = [difference(a, b) for a, b in zip(cpu(value), cpu(reference))]
        return {'passed': len(value) == len(reference) and all(p['passed'] for p in parts), 'parts': parts}
    return difference(cpu(value), cpu(reference))


def measure(call, args, repeats):
    # First call and compile are measured by caller; this warmup is excluded too.
    timed(call, args)
    durations = [timed(call, args)[1] for _ in range(repeats)]
    ms = torch.tensor(durations, dtype=torch.float64)*1000
    return {'samples_s': durations, 'mean_ms': float(ms.mean()),
            'p50_ms': float(ms.quantile(0.5)), 'p90_ms': float(ms.quantile(0.9)),
            'max_ms': float(ms.max()), 'repeats': repeats}


def load_case(path, row, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    inputs = saved['inputs']
    if row >= inputs['input_ids'].shape[0]:
        raise ValueError('Anchor row out of bounds')
    valid = inputs['attention_mask'][row].bool()
    if 'pixel_values' in inputs and inputs['input_ids'].shape[0] != 1:
        raise ValueError('Use a saved B1 image anchor')
    batch = {k: v.to(device) for k, v in inputs.items() if k not in ('input_ids', 'attention_mask')}
    batch['input_ids'] = inputs['input_ids'][row:row+1, valid].contiguous().to(device)
    batch['attention_mask'] = torch.ones_like(batch['input_ids'])
    return batch, saved['embeddings'][row:row+1, valid].contiguous()


@torch.inference_mode()
def run(args, result):
    import torch_npu  # noqa: F401
    if not torch.npu.is_available():
        raise RuntimeError('NPU required; no fallback')
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    result['device_name'] = torch.npu.get_device_name(args.device)
    emit('model_load_start')
    model = LocalColQwen3.from_pretrained(args.model, device=args.device)
    vision, text = PreparedVisionStage(model).eval(), PreparedTextStage(model).eval()
    compiler = StageCompiler(args.model, args.cache_root, emit) if not args.eager_only else None
    result['cache_records'] = compiler.records if compiler else []
    result['cases'] = []
    query_outputs, image_outputs = [], []
    emit('model_load_finish')
    for index, path in enumerate(args.anchors):
        name = f'{index}_{path.parent.parent.name}_{path.stem}'
        row = {'name': name, 'anchor': str(path), 'anchor_sha256': sha256(path), 'status': 'started'}
        result['cases'].append(row)
        emit('case_start', name=name)
        batch, anchor = load_case(path, args.row, args.device)
        expected, row['reference_eager_s'] = timed(lambda: model(**batch), ())
        row['reference_vs_saved_hf'] = compare(expected, anchor)
        if not row['reference_vs_saved_hf']['passed']:
            raise RuntimeError('B1 local reference differs from saved HF anchor')
        prep = prepare_inputs(model, batch)
        eager_vision = None
        if prep.vision_args is not None:
            eager_vision, _ = timed(vision, prep.vision_args)
        text_args = prepare_text(model, prep, eager_vision)
        eager_text, _ = timed(text, text_args)
        eager_embedding = finish_embeddings(model, prep, eager_text)
        row['prepared_eager_vs_reference'] = compare(eager_embedding, expected)
        if not row['prepared_eager_vs_reference']['passed']:
            raise RuntimeError('Prepared eager failed before compilation; reference untouched')
        emit('prepared_eager_pass', name=name, comparison=row['prepared_eager_vs_reference'])
        row['stages'] = {}
        calls = {}
        for stage_name, module, tensors, reference in [
            ('vision', vision, prep.vision_args, eager_vision),
            ('text', text, text_args, eager_text),
        ]:
            if tensors is None:
                continue
            stats = {'input_shapes': [list(t.shape) for t in tensors],
                     'eager': measure(module, tensors, args.repeats)}
            row['stages'][stage_name] = stats
            if compiler:
                call = compiler.get(stage_name, module, tensors)
                calls[stage_name] = call
                emit('graph_first_call_start', stage=stage_name, name=name)
                candidate, first_s = timed(call, tensors)
                emit('graph_first_call_finish', stage=stage_name, name=name, seconds=first_s)
                stats['first_call_s'] = first_s
                # Hidden-state drift is diagnostic; acceptance is embeddings/MaxSim.
                stats['compiled_vs_prepared_eager'] = compare(candidate, reference)
                stats['compiled'] = measure(call, tensors, args.repeats)
                stats['warm_speedup'] = stats['eager']['mean_ms']/stats['compiled']['mean_ms']
                for lane in ('eager', 'compiled'):
                    stats[lane]['real_tokens_per_s'] = tensors[0].numel()/tensors[0].shape[-1]/(stats[lane]['mean_ms']/1000)
                emit('stage_measured', name=name, stage=stage_name, stats=stats)
        if compiler:
            # Combined path uses compiled vision outputs, not the eager features.
            actual_vision = calls['vision'](*prep.vision_args) if prep.vision_args is not None else None
            actual_text_args = prepare_text(model, prep, actual_vision)
            actual_hidden = calls['text'](*actual_text_args)
            actual_embedding = finish_embeddings(model, prep, actual_hidden)
            row['compiled_vs_reference'] = compare(actual_embedding, expected)
            row['compiled_vs_saved_hf'] = compare(actual_embedding, anchor)
            valid_norms = actual_embedding.float().norm(dim=-1)
            row['max_unit_norm_error'] = float((valid_norms-1).abs().max())
            row['status'] = ('passed' if row['compiled_vs_reference']['passed']
                             and row['compiled_vs_saved_hf']['passed'] and row['max_unit_norm_error'] < .002
                             else 'failed')
        else:
            actual_embedding = eager_embedding
            row['status'] = 'passed'
        (image_outputs if prep.vision_args is not None else query_outputs).append(
            (name, cpu(expected), cpu(actual_embedding)))
        torch.save({'inputs': {k: v.cpu() for k,v in batch.items()},
                    'reference': expected.cpu(), 'prepared': eager_embedding.cpu(),
                    'candidate': actual_embedding.cpu()}, args.output_dir/f'{index}.pt')
        emit('case_finish', name=name, status=row['status'], comparison=row.get('compiled_vs_reference'))
        (args.output_dir/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        if row['status'] != 'passed':
            raise RuntimeError('Compiled embedding parity failed; do not accept speed results')
    result['maxsim'] = []
    for query in query_outputs:
        for document in image_outputs:
            scores = []
            for lane in (1, 2):
                q = query[lane][0].to(args.device)
                d = document[lane][0].to(args.device)
                scores.append(torch.einsum('qd,kd->qk', q, d).max(-1).values.sum().view(1).cpu())
            item = difference(scores[1], scores[0], atol=.02, rtol=.001)
            item.update(query=query[0], document=document[0], reference=float(scores[0]), candidate=float(scores[1]))
            result['maxsim'].append(item)
    if not all(item['passed'] for item in result['maxsim']):
        raise RuntimeError('MaxSim parity failed')
    result['status'] = 'passed'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--anchors', type=Path, nargs='+', required=True)
    p.add_argument('--row', type=int, default=0)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, default=Path('.runtime_cache/21_colqwen3/prepared'))
    p.add_argument('--device', default='npu:0')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--eager-only', action='store_true')
    args = p.parse_args()
    if not args.device.startswith('npu:') or args.repeats < 2 or args.row < 0:
        p.error('NPU device, >=2 repeats and nonnegative row required')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {'status': 'started', 'command': sys.argv,
              'scope': 'exact B1 prepared transformer stages, not end-to-end throughput or retrieval accuracy',
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'host': platform.node(), 'physical_npu': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
              'versions': {k: importlib.metadata.version(k) for k in ('torch', 'torch-npu')},
              'eager_only': args.eager_only}
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
