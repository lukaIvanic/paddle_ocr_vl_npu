"""Owned RWKV-7 embedding forward and real-case CPU/NPU comparison.

Math follows the pinned EmbeddingRWKV C/CUDA sources in PLAN.md (Apache-2.0).
Dense projections cover B*T rows; recurrence is the separately named Ascend op.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from probe_wkv7 import load_bridge
from run_cpu_reference import CReference, CHECKPOINT_SHA256, load_cases, sha256


from local_modeling_rwkv_embedding import Embedding


def comparison(actual, expected):
    delta = actual - expected
    return {'max_abs': float(np.abs(delta).max()),
            'normalized_rmse': float(np.sqrt(np.mean(delta**2)) / max(np.sqrt(np.mean(expected**2)), 1e-12))}


def timings(call, synchronize, repeats, warmups):
    for _ in range(warmups):
        call()
    synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        synchronize()
        times.append(time.perf_counter() - start)
    return {'samples_seconds': times, 'median_seconds': statistics.median(times),
            'min_seconds': min(times)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--anchors', type=Path, required=True)
    p.add_argument('--build-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=('raw_eager', 'torchair'), default='raw_eager')
    p.add_argument('--dtype', choices=('fp16', 'bf16', 'fp32'), default='fp16',
                   help='Dense projection precision; fp32 is a diagnostic control. State/pointwise math stays FP32.')
    p.add_argument('--case-ids', default='')
    p.add_argument('--batch-size', type=int, choices=(1, 2), default=1)
    p.add_argument('--cpu-repeats', type=int, default=2)
    p.add_argument('--npu-repeats', type=int, default=5)
    p.add_argument('--allow-shared-device', action='store_true')
    args = p.parse_args()
    if not 1 <= args.cpu_repeats <= 10 or not 1 <= args.npu_repeats <= 100:
        p.error('Use CPU repeats 1..10 and NPU repeats 1..100')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'script_sha256': sha256(__file__), 'checkpoint_sha256': CHECKPOINT_SHA256,
              'physical_npu': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), 'backend': args.backend,
              'dense_dtype': args.dtype, 'state_and_pointwise_dtype': 'float32',
              'shared_device': args.allow_shared_device, 'timing_scope': 'forward only, prepared inputs; no trace or transfers',
              'cpu_threads': 4, 'batches': [], 'all_checks_passed': False}
    cpu = None
    try:
        import torch_npu
        torch.set_num_threads(4)
        load_bridge(Path(__file__).parent / 'wkv7_npu', args.build_root)
        status = subprocess.check_output(['/usr/local/bin/npu-status'], text=True)
        report['device_status'] = status
        selected = next((x for x in status.splitlines() if x.startswith(f"NPU {report['physical_npu']}: ")), '')
        if 'Health=OK' not in selected or (not args.allow_shared_device and ': free ' not in selected):
            raise RuntimeError('Selected NPU is unhealthy or occupied without shared-device authorization')
        torch.npu.set_device(0)
        free, total = torch.npu.mem_get_info()
        report['hbm_before_model'] = {'free_bytes': free, 'total_bytes': total}
        if free < 2 * 1024**3:
            raise RuntimeError('Need at least 2 GiB free HBM for model smoke')
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = False
        report.update(device=torch.npu.get_device_name(0), torch=torch.__version__, torch_npu=torch_npu.__version__)
        cases = load_cases(Path(__file__).parent / 'data/smoke_cases.json')
        wanted = args.case_ids.split(',') if args.case_ids else [x['id'] for x in cases]
        if not set(wanted).issubset({x['id'] for x in cases}):
            raise ValueError('Unknown case id')
        cases = [x for x in cases if x['id'] in wanted]
        cpu = CReference(args.runtime, 4)
        anchor_report = json.loads((args.anchors / 'result.json').read_text())
        anchor_map = {row['cases'][0]['id']: row for row in anchor_report['batches']}
        batches = []
        for offset in range(0, len(cases), args.batch_size):
            batch = cases[offset:offset + args.batch_size]
            ids, mask = cpu.prepare_batch([x['encoded_text'] for x in batch])
            batches.append((batch, ids, mask))
        if args.backend == 'torchair' and len({tuple(ids.shape) for _, ids, _ in batches}) != 1:
            raise ValueError('TorchAir gate uses one B/T shape per process; select case ids')
        dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.dtype]
        model = Embedding(args.checkpoint, 'npu:0', dtype)
        forward = model.forward
        if args.backend == 'torchair':
            import torchair
            from torchair.configs.compiler_config import CompilerConfig
            from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
            from torchair import ge
            @register_fx_node_ge_converter(torch.ops.rwkv_reference.wkv7.default)
            def convert(k, v, w, r, a, b, hi, meta_outputs=None):
                return ge.custom_op('RwkvReferenceWkv7', inputs=dict(k=k,v=v,w=w,r=r,a=a,b=b,hi=hi),
                                    attrs={}, outputs=['o','ho'])
            forward = torchair.inference.cache_compile(model.forward, config=CompilerConfig(), dynamic=False,
                        cache_dir=str(args.output / 'graph_cache'), ge_cache=True)
        layer_limit, embedding_limit = ((1e-4, 1e-5) if args.dtype == 'fp32' else (.02, .002))
        report['thresholds'] = {'layer_normalized_rmse': layer_limit, 'embedding_max_abs': embedding_limit,
                                'embedding_min_cosine': .9995}
        with torch.inference_mode():
            for batch, ids, mask in batches:
                expected, layers = cpu.encode(ids, mask, trace=True)
                if len(batch) == 1:
                    anchor = anchor_map[batch[0]['id']]
                    path = args.anchors / anchor['artifact']
                    if sha256(path) != anchor['artifact_sha256']:
                        raise ValueError('CPU anchor hash mismatch')
                    with np.load(path) as saved:
                        assert np.array_equal(ids, saved['input_ids']) and np.array_equal(mask, saved['eos_mask'])
                        assert np.array_equal(expected, saved['embeddings']) and np.array_equal(layers, saved['layer_outputs'])
                ni = torch.from_numpy(ids.astype(np.int64)).to('npu')
                nm = torch.from_numpy(mask.astype(np.float32)).to('npu')
                actual, traced = model.run(ni, nm, True)
                layer_metrics = [comparison(x.cpu().numpy(), y) for x, y in zip(traced, layers)]
                assert all(x['normalized_rmse'] <= layer_limit for x in layer_metrics), layer_metrics
                start = time.perf_counter()
                result = forward(ni, nm)
                torch.npu.synchronize()
                first_seconds = time.perf_counter() - start
                got = result.cpu().numpy()
                metrics = comparison(got, expected)
                cosine = (got * expected).sum(-1) / (np.linalg.norm(got, axis=-1) * np.linalg.norm(expected, axis=-1))
                assert np.isfinite(got).all() and metrics['max_abs'] <= embedding_limit and cosine.min() >= .9995, (metrics, cosine)
                eager_embedding = actual.cpu().numpy()
                assert np.allclose(got, eager_embedding, atol=embedding_limit, rtol=.01)
                repeated = forward(ni, nm).cpu().numpy()
                assert np.array_equal(got, repeated), 'Repeated embedding changed'
                cpu_time = timings(lambda: cpu.encode(ids, mask, False), lambda: None, args.cpu_repeats, 1)
                npu_time = timings(lambda: forward(ni, nm), torch.npu.synchronize, args.npu_repeats, 2)
                eager_time = (timings(lambda: model.forward(ni, nm), torch.npu.synchronize, args.npu_repeats, 2)
                              if args.backend == 'torchair' else npu_time)
                row = {'case_ids': [x['id'] for x in batch], 'shape': list(ids.shape),
                       'raw_tokens': [len(cpu.tokenize(x['encoded_text'])) for x in batch],
                       'selected_eos_counts': mask.sum(1).tolist(), 'input_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
                       'mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest(), 'layers': layer_metrics,
                       'embedding': metrics, 'cosines': cosine.tolist(), 'repeat_bitwise_equal': True,
                       'forward_vs_traced_eager_bitwise_equal': bool(np.array_equal(got, eager_embedding)),
                       'first_forward_seconds': first_seconds, 'cpu': cpu_time, 'npu': npu_time,
                       'eager': eager_time,
                       'cpu_to_npu_ratio': cpu_time['median_seconds'] / npu_time['median_seconds']}
                report['batches'].append(row)
                print(json.dumps(row), flush=True)
        report['all_checks_passed'] = True
    except Exception as e:
        report['error'] = f'{type(e).__name__}: {e}'
        raise
    finally:
        if cpu is not None:
            cpu.close()
        (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
