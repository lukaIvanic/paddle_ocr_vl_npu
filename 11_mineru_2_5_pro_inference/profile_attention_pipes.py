#!/usr/bin/env python3
"""Collect PMU groups around existing production replays, preserving cache keys.

One model load per lane, separate profiler sessions per group. Only the timing
boundary is hooked, never model computation. Unsupported counters are explicit.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

METRICS = {
    'pipe': ('PipeUtilization', 'ACL_AICORE_PIPE_UTILIZATION'),
    'arithmetic': ('ArithmeticUtilization', 'ACL_AICORE_ARITHMETIC_UTILIZATION'),
    'memory': ('Memory', 'ACL_AICORE_MEMORY_BANDWIDTH'),
    'memory_l0': ('MemoryL0', 'ACL_AICORE_L0B_AND_WIDTH'),
    'memory_ub': ('MemoryUB', 'ACL_AICORE_MEMORY_UB'),
    'resource_conflict': ('ResourceConflictRatio', 'ACL_AICORE_RESOURCE_CONFLICT_RATIO'),
    'l2': ('L2Cache', 'ACL_AICORE_L2_CACHE'),
    'memory_access': ('MemoryAccess', 'ACL_AICORE_MEMORY_ACCESS'),
}


def save(path, data):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def lane(args):
    import bench_production_vision_attention as bench
    import torch_npu.profiler as prof
    from profile_vision_prefill_lab import _run_parser

    original_measure = bench.measure
    records = []

    def measure_and_profile(fn, steps):
        timing, output = original_measure(fn, steps)
        if args.operator_window:
            import torch_npu
            bench.phase('operator_window_start', name='attention_hot')
            mark = torch_npu.npu.mstx.range_start('attention_hot')
            if not isinstance(mark, int) or mark <= 0:
                raise RuntimeError('MSTX range creation failed; cannot select the production window')
            try:
                current = fn()
                bench.synchronize()
            finally:
                torch_npu.npu.mstx.range_end(mark)
            parity = bench.differences(output, current)
            save(args.output_dir/'operator_window.json',dict(name='attention_hot',parity=parity))
            if not parity['exact'] or parity['nonfinite']:
                raise RuntimeError('operator-profile replay differs from warm candidate')
            bench.phase('operator_window_finish', name='attention_hot')
            return timing, output
        query = getattr(prof, 'supported_ai_core_metrics', None)
        available = query() if query else None
        capabilities = dict(available=sorted(map(str, available)) if available is not None else None,
                            requested=args.metrics.split(','), profiles=records)
        save(args.output_dir / 'metric_sessions.json', capabilities)
        for metric in args.metrics.split(','):
            enum_name, acl_name = METRICS[metric]
            value = getattr(prof.AiCMetrics, enum_name, None)
            row = dict(metric=metric, profile_forwards=args.profile_steps)
            records.append(row)
            if value is None or (available is not None and acl_name not in available and value not in available):
                row.update(status='unsupported_by_runtime', reason='absent enum or capability query rejection')
                save(args.output_dir / 'metric_sessions.json', capabilities)
                bench.phase('metric_unsupported', **row)
                continue
            root = args.output_dir / 'metrics' / metric
            root.mkdir(parents=True)
            row.update(status='started')
            save(args.output_dir / 'metric_sessions.json', capabilities)
            bench.phase('metric_start', metric=metric)
            config = prof._ExperimentalConfig(profiler_level=prof.ProfilerLevel.Level1,
                aic_metrics=value, l2_cache=metric == 'l2', export_type=prof.ExportType.Text)
            start = time.monotonic()
            with prof.profile(activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
                schedule=prof.schedule(wait=0, warmup=0, active=args.profile_steps, repeat=1),
                experimental_config=config, record_shapes=True,
                on_trace_ready=prof.tensorboard_trace_handler(str(root / 'profile'), analyse_flag=True)) as recording:
                for _ in range(args.profile_steps):
                    current = fn()
                    bench.synchronize()
                    recording.step()
            parity = bench.differences(output, current)
            if not parity['exact'] or parity['nonfinite']:
                raise RuntimeError(f'profiled replay changed output: {metric}: {parity}')
            parsed = _run_parser(root / 'profile', root, topn=100)
            csv_files = list((root/'profile').rglob('kernel_details.csv'))
            if len(csv_files) != 1:
                row.update(status='missing_kernel_csv', parsed=parsed,
                           collection_wall_s=time.monotonic()-start)
                save(args.output_dir / 'metric_sessions.json', capabilities)
                raise RuntimeError(f'expected one device kernel CSV for {metric}, got {len(csv_files)}; preserve capture and diagnose')
            row.update(status='completed', collection_wall_s=time.monotonic()-start,
                       repeat_parity=parity, parsed=parsed)
            save(args.output_dir / 'metric_sessions.json', capabilities)
            bench.phase('metric_finish', metric=metric, collection_wall_s=row['collection_wall_s'])
        return timing, output

    bench.measure = measure_and_profile
    try:
        bench.replay(argparse.Namespace(capture_dir=args.capture_dir, cache_root=args.cache_root,
            route=args.route, variant=args.variant, model=None, output_dir=args.output_dir,
            steps=args.steps, profile=False))
    finally:
        bench.measure = original_measure


def suite(args):
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    summary = dict(lanes=[], requested_metrics=args.metrics.split(','))
    for variant in args.variants.split(','):
        for route in args.routes.split(','):
            name = f'{variant}_{route}'
            command = [sys.executable, '-u', str(Path(__file__).resolve()), 'lane',
                '--capture-dir', str(args.capture_dir.resolve()), '--cache-root', str(args.cache_root.resolve()),
                '--output-dir', str(root/name), '--variant', variant, '--route', route,
                '--metrics', args.metrics, '--steps', str(args.steps), '--profile-steps', str(args.profile_steps)]
            (root/f'{name}.command.sh').write_text(shlex.join(command)+'\n')
            start = time.monotonic()
            print(f'PIPES start {name}', flush=True)
            with (root/f'{name}.log').open('w') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                timed_out = False
                while child.poll() is None:
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        print(f'PIPES heartbeat {name} elapsed={time.monotonic()-start:.1f}s', flush=True)
                    if child.poll() is None and time.monotonic()-start > args.timeout_s:
                        timed_out = True
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            pass
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait()
            row = dict(name=name, exit_code=child.returncode, timeout=timed_out, wall_s=time.monotonic()-start)
            summary['lanes'].append(row)
            save(root/'suite.json', summary)
            print('PIPES finish '+json.dumps(row), flush=True)
            if child.returncode or timed_out:
                raise RuntimeError('lane failed; stop and inspect causal log/device health, no automatic fallback')
    print('PIPES complete', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['suite', 'lane'])
    p.add_argument('--capture-dir', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--metrics', default=','.join(METRICS))
    p.add_argument('--variants', default='baseline,eager_pfa,unpad_d80,unpad_d128')
    p.add_argument('--routes', default='bucket_768,bucket_5632')
    p.add_argument('--variant')
    p.add_argument('--route')
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--profile-steps', type=int, default=3)
    p.add_argument('--timeout-s', type=int, default=900)
    p.add_argument('--operator-window', action='store_true', help='Lane only: mark one warm production forward for external msopprof, no torch profiler')
    args = p.parse_args()
    variants = {'baseline','eager_pfa','unpad_d80','unpad_d128','pfa_approx','pfa_d128','pfa_d128_approx'}
    routes = {'bucket_768','packed_768','bucket_5632'}
    if set(args.metrics.split(','))-set(METRICS) or min(args.steps,args.profile_steps,args.timeout_s) <= 0:
        p.error('unknown metric or nonpositive count')
    if set(args.variants.split(','))-variants or set(args.routes.split(','))-routes:
        p.error('unknown variant/route')
    if args.mode == 'lane' and (args.variant not in variants or args.route not in routes):
        p.error('lane requires valid --variant and --route')
    if args.operator_window and args.mode != 'lane':
        p.error('--operator-window requires lane mode')
    if args.operator_window:
        # msopprof application replay may launch the same command repeatedly.
        args.output_dir = args.output_dir / f'process_{os.getpid()}'
    (suite if args.mode == 'suite' else lane)(args)


if __name__ == '__main__':
    main()
