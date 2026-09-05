#!/usr/bin/env python3
"""Profile warm vision calls intercepted in the unchanged production page path.

Diagnostic only: replays distort page throughput and routing counters. Never use
the nested page-run summary as an E2E benchmark. No model/runtime source changes,
new cache identities, synthetic inputs, or replacement packing implementation.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import torch

from prefill_timing import PrefillDeviceTimeline
from profile_vision_prefill_lab import npu_profiler_config, _run_parser
from run_transformers_recognition_smoke import synchronize
from vision_prefill_compile import select_vision_bucket


def stats(values):
    ordered = sorted(values)
    def quantile(p):
        index = (len(ordered) - 1) * p
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    return dict(count=len(values), mean=sum(values)/len(values),
                p50=quantile(.5), p90=quantile(.9), p99=quantile(.99),
                min=ordered[0], max=ordered[-1])


def measure(fn, count):
    wall, device = [], []
    for _ in range(count):
        synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - begin) * 1000)
        device.append(start.elapsed_time(end))
    return dict(wall_ms=stats(wall), device_ms=stats(device),
                wall_samples_ms=wall, device_samples_ms=device), output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-command', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--routes', default='bucket_384,bucket_768,bucket_1536,bucket_3072,bucket_5632,packed_768')
    parser.add_argument('--metrics', default='pipe,memory')
    parser.add_argument('--baseline-steps', type=int, default=30)
    parser.add_argument('--profile-steps', type=int, default=3)
    args = parser.parse_args()
    if min(args.limit, args.baseline_steps, args.profile_steps) <= 0:
        parser.error('counts must be positive')
    routes = set(args.routes.split(','))
    metrics = args.metrics.split(',')
    if not set(metrics) <= {'pipe', 'memory', 'l2', 'memory_access'}:
        parser.error('unknown metric')
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    command = shlex.split(args.reference_command.read_text())
    command[0] = sys.executable
    def replace(flag, value):
        command[command.index(flag) + 1] = str(value)
    replace('--limit', args.limit)
    replace('--warmup-pages', 0)  # only real runtime calls, no resized warmup tensors
    replace('--output-dir', root/'diagnostic_pages_NOT_THROUGHPUT')
    # Preserve the reference's processor limits and all kernel/packing settings.
    (root/'command.sh').write_text(shlex.join(command)+'\n')
    (root/'commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'], text=True))
    (root/'visible_device.txt').write_text(os.environ.get('ASCEND_RT_VISIBLE_DEVICES','')+'\n')
    (root/'pid.txt').write_text(str(os.getpid())+'\n')
    original = PrefillDeviceTimeline.measure
    results = {}
    auxiliaries = {}
    buckets = tuple(map(int, command[command.index('--local-vision-buckets')+1].split(',')))

    def save():
        (root/'result.json').write_text(json.dumps(dict(
            scope='production vision transformer boundary (32 blocks); direct includes padding/mask preparation, packed receives prebuilt inputs',
            excluded='image loading/processor/H2D, text prefill, decode; patch/position/merger separately event-timed',
            diagnostic_page_throughput_valid=False, requested_routes=sorted(routes),
            covered_routes=sorted(results), missing_routes=sorted(routes-set(results)),
            baseline_steps=args.baseline_steps, profile_steps=args.profile_steps,
            results=results, auxiliary_stage_timings=auxiliaries), indent=2)+'\n')

    @torch.inference_mode()
    def instrument(timeline, name, fn, *, tags=None):
        # Complete the real production invocation before any profiled replay.
        output = original(timeline, name, fn, tags=tags)
        if name == 'vision_transformer_blocks' and tags and tags['route'] in routes and tags['route'] not in results:
            route = tags['route']
            print(f'PROFILE_ROUTE start route={route} tags={json.dumps(tags)}', flush=True)
            synchronize()
            anchor = output.detach().clone()
            for _ in range(3):
                fn()
            synchronize()
            before, _ = measure(fn, args.baseline_steps)
            profiles = {}
            import torch_npu.profiler as prof
            for metric in metrics:
                destination = root/route/metric
                destination.mkdir(parents=True)
                print(f'PROFILE_CAPTURE start route={route} metric={metric}', flush=True)
                wall = []
                with prof.profile(
                    activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
                    experimental_config=npu_profiler_config(metric),
                    on_trace_ready=prof.tensorboard_trace_handler(str(destination/'raw'), analyse_flag=True),
                    record_shapes=True, profile_memory=False, with_stack=True,
                ) as capture:
                    for step in range(args.profile_steps):
                        synchronize()
                        begin = time.perf_counter()
                        with torch.profiler.record_function(f'mineru.production.{route}.{metric}.step{step}'):
                            fn()
                            synchronize()
                        wall.append((time.perf_counter()-begin)*1000)
                        capture.step()
                print(f'PROFILE_CAPTURE finish route={route} metric={metric}', flush=True)
                profiles[metric] = dict(profiled_wall_ms=stats(wall),
                    parser=_run_parser(destination/'raw', destination, topn=60))
            after, replay = measure(fn, args.baseline_steps)
            difference = (anchor.float() - replay.float()).abs()
            parity = dict(exact=bool(torch.equal(anchor,replay)),
                          max_abs=float(difference.max().item()),
                          nonfinite=int((~torch.isfinite(replay)).sum().item()))
            if parity['nonfinite'] or not torch.allclose(anchor,replay,atol=1e-3,rtol=1e-3):
                raise RuntimeError(f'vision replay changed output: {route}: {parity}')
            device_ms = (before['device_ms']['mean']+after['device_ms']['mean'])/2
            results[route] = dict(tags=tags, output_shape=list(output.shape), before=before,
                after=after, parity=parity, profiles=profiles,
                real_tok_s=tags['real_tokens']*1000/device_ms,
                physical_tok_s=tags['physical_tokens']*1000/device_ms)
            save()
            print(f'PROFILE_ROUTE finish route={route} device_ms={device_ms:.4f} parity={parity}', flush=True)
        elif name in ('vision_patch_embed','vision_position_prepare','vision_merger'):
            real = (int(output[0][0].shape[0]) if name == 'vision_position_prepare'
                    else int(output.shape[0]) * (4 if name == 'vision_merger' else 1))
            bucket = select_vision_bucket(real, buckets)
            key = f'{name}:real_{real}:bucket_{bucket}'
            # One real shape per selected bucket/stage is enough for auxiliary costs.
            stage_route = f'{name}:bucket_{bucket}'
            if f'bucket_{bucket}' in routes and stage_route not in auxiliaries:
                synchronize()
                for _ in range(3):
                    fn()
                timing, _ = measure(fn, args.baseline_steps)
                auxiliaries[stage_route] = dict(real_tokens=real, sample=key, timing=timing)
                save()
        return output

    lock_dir = Path(command[command.index('--local-vision-torchair-cache-dir')+1])
    code = 1
    try:
        with (lock_dir/'production_profile.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            PrefillDeviceTimeline.measure = instrument
            sys.argv = command[1:]
            import run_official_transformers_omnidocbench as runner
            runner.main()
            save()
            code = 0 if routes <= set(results) else 2
    finally:
        PrefillDeviceTimeline.measure = original
        save()
        (root/'exit_code.txt').write_text(str(code)+'\n')
    print(f'PROFILE_COMPLETE exit={code} missing={sorted(routes-set(results))}', flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
