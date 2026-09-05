#!/usr/bin/env python3
"""Sequential subprocess driver; separate logs, deadlines and heartbeat per lane."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-command', type=Path)
    p.add_argument('--capture-dir', type=Path)
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--routes', default='bucket_768,packed_768,bucket_5632')
    p.add_argument('--variants', default='baseline,pfa_d128,pfa_approx,pfa_d128_approx,eager_pfa,unpad_d80,unpad_d128')
    p.add_argument('--timeout-s', type=int, default=900)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--profile', action='store_true')
    args = p.parse_args()
    if not args.capture_dir and not args.reference_command:
        p.error('provide an existing capture or a production reference command')
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name('bench_production_vision_attention.py')
    summary = dict(lanes=[], scope='diagnostic, not page throughput')

    def run(name, command):
        log = root / (name + '.log')
        (root / (name + '.command.sh')).write_text(shlex.join(command) + '\n')
        start = time.monotonic()
        print(f'MATRIX start lane={name} log={log}', flush=True)
        with log.open('w') as out:
            child = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    print(f'MATRIX heartbeat lane={name} elapsed_s={time.monotonic()-start:.1f} log_bytes={log.stat().st_size}', flush=True)
                if child.poll() is None and time.monotonic() - start > args.timeout_s:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    # The Python parent may exit before its compiler children.
                    # This group was created solely for this diagnostic lane.
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    summary['lanes'].append(dict(name=name, status='timeout', log=str(log)))
                    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
                    raise RuntimeError('lane timed out; stop matrix and inspect NPU health before any retry')
        row = dict(name=name, exit_code=child.returncode, wall_s=time.monotonic()-start, log=str(log))
        summary['lanes'].append(row)
        (root / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(f'MATRIX finish {json.dumps(row)}', flush=True)
        if child.returncode:
            raise RuntimeError(f'lane failed: {name}; inspect log, no automatic fallback')

    capture = args.capture_dir or root / 'capture'
    if not args.capture_dir:
        run('capture', [sys.executable, str(script), 'capture', '--reference-command', str(args.reference_command),
            '--output-dir', str(capture), '--routes', args.routes])
    for variant in args.variants.split(','):
        for route in args.routes.split(','):
            name = f'{variant}_{route}'
            command = [sys.executable, str(script), 'replay', '--capture-dir', str(capture),
                '--cache-root', str(args.cache_root), '--route', route, '--variant', variant,
                '--output-dir', str(root / name), '--steps', str(args.steps)]
            if args.profile:
                command.append('--profile')
            run(name, command)
    print('MATRIX complete', flush=True)


if __name__ == '__main__':
    main()
