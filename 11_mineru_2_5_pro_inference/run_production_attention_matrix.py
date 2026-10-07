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
from vision_diagnostic_runner import run_lane


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
    p.add_argument('--config-json', type=Path)
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
        print(f'MATRIX start lane={name} log={log}', flush=True)
        try:
            row = run_lane(command, root/(name+'.receipt'), args.timeout_s, log)
        except Exception:
            receipt = root/(name+'.receipt')/'exit.json'
            row = json.loads(receipt.read_text()) if receipt.exists() else dict(status='launch_failed')
            summary['lanes'].append(dict(name=name, **row))
            (root/'summary.json').write_text(json.dumps(summary,indent=2))
            raise
        summary['lanes'].append(dict(name=name, **row))
        (root/'summary.json').write_text(json.dumps(summary,indent=2))
        print(f'MATRIX finish {json.dumps(row)}', flush=True)

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
            if args.config_json:
                command.extend(['--config-json',str(args.config_json)])
            if args.profile:
                command.append('--profile')
            run(name, command)
    print('MATRIX complete', flush=True)


if __name__ == '__main__':
    main()
