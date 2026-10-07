#!/usr/bin/env python3
"""Immutable launch receipts, host/device snapshots and bounded child processes."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time


def command_result(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return dict(argv=argv, exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(argv=argv, unavailable=str(error))


def snapshot():
    visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')
    card = visible.split(',')[0] if visible else None
    commands = [ ['uptime'], ['nproc'], ['npu-smi', 'info'],
                ['ps', '-eo', 'pid,ppid,comm,pcpu,pmem,etimes'] ]
    if card and card.isdigit():
        for kind in ['health', 'power', 'usages', 'sensors', 'common', 'work-mode']:
            commands.append(['npu-smi', 'info', '-t', kind, '-i', card, '-c', '0'])
    records = [command_result(c) for c in commands]
    health = next((r.get('stdout', '') for r in records if 'health' in r['argv']), '')
    bad = bool(re.search(r'health[^\n]*\b(alarm|warning|critical|error|fault|abnormal)\b', health, re.I))
    telemetry = '\n'.join(r.get('stdout', '') for r in records[2:] if r.get('exit_code') == 0)
    clock_lines = [s for s in telemetry.splitlines() if re.search(r'\b(clock|freq\w*|MHz|GHz)\b', s, re.I)]
    mode = next((r for r in records if 'work-mode' in r['argv']), {})
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                hostname=os.uname().nodename, visible_devices=visible,
                load_average=list(os.getloadavg()), cpu_count=os.cpu_count(),
                affinity_cpu_count=len(os.sched_getaffinity(0)), commands=records,
                clock_data=clock_lines or 'not exposed by successful queried interfaces',
                performance_mode_data=mode or 'not available', device_error=bad,
                job_scope='process names/PIDs and NPU process table; command arguments omitted to avoid credentials')


def write_new(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')


def run_lane(command, receipt_dir, timeout_s, log_path=None):
    root = Path(receipt_dir)
    root.mkdir(parents=True, exist_ok=False)
    write_new(root/'command.json', dict(argv=command, shell_display=shlex.join(command),
              cwd=os.getcwd(), source_commit=command_result(['git','rev-parse','HEAD']),
              timeout_s=timeout_s, environment={k:os.environ.get(k) for k in
              ['ASCEND_RT_VISIBLE_DEVICES','TORCH_DEVICE_BACKEND_AUTOLOAD','LD_LIBRARY_PATH',
               'OMP_NUM_THREADS','MKL_NUM_THREADS','VLLM_WORKER_MULTIPROC_METHOD']}))
    before = snapshot()
    write_new(root/'before.json', before)
    if before['device_error']:
        write_new(root/'exit.json', dict(status='device_error_before_launch', exit_code=1))
        raise RuntimeError('selected device reports an error; no child launched')
    log = Path(log_path) if log_path else root/'run.log'
    start = time.monotonic()
    timed_out = False
    with log.open('x') as out:
        child = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        write_new(root/'process.json', dict(pid=child.pid, process_group=child.pid))
        while child.poll() is None:
            try:
                child.wait(timeout=min(15, max(.1, timeout_s-(time.monotonic()-start))))
            except subprocess.TimeoutExpired:
                print(f'LANE heartbeat pid={child.pid} elapsed_s={time.monotonic()-start:.1f} log={log}', flush=True)
            if child.poll() is None and time.monotonic()-start >= timeout_s:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                # Kill only compiler descendants in the process group we created.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
    elapsed = time.monotonic()-start
    after = snapshot()
    write_new(root/'after.json', after)
    status = 'timeout' if timed_out else 'device_error' if after['device_error'] else 'completed' if child.returncode == 0 else 'failed'
    result = dict(status=status, exit_code=child.returncode, child_wall_s=elapsed,
                  log=str(log), receipt_dir=str(root))
    write_new(root/'exit.json', result)
    print('LANE finish '+json.dumps(result), flush=True)
    if status != 'completed':
        raise RuntimeError(f'{status}: stop and inspect {root}; no automatic retry or fallback')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--timeout-s', type=int, default=1800)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command or a.timeout_s <= 0:
        p.error('provide a command and positive deadline')
    run_lane(command, a.output_dir, a.timeout_s)


if __name__ == '__main__':
    main()
