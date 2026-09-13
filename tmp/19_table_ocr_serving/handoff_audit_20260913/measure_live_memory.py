"""Observe an existing OCR server while replaying its saved 1,000-table load.

Run inside the validation container after source npu-setup. This does not
modify or restart the server, import torch, or change NPU execution. Linux
smaps_rollup measures resident/proportional host RAM; npu-smi measures HBM.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-pid', type=int, required=True)
    parser.add_argument('--http-pid', type=int, required=True)
    parser.add_argument('--npu-host-pid', type=int, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    repo = Path('/workspace/repos/paddle_ocr_vl_npu')
    args.output.mkdir(parents=True, exist_ok=False)
    python = '/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python'
    with urlopen('http://127.0.0.1:8767/ready', timeout=5) as reply:
        ready = json.load(reply)
    assert ready['ready'] and ready['worker_pid'] == args.worker_pid
    assert ready['configuration']['batch_size'] == 8
    assert ready['configuration']['decode_vocab']['selected_vocab_size'] == 60416
    (args.output / 'ready.json').write_text(json.dumps(ready, indent=2))
    (args.output / 'source_commit.txt').write_text(subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True))
    stage = {'phase': 'baseline_idle', 'completed_rounds': 0}
    stop = threading.Event()
    failures = []

    def memory(pid):
        result = {'pid': pid}
        for line in (Path('/proc') / str(pid) / 'smaps_rollup').read_text().splitlines():
            if ':' not in line:
                continue
            name, value = line.split(':', 1)
            if name in ('Rss', 'Pss', 'Private_Clean', 'Private_Dirty', 'Anonymous', 'Swap'):
                result[name + '_KiB'] = int(value.split()[0])
        return result

    def snapshot():
        # Include compiler/helper descendants, but not this monitor or the client.
        parents = {}
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():
                continue
            try:
                status = (path / 'status').read_text()
                parents[int(path.name)] = int(re.search(r'^PPid:\s+(\d+)', status, re.M)[1])
            except (FileNotFoundError, ProcessLookupError):
                continue
        pids = {args.http_pid}
        while True:
            expanded = pids | {pid for pid, parent in parents.items() if parent in pids}
            if expanded == pids:
                break
            pids = expanded
        processes = [memory(pid) for pid in sorted(pids)]
        raw = subprocess.check_output(['npu-smi', 'info'], text=True, timeout=10)
        npu_processes = re.findall(r'^\|\s*6\s+0\s*\|\s*(\d+)\s*\|[^|]*\|\s*(\d+)\s*\|', raw, re.M)
        assert {int(pid) for pid, _ in npu_processes} == {args.npu_host_pid}, 'NPU6 ownership changed'
        lines = raw.splitlines()
        device_line = next(i for i, line in enumerate(lines) if re.match(r'^\|\s*6\s+910B2', line))
        hbm = int(re.search(r'(\d+)\s*/\s*65536', lines[device_line + 1])[1])
        row = dict(timestamp=datetime.now(timezone.utc).isoformat(), monotonic_s=time.monotonic(),
                   **stage, processes=processes, device_hbm_MB=hbm,
                   npu_process_memory_MB=int(npu_processes[0][1]), npu_smi=raw)
        row['worker_Rss_KiB'] = next(p['Rss_KiB'] for p in processes if p['pid'] == args.worker_pid)
        row['http_Rss_KiB'] = next(p['Rss_KiB'] for p in processes if p['pid'] == args.http_pid)
        row['server_tree_Pss_KiB'] = sum(p['Pss_KiB'] for p in processes)
        return row

    def monitor():
        try:
            with (args.output / 'memory.jsonl').open('w') as handle:
                while not stop.is_set():
                    row = snapshot()
                    handle.write(json.dumps(row) + '\n')
                    handle.flush()
                    stop.wait(5)
        except BaseException as error:
            failures.append(repr(error))
            stop.set()

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        print('BASELINE', str(args.output), flush=True)
        if stop.wait(20):
            raise RuntimeError(failures)
        for number in range(1, args.rounds + 1):
            folder = args.output / f'round{number}'
            folder.mkdir()
            command = [python, '-u', str(repo / '09_persistent_page_engine/scripts/table_request_load_simulator.py'),
                '--api-url', 'http://127.0.0.1:8767/v1/ocr', '--cohort', 'all', '--qps', '6',
                '--max-requests', '1000', '--seed', '1', '--shuffle-all',
                '--source-jsonl', '/workspace/repos/table_step1_be691de1_20260910/tmp/09_persistent_page_engine/table_b1_latency_full_04fbc8e/client/tables.jsonl',
                '--schedule-jsonl', str(repo / 'tmp/19_table_ocr_serving/integrated_runtime_20260913/poisson1000/cached/b8/qps6/measured/schedule.jsonl'),
                '--output-dir', str(folder / 'results')]
            (folder / 'command.json').write_text(json.dumps(command, indent=2))
            stage['phase'] = f'round{number}_load'
            print('START', number, flush=True)
            with (folder / 'client.log').open('w') as log:
                process = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
                started = time.monotonic()
                try:
                    while process.poll() is None:
                        if stop.wait(1):
                            raise RuntimeError(failures)
                        if time.monotonic() - started > 900:
                            raise TimeoutError('Measured client exceeded 15 minutes')
                    assert process.returncode == 0, (number, process.returncode)
                finally:
                    if process.poll() is None:
                        process.terminate()  # Only the client started by this script.
                        process.wait(timeout=10)
            summary = json.loads((folder / 'results/summary.json').read_text())
            assert summary['completed_request_count'] == 1000 and summary['failed_request_count'] == 0
            stage.update(phase=f'round{number}_drained_idle', completed_rounds=number)
            print('DRAINED', number, json.dumps(summary['request_latency_s']), flush=True)
            if stop.wait(20):
                raise RuntimeError(failures)
        print('DONE', str(args.output), flush=True)
    finally:
        stop.set()
        thread.join(timeout=15)
        (args.output / 'monitor_errors.json').write_text(json.dumps(failures))
    assert not failures, failures


if __name__ == '__main__':
    main()
