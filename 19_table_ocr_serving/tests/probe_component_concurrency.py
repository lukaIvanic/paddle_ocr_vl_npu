"""Exercise a platform-style synchronous wrapper, not MEP itself.

fake: real InferenceServer, spawned child, queues and reply dispatch; no Torch/NPU.
npu: the same caller pattern against the unmodified recognition engine.
CPU fault injections demonstrate failure handling, not a diagnosis of MEP.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from functools import partial
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import threading
import time
import traceback
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import p01_serve as serve


class ComponentAdapter:
    """One shared server; concurrent calc calls decode base64 then wait for OCR."""

    def __init__(self, server, *, parallelism=99999, validate_opencv=False):
        self.server = server
        self.parallelism = parallelism
        self.validate_opencv = validate_opencv
        self.active = 0
        self.peak_active = 0
        self.lock = threading.Lock()

    def _increment(self):
        with self.lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)

    def calc(self, request):
        # Deliberately preserve the photographed wrapper's non-atomic admission.
        if self.active >= self.parallelism:
            return dict(code=41003, message='request para exceed max.')
        self._increment()
        try:
            internal_id = f"{request['meta']['uuId']}:{uuid.uuid4().hex}"
            parts = request['data']['messages'][-1]['content']
            image_bytes = base64.b64decode(parts[0]['image_url']['url'].split(',', 1)[1])
            crop_type = {value: key for key, value in serve.PROMPTS.items()}[parts[1]['text']]
            if self.validate_opencv:
                import cv2
                import numpy as np
                decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
                    raise ValueError('invalid image')
            result = self.server.recognize(internal_id, crop_type, image_bytes)
            # Test-only check: detect replies delivered to the wrong caller.
            assert result['request_id'] == internal_id, 'reply routed to wrong caller'
            if not result['ok']:
                return dict(code=50001, message=f"recognition failed: {result['error']}")
            return dict(code=200, results=result['payload']['raw_text'],
                        payload=result['payload'])
        except Exception as exc:
            # Match the platform's generic message, but retain diagnosis in this probe.
            return dict(code=50001, message=f'Exception occurred when predict: {exc}',
                        exception_type=type(exc).__name__, traceback=traceback.format_exc())
        finally:
            with self.lock:
                self.active -= 1


def fake_inference_worker(jobs, results, config, *, gate):
    """CPU reply simulator: varied delays permit out-of-order completion."""
    results.put(dict(kind='ready', worker_pid=os.getpid(),
                     configuration={'probe': 'fake', 'torch_imported': 'torch' in sys.modules}))
    count = 0

    def finish(job):
        if not gate.wait(timeout=15):
            raise RuntimeError('probe gate was not released')
        digest = hashlib.sha256(job['image_bytes']).hexdigest()
        time.sleep((int(digest[:2], 16) % 5 + 1) * .002)
        results.put(dict(kind='result', request_id=job['request_id'], ok=True,
                         payload={'raw_text': job['crop_type'] + ':' + digest,
                                  'crop_type': job['crop_type']}))

    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = []
        while True:
            job = jobs.get()
            if job is None:
                break
            pending.append(pool.submit(finish, job))
            count += 1
        for future in pending:
            future.result()
    results.put(dict(kind='service_summary', payload={'requests': count}))


@contextmanager
def running_server(config, backend, *, hold_results=False):
    gate = mp.get_context('spawn').Event()
    if not hold_results:
        gate.set()
    if backend == 'fake':
        with patch.object(serve, 'run_inference_process', partial(fake_inference_worker, gate=gate)):
            server = serve.InferenceServer(config)
    else:
        server = serve.InferenceServer(config)
    try:
        server.start()
        if backend == 'fake':
            assert not server.worker_runtime_info['torch_imported']
        yield server, gate
    finally:
        gate.set()
        server.close()
        for channel in (server.jobs, server.results):
            channel.close()
            channel.join_thread()


def wait_for_pending(server, count):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with server.requests_lock:
            if len(server.pending_requests) == count:
                return
        time.sleep(.001)
    raise AssertionError(f'expected {count} pending requests')


def make_requests():
    requests = []
    for crop_type, prompt in serve.PROMPTS.items():
        data = (ROOT / 'presets' / f'{crop_type}.png').read_bytes()
        requests.append(dict(meta={'uuId': 'intentionally-repeated-client-id'}, data={
            'messages': [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {
                    'url': 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')}},
                {'type': 'text', 'text': prompt},
            ]}]}))
    return requests


def output_signature(reply):
    if reply['code'] != 200:
        return None
    payload = reply['payload']
    return {key: payload.get(key) for key in ('raw_text', 'text', 'token_ids', 'stop_reason', 'crop_type')}


def run_matrix(config, args, requests):
    rows = []
    with running_server(config, args.backend) as (server, _):
        adapter = ComponentAdapter(server, validate_opencv=args.opencv_validation)
        references = [adapter.calc(request) for request in requests]
        assert all(row['code'] == 200 for row in references), references
        for concurrency in args.concurrency:
            adapter.peak_active = 0
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as callers:
                replies = list(callers.map(adapter.calc,
                    [requests[index % len(requests)] for index in range(args.requests)]))
            errors = [row for row in replies if row['code'] != 200]
            mismatches = sum(output_signature(row) != output_signature(references[index % len(requests)])
                             for index, row in enumerate(replies) if row['code'] == 200)
            row = dict(concurrency=concurrency, requests=len(replies), failures=len(errors),
                       output_mismatches=mismatches, peak_wrapper_calls=adapter.peak_active,
                       elapsed_s=time.perf_counter() - start, error_samples=errors[:3])
            rows.append(row)
            print('MATRIX ' + json.dumps(row), file=sys.__stdout__, flush=True)
            assert adapter.active == 0
        return rows


def run_controlled_failures(config, request, validate_opencv):
    findings = {}
    # A held fake result guarantees that the real server remains at capacity.
    limited = replace(config, max_in_flight_requests=1, log_folder=config.log_folder / 'capacity')
    with running_server(limited, 'fake', hold_results=True) as (server, gate):
        adapter = ComponentAdapter(server, validate_opencv=validate_opencv)
        with ThreadPoolExecutor(max_workers=1) as callers:
            first = callers.submit(adapter.calc, request)
            try:
                wait_for_pending(server, 1)
                rejected = adapter.calc(request)
                assert rejected['exception_type'] == 'InferenceCapacityFull', rejected
                findings['capacity_at_two'] = rejected
            finally:
                gate.set()
            assert first.result(timeout=5)['code'] == 200

    timed = replace(config, request_timeout_s=.05, log_folder=config.log_folder / 'timeout')
    with running_server(timed, 'fake', hold_results=True) as (server, gate):
        adapter = ComponentAdapter(server, validate_opencv=validate_opencv)
        expired = adapter.calc(request)
        assert expired['exception_type'] == 'InferenceTimeout', expired
        findings['timeout'] = expired
        gate.set()
        wait_for_pending(server, 0)
        assert adapter.calc(request)['code'] == 200, 'late result broke subsequent request'

    race_config = replace(config, log_folder=config.log_folder / 'admission_race')
    with running_server(race_config, 'fake') as (server, _):
        adapter = ComponentAdapter(server, parallelism=1, validate_opencv=validate_opencv)
        # Force an allowed interleaving: both pass the check before either increments.
        barrier = threading.Barrier(2, timeout=5)
        original_increment = adapter._increment

        def simultaneous_increment():
            barrier.wait()
            original_increment()
            barrier.wait()

        with patch.object(adapter, '_increment', simultaneous_increment):
            with ThreadPoolExecutor(max_workers=2) as callers:
                replies = list(callers.map(adapter.calc, [request, request]))
        assert all(row['code'] == 200 for row in replies), replies
        assert adapter.peak_active == 2 and adapter.active == 0
        findings['forced_admission_race'] = dict(configured_limit=1, admitted_peak=2,
                                                inference_failures=0)
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('fake', 'npu'), default='fake')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--requests', type=int, default=100)
    parser.add_argument('--concurrency', nargs='+', type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument('--opencv-validation', action='store_true')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--graph-cache-directory', type=Path)
    parser.add_argument('--decode-batch-size', type=int, default=8)
    parser.add_argument('--full-decode-lm-head', action='store_true')
    args = parser.parse_args()
    if args.requests < 1 or any(value < 1 for value in args.concurrency):
        parser.error('request and concurrency counts must be positive')
    if args.backend == 'npu' and (not args.model_path or not args.graph_cache_directory):
        parser.error('NPU mode requires model and graph-cache paths')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = serve.ServeConfig(
        model_path=args.model_path or Path('/unused'),
        graph_cache_directory=args.graph_cache_directory or Path('/unused'),
        log_folder=args.output_dir / 'matrix', decode_batch_size=args.decode_batch_size,
        full_decode_lm_head=args.full_decode_lm_head)
    report = dict(backend=args.backend, python=sys.version, platform=platform.platform(),
                  opencv_validation=args.opencv_validation, max_in_flight_requests=64,
                  decode_batch_size=args.decode_batch_size,
                  full_decode_lm_head=args.full_decode_lm_head,
                  visible_npus=os.getenv('ASCEND_RT_VISIBLE_DEVICES'),
                  scope='Platform-like wrapper, not MEP; fake mode does not execute OCR.')
    with (args.output_dir / 'console.log').open('w') as log, redirect_stdout(log):
        requests = make_requests()
        report['matrix'] = run_matrix(config, args, requests)
        if args.backend == 'fake':
            report['controlled_failures'] = run_controlled_failures(config, requests[0], args.opencv_validation)
    report['passed'] = all(row['failures'] == row['output_mismatches'] == 0 for row in report['matrix'])
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"Report: {args.output_dir / 'report.json'}; passed={report['passed']}")
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
