"""Compare CPU preparation pool sizes without adding a production setting.

Each pool size gets a fresh inference process, one real warmup, and the same
OmniDocBench crops/Poisson arrivals via the ordinary HTTP handler. Only the
named preparation executor is overridden inside the child; model files and
graph-cache identities remain unchanged. This does not reproduce MEP hosting.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import benchmark_tables as benchmark
import p01_serve as serve


def run_worker(jobs, results, config, *, workers, evidence_path):
    try:
        import p02_serving_runtime as runtime

        def preparation_executor(*args, **kwargs):
            assert not args, args
            assert kwargs['thread_name_prefix'] == 'paddleocr-vl-open-cpu-prepare', kwargs
            assert kwargs['max_workers'] == 1, kwargs
            kwargs['max_workers'] = workers
            evidence_path.write_text(json.dumps(dict(
                workers=workers, pid=os.getpid(), executor_arguments=kwargs,
            ), indent=2) + '\n')
            return ThreadPoolExecutor(**kwargs)

        with patch.object(runtime, 'ThreadPoolExecutor', preparation_executor):
            serve.run_inference_process(jobs, results, config)
    except BaseException as exc:
        results.put({'kind': 'startup_error', **serve._describe_failure(exc)})


def run_once(args, workers, directory, schedule, crop_pngs):
    directory.mkdir(parents=True, exist_ok=False)
    config = serve.ServeConfig(
        model_path=args.model_path, graph_cache_directory=args.graph_cache_directory,
        log_folder=directory / 'logs', decode_batch_size=8, port=0,
        metrics_level='detailed',
    )
    target = partial(run_worker, workers=workers, evidence_path=directory / 'executor.json')
    with patch.object(serve, 'run_inference_process', target):
        server = serve.InferenceServer(config)
    http = serve.HttpServer(config, server)
    http_thread = None
    try:
        server.start()
        # The normal run() installs process signals. This test owns shutdown and
        # runs the same HTTP server in a thread so its benchmark can run alongside.
        http.server_bind()
        http.server_activate()
        http_thread = threading.Thread(target=http.serve_forever, daemon=True)
        http_thread.start()
        client_args = argparse.Namespace(
            api_url=f'http://127.0.0.1:{http.server_address[1]}/v1/ocr',
            crop_type=args.crop_type, qps=args.qps, timeout_s=60.0,
        )
        with (directory / 'results.jsonl').open('x') as output:
            summary = asyncio.run(benchmark.run_benchmark(client_args, schedule, crop_pngs, output))
        summary['worker_runtime_info'] = server.worker_runtime_info
    finally:
        if http_thread is not None:
            http.shutdown()
            http_thread.join()
        http.close()
        server.close()
        for channel in (server.jobs, server.results):
            channel.close()
            channel.join_thread()
    assert json.loads((directory / 'executor.json').read_text())['workers'] == workers
    records = [json.loads(line) for line in (directory / 'results.jsonl').read_text().splitlines()]
    # Keep per-request evidence; aggregate existing CPU timings without adding
    # instrumentation to preparation or changing stream synchronization.
    summary['cpu_timing_s'] = {}
    for field in ('cpu_image_decode', 'cpu_image_and_prompt_preprocess',
                  'cpu_mrope_index', 'cpu_pin_memory',
                  'cpu_preprocess_background_queue_wait',
                  'cpu_preprocess_background_service',
                  'cpu_preprocess_background_consumer_wait'):
        values = [record['response']['timing_s'][field] for record in records
                  if record['status'] == 'ok']
        summary['cpu_timing_s'][field] = dict(
            mean=statistics.mean(values) if values else None,
            p95=benchmark.percentile(values, .95), max=max(values) if values else None,
        )
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary, records


def signature(record):
    response = record['response'] or {}
    return {key: response.get(key) for key in
            ('raw_text', 'text', 'token_ids', 'stop_reason', 'crop_type')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--graph-cache-directory', type=Path, required=True)
    parser.add_argument('--omnidocbench', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4, 1])
    parser.add_argument('--requests', type=int, default=100)
    parser.add_argument('--qps', type=float, default=6.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--crop-type', choices=benchmark.CROP_CATEGORIES, default='table')
    args = parser.parse_args()
    if args.requests < 1 or args.qps <= 0 or any(n < 1 for n in args.workers):
        parser.error('counts and QPS must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    crops = benchmark.read_crop_annotations(args.omnidocbench, args.crop_type)
    schedule = benchmark.create_request_schedule(crops, args.requests, args.qps, args.seed)
    crop_pngs = benchmark.prepare_crop_pngs(args.omnidocbench, schedule)
    (args.output_dir / 'schedule.json').write_text(json.dumps(schedule, indent=2) + '\n')
    (args.output_dir / 'input_sha256.json').write_text(json.dumps({
        crop_id: hashlib.sha256(data).hexdigest() for crop_id, data in crop_pngs.items()
    }, indent=2) + '\n')
    report = dict(settings=vars(args).copy(), python=sys.version,
                  visible_npus=os.getenv('ASCEND_RT_VISIBLE_DEVICES'), runs=[])
    reference = None
    for index, workers in enumerate(args.workers):
        directory = args.output_dir / f'{index:02d}_workers{workers}'
        print(f'WORKER_TEST_START workers={workers} output={directory}', flush=True)
        summary, records = run_once(args, workers, directory, schedule, crop_pngs)
        if reference is None:
            reference = records
        # Responses arrive out of order. Compare by the saved request sequence.
        baseline_by_sequence = {record['sequence']: record for record in reference}
        differences = [record['sequence'] for record in records if
                       signature(record) != signature(baseline_by_sequence[record['sequence']])]
        row = dict(workers=workers, output_dir=str(directory), summary=summary,
                   output_mismatch_sequences=differences)
        report['runs'].append(row)
        (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
        print('WORKER_TEST_DONE ' + json.dumps({
            'workers': workers, 'failed': summary['failed'], 'mismatches': len(differences),
            'mean_s': summary['mean_s'], 'p95_s': summary['p95_s'],
            'cpu_timing_s': summary['cpu_timing_s'],
        }), flush=True)
        if summary['failed'] or differences:
            raise SystemExit('Failure or output difference; investigate before continuing.')


if __name__ == '__main__':
    main()
