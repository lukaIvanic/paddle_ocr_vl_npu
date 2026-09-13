"""Replay the saved B8/6-QPS 100-table check against the current product.

Run on the direct 910B host. Reuse the frozen startup/warmup/ownership harness;
adapt only server startup to the current CLI. The saved Poisson client and
100-request schedule are unchanged. No concurrency cap or new arrival draws.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HISTORICAL = Path('/data1/lukaiv/workspace/repos/table_step1_be691de1_20260910')
RUNTIME_REPO = Path('/data1/lukaiv/workspace/repos/paddle_ocr_vl_npu')
CONTAINER_RUNTIME = '/workspace/repos/paddle_ocr_vl_npu'
BASE = Path('tmp/19_table_ocr_serving/refactor_poisson100_20260913')
REFERENCE = Path('tmp/19_table_ocr_serving/preprocess_poisson100_20260911')
CACHE = CONTAINER_RUNTIME + '/.runtime_cache/19_current_checkpoint_20260913'
LOCKED = 'be691de190ae099d1a9b0ba80865006b122ecc00'
CLIENT_SHA = '4651fe03b51c3aa8d8f15f78c6077d823b63c7c8498848379655aa10533c8892'
ORDER_SHA = '944d66f46da6db1aa55d59fe31f82d2e57f3f414911bdb2a7e7926f3f64efd0b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--npu', type=int, required=True, choices=range(8))
    parser.add_argument('--output-dir', type=Path, default=BASE)
    parser.add_argument('--compare-first-use', action='store_true',
        help='Skip real-request warmup, then replay the same 100 arrivals twice on one server.')
    parser.add_argument('--logging-validation', action='store_true')
    parser.add_argument('--metrics-level', choices=('basic', 'detailed'), default='detailed')
    parser.add_argument('--logging-disabled-control', action='store_true')
    args = parser.parse_args()
    assert not args.logging_disabled_control or args.logging_validation
    commit = subprocess.check_output(['git', '-C', str(RUNTIME_REPO), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == args.expected_commit
    assert not subprocess.check_output(['git', '-C', str(RUNTIME_REPO), 'status', '--porcelain', '--', '19_table_ocr_serving'], text=True).strip()
    assert subprocess.check_output(['git', '-C', str(HISTORICAL), 'rev-parse', 'HEAD'], text=True).strip() == LOCKED
    client = RUNTIME_REPO / '09_persistent_page_engine/scripts/table_request_load_simulator.py'
    assert hashlib.sha256(client.read_bytes()).hexdigest() == CLIENT_SHA
    schedule = RUNTIME_REPO / REFERENCE / 'schedule.jsonl'
    schedule_rows = [json.loads(line) for line in schedule.read_text().splitlines()]
    assert len(schedule_rows) == 100
    assert hashlib.sha256('\n'.join(row['request_id'] for row in schedule_rows).encode()).hexdigest() == ORDER_SHA
    assert (RUNTIME_REPO / '.runtime_cache/19_current_checkpoint_20260913').is_dir()
    # Model source is unchanged from the already-compiled checkpoint.
    for name in ('p04_paddle_ocr_vl_1_6_modeling.py', 'p05_vision_prefill.py', 'p06_text_prefill_and_decode.py'):
        relative = '19_table_ocr_serving/' + name
        reference_commit = commit if args.logging_validation else 'ce7a92b1'
        assert (RUNTIME_REPO / relative).read_bytes() == subprocess.check_output(['git', '-C', str(RUNTIME_REPO), 'show', reference_commit + ':' + relative])

    path = HISTORICAL / '09_persistent_page_engine/scripts/table_poisson_frontier.py'
    source = path.read_text()
    server_script = CONTAINER_RUNTIME + '/19_table_ocr_serving/p01_serve.py'
    if args.logging_disabled_control:
        server_script = CONTAINER_RUNTIME + '/tmp/19_table_ocr_serving/logging_20260913/logging_control.py'
    replacements = {
        '"serve_crop_ocr_api.py" in cmd': repr(server_script) + ' in cmd',
        'SCRIPTS + "serve_crop_ocr_api.py"': repr(server_script),
        'self.marker = str(relative / "service.json")': 'self.marker = str(relative / "service_logs")',
        ', "--min-pixels", "28224", "--max-pixels", "802816"': ', *SERVER_ARGS',
        '"--service-summary-output", self.marker': '"--log-folder", self.marker',
    }
    if args.logging_validation:
        replacements['"--queue-capacity", "64"'] = '"--max-in-flight-requests", "64"'
    for old, new in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    if args.compare_first_use:
        # Only remove the benchmark client's warmup invocation. Model setup and
        # its synthetic graph calls remain exactly as they are in production.
        warmup = '''        self.run_client(docker(PYTHON, "-u", SCRIPTS + "table_closed_loop_api_client.py",
            "--api-url", API + "/v1/ocr", "--set", "warm", "--count", "1", "--max-in-flight", "1",
            "--output-dir", str(relative / "warm/results")), folder / "warm", 1200)
'''
        assert source.count(warmup) == 1
        source = source.replace(warmup, '')
        source = source.replace('cached production setup, then one full request warmup',
            'cached production setup; NO real-request warmup')
    ns = {'__file__': str(path), '__name__': 'locked_refactor_poisson100_harness'}
    exec(compile(source, str(path), 'exec'), ns)
    original_fingerprint = ns['fingerprint']

    def fingerprint(repo):
        digest = hashlib.sha256(original_fingerprint(repo).encode())
        for item in sorted((RUNTIME_REPO / '19_table_ocr_serving').rglob('*')):
            if item.suffix in {'.py', '.json'}:
                digest.update(str(item.relative_to(RUNTIME_REPO)).encode())
                digest.update(item.read_bytes())
        return digest.hexdigest()

    ns['fingerprint'] = fingerprint
    ns['CONTAINER_REPO'] = '/workspace/repos/table_step1_be691de1_20260910'
    ns['SERVER_ARGS'] = [
        '--model-path', '/workspace/models/PaddleOCR-VL-1.6', '--device', 'npu:0',
        '--graph-cache-directory', CACHE, '--metrics-level', args.metrics_level,
    ]
    sweep = ns['Sweep'](argparse.Namespace(npu=args.npu, count=100, output_dir=args.output_dir))
    sweep.write('plan.json', dict(runtime_commit=commit, client_sha256=CLIENT_SHA,
        lifecycle_harness_commit=LOCKED, physical_npu=args.npu, batch=8,
        target_qps=6, requests=100, arrival_schedule=str(REFERENCE / 'schedule.jsonl'),
        schedule_sha256=hashlib.sha256(schedule.read_bytes()).hexdigest(),
        ordered_ids_sha256=ORDER_SHA, metrics_level=args.metrics_level, head_rows=60416,
        logging_disabled_control=args.logging_disabled_control,
        cache_root=CACHE, reference=str(REFERENCE / 'b8_both_measured'),
        note=('Cached startup, synthetic graph warmups only; two identical 100-request passes on one server, each awaited in full.'
            if args.compare_first_use else 'Cached startup, one full real warmup outside measurement; no client concurrency cap.')))
    try:
        sweep.log('REFACTOR CHECK B8, saved 100 arrivals at target 6 QPS')
        folder, relative = sweep.start(8)
        ready = json.loads((folder / 'ready.json').read_text())['configuration']
        assert ready['batch_size'] == 8 and ready['cache_length'] == 4096
        assert ready['decode_vocab']['selected_vocab_size'] == 60416
        assert ready['decode_vocab']['token_ids_sha256'] == 'c730b5388f9871ead92e2cb484f8df81ba69518f1e8baeccc5a37f44c1514637'
        assert ready['decode_device_timing'] is (not args.logging_validation)
        assert ready['request_scheduling_metrics'] is (not args.logging_validation)
        passes = ('first', 'second') if args.compare_first_use else ('measured',)
        pass_results = {}
        for name in passes:
            target = relative / name / 'results'
            sweep.log(f'BEGIN {name}: same 100 tables, same saved arrival schedule')
            # The client exits only after every response arrives. No shutdown or
            # drain request is sent between passes; this server stays running.
            sweep.run_client(ns['docker'](ns['PYTHON'], '-u', CONTAINER_RUNTIME + '/09_persistent_page_engine/scripts/table_request_load_simulator.py',
                '--api-url', ns['API'] + '/v1/ocr', '--cohort', 'all', '--qps', '6',
                '--max-requests', '100', '--seed', '1',
                '--source-jsonl', ns['CONTAINER_REPO'] + '/tmp/09_persistent_page_engine/table_b1_latency_full_04fbc8e/client/tables.jsonl',
                '--schedule-jsonl', CONTAINER_RUNTIME + '/' + str(REFERENCE / 'schedule.jsonl'),
                '--output-dir', str(target)), folder / name, 1200)
            result_dir = HISTORICAL / target
            measured_schedule = [json.loads(line) for line in (result_dir / 'schedule.jsonl').read_text().splitlines()]
            assert measured_schedule == schedule_rows, 'Request IDs or arrival timestamps changed'
            summary = json.loads((result_dir / 'summary.json').read_text())
            assert summary['completed_request_count'] == 100 and summary['failed_request_count'] == 0
            pass_results[name] = dict(schedule_identical=True,
                latency_s=summary['request_latency_s'], scheduled_latency_s=summary['scheduled_latency_s'],
                completion_qps=100 / summary['run_wall_s'], max_outstanding=summary['max_active_requests'])
            sweep.write('status.json', dict(status='complete' if name == passes[-1] else 'between_passes', passes=pass_results))
            sweep.log(f'COMPLETE {name}: all 100 responses received ' + json.dumps(summary['request_latency_s']))
    except BaseException as exc:
        sweep.write('status.json', dict(status='failed', error=repr(exc)))
        raise
    finally:
        sweep.stop()
        sweep.ownership.close()


if __name__ == '__main__':
    main()
