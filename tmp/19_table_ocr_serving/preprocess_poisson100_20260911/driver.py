"""CPU-readiness measurements on one 910B; run on the direct host.

Reuse the locked client, warmup, ownership checks, and shutdown machinery.
Reuses the already-built 60k graphs; one real request warms each process.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

HISTORICAL = Path('/data1/lukaiv/workspace/repos/table_step1_be691de1_20260910')
RUNTIME_REPO = Path('/data1/lukaiv/workspace/repos/paddle_ocr_vl_npu')
CONTAINER_RUNTIME = '/workspace/repos/paddle_ocr_vl_npu'
BASE = Path('tmp/19_table_ocr_serving/preprocess_poisson100_20260911')
EXPECTED_ORDER = '944d66f46da6db1aa55d59fe31f82d2e57f3f414911bdb2a7e7926f3f64efd0b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--npu', type=int, default=6)
    args = parser.parse_args()
    commit = subprocess.check_output(['git', '-C', str(RUNTIME_REPO), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == args.expected_commit
    assert not subprocess.check_output(['git', '-C', str(RUNTIME_REPO), 'status', '--porcelain', '--', '19_table_ocr_serving'], text=True).strip()
    assert subprocess.check_output(['git', '-C', str(HISTORICAL), 'rev-parse', 'HEAD'], text=True).strip() == 'be691de190ae099d1a9b0ba80865006b122ecc00'
    path = HISTORICAL / '09_persistent_page_engine/scripts/table_poisson_frontier.py'
    source = path.read_text()
    replacements = {
        '"serve_crop_ocr_api.py" in cmd': '"tmp/19_table_ocr_serving/preprocess_options_20260911/preprocess_options_server.py" in cmd',
        'SCRIPTS + "serve_crop_ocr_api.py"': repr(CONTAINER_RUNTIME + '/tmp/19_table_ocr_serving/preprocess_options_20260911/preprocess_options_server.py'),
        ', "--min-pixels", "28224", "--max-pixels", "802816"': ', *SERVER_ARGS',
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    ns = {'__file__': str(path), '__name__': 'locked_head_ab_harness'}
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
    aggregate = HISTORICAL / BASE
    aggregate.mkdir(parents=True, exist_ok=False)
    rows = []
    (aggregate / 'plan.json').write_text(json.dumps({
        'runtime_commit': commit, 'historical_commit': 'be691de1', 'npu': args.npu,
        'matrix': [[8, 'pillow'], [8, 'both']],
        'count': 100, 'seed': 1, 'ordered_ids_sha256': EXPECTED_ORDER,
        'expected_new_decode_graphs': 0, 'decode_device_timing': True,
        'target_qps': 6, 'arrival_process': 'poisson',
        'note': 'Open-loop: scheduled-arrival and actual-submission-to-response latency; warmup outside measured client. Device timing enabled equally in every variant. First-token head remains full.'
    }, indent=2) + '\n')
    for batch, head in ((8, 'pillow'), (8, 'both')):
        ns['SERVER_ARGS'] = [
            '--model', '/workspace/models/PaddleOCR-VL-1.6', '--device', 'npu:0',
            '--decode-device-timing',
            '--bench-resize', 'kornia_rs' if head in ('kornia', 'both') else 'pillow',
            '--bench-uint8', '1' if head in ('uint8', 'both') else '0',
            '--torchair-cache-dir', CONTAINER_RUNTIME + '/.runtime_cache/19_table_ocr_serving_torchair',
            '--vision-torchair-cache-dir', CONTAINER_RUNTIME + '/.runtime_cache/19_table_ocr_serving_vision_torchair',
            '--text-torchair-cache-dir', CONTAINER_RUNTIME + '/.runtime_cache/19_table_ocr_serving_text_torchair',
        ]
        for phase in ('measured',):
            folder = BASE / f'b{batch}_{head}_{phase}'
            sweep = ns['Sweep'](argparse.Namespace(npu=args.npu, count=100, output_dir=folder))
            sweep.write('identity.json', {'runtime_commit': commit, 'batch': batch, 'head': head, 'phase': phase})
            try:
                sweep.log(f'CPU_POISSON {head} B{batch}/C{batch} {phase}')
                run_dir, relative = sweep.start(batch)
                ready = json.loads((run_dir / 'ready.json').read_text())['configuration']
                assert ready['full_decode_lm_head'] == (head == 'full')
                assert ready['preprocessing_benchmark'] == {
                    'resize_backend': 'kornia_rs' if head in ('kornia', 'both') else 'pillow',
                    'compact_uint8': head in ('uint8', 'both'),
                }
                assert ready['batch_size'] == batch and ready['cache_length'] == 4096
                mapping = json.loads((RUNTIME_REPO / '19_table_ocr_serving/presets/table_compact_vocab/native_han_core_60416.json').read_text())
                assert ready['decode_vocab']['selected_vocab_size'] == 60416
                assert ready['decode_vocab']['token_ids_sha256'] == mapping['token_ids_sha256']
                if phase == 'compile_warm':
                    sweep.write('status.json', {'status': 'warm_complete', 'measured_requests': 0})
                    continue
                target = relative / 'measured/results'
                sweep.run_client(ns['docker'](ns['PYTHON'], '-u',
                    CONTAINER_RUNTIME + '/09_persistent_page_engine/scripts/table_request_load_simulator.py',
                    '--api-url', ns['API'] + '/v1/ocr', '--cohort', 'all', '--qps', '6',
                    '--max-requests', '100', '--seed', '1',
                    '--source-jsonl', ns['CONTAINER_REPO'] + '/tmp/09_persistent_page_engine/table_b1_latency_full_04fbc8e/client/tables.jsonl',
                    '--schedule-jsonl', CONTAINER_RUNTIME + '/' + str(BASE / 'schedule.jsonl'),
                    '--output-dir', str(target)), run_dir / 'measured', 1200)
                result_dir = HISTORICAL / target
                summary = json.loads((result_dir / 'summary.json').read_text())
                ids = [json.loads(line)['request_id'] for line in (result_dir / 'schedule.jsonl').read_text().splitlines()]
                assert hashlib.sha256('\n'.join(ids).encode()).hexdigest() == EXPECTED_ORDER
                assert summary['completed_request_count'] == 100 and summary['failed_request_count'] == 0
                row = {'batch': batch, 'head': head, 'qps': summary['completed_request_count']/summary['run_wall_s'],
                       'latency_s': summary['request_latency_s'],
                       'scheduled_latency_s': summary['scheduled_latency_s'],
                       'max_outstanding': summary['max_active_requests'],
                       'drain_s': summary['drain_after_last_arrival_s'], 'artifact': str(target)}
                rows.append(row)
                (aggregate / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
                sweep.write('status.json', {'status': 'measured_complete', 'row': row})
                sweep.log('CPU_POISSON RESULT ' + json.dumps(row))
            except BaseException as exc:
                sweep.write('status.json', {'status': 'failed', 'error': repr(exc)})
                raise
            finally:
                sweep.stop()
                sweep.ownership.close()
        time.sleep(3)
    (aggregate / 'status.json').write_text(json.dumps({'status': 'complete', 'runs': len(rows)}) + '\n')
    print('CPU_POISSON COMPLETE; owned servers stopped and NPU released', flush=True)


if __name__ == '__main__':
    main()
