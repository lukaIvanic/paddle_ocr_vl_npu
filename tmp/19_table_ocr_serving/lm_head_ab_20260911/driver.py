"""Four closed-loop head comparisons on one 910B; run on the direct host.

Reuse the locked client, warmup, ownership checks, and shutdown machinery.
Each variant first compiles/warms, then restarts for cached measurements.
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
BASE = Path('tmp/19_table_ocr_serving/lm_head_ab_20260911')
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
        '"serve_crop_ocr_api.py" in cmd': '"19_table_ocr_serving/serve.py" in cmd',
        'SCRIPTS + "serve_crop_ocr_api.py"': repr(CONTAINER_RUNTIME + '/19_table_ocr_serving/serve.py'),
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
        'matrix': [[2, 'trimmed'], [2, 'full'], [8, 'trimmed'], [8, 'full']],
        'count': 100, 'seed': 1, 'ordered_ids_sha256': EXPECTED_ORDER,
        'expected_distinct_graphs': 19, 'decode_device_timing': True,
        'note': 'Actual closed-loop submission-to-response latency; warmup outside measured client. Device timing enabled equally in every variant. First-token head remains full.'
    }, indent=2) + '\n')
    for batch, head in ((2, 'trimmed'), (2, 'full'), (8, 'trimmed'), (8, 'full')):
        ns['SERVER_ARGS'] = [
            '--model', '/workspace/models/PaddleOCR-VL-1.6', '--device', 'npu:0',
            '--decode-device-timing',
            '--torchair-cache-dir', CONTAINER_RUNTIME + '/.runtime_cache/19_table_ocr_serving_torchair',
            '--vision-torchair-cache-dir', CONTAINER_RUNTIME + '/.runtime_cache/19_table_ocr_serving_vision_torchair',
            '--text-torchair-cache-dir', CONTAINER_RUNTIME + '/.runtime_cache/19_table_ocr_serving_text_torchair',
        ] + (['--full-decode-lm-head'] if head == 'full' else [])
        for phase in ('compile_warm', 'measured'):
            folder = BASE / f'b{batch}_{head}_{phase}'
            sweep = ns['Sweep'](argparse.Namespace(npu=args.npu, count=100, output_dir=folder))
            sweep.write('identity.json', {'runtime_commit': commit, 'batch': batch, 'head': head, 'phase': phase})
            try:
                sweep.log(f'HEAD_AB {head} B{batch}/C{batch} {phase}')
                run_dir, relative = sweep.start(batch)
                ready = json.loads((run_dir / 'ready.json').read_text())['configuration']
                assert ready['full_decode_lm_head'] == (head == 'full')
                assert ready['batch_size'] == batch and ready['cache_length'] == 4096
                if phase == 'compile_warm':
                    sweep.write('status.json', {'status': 'warm_complete', 'measured_requests': 0})
                    continue
                target = relative / 'measured/results'
                sweep.run_client(ns['docker'](ns['PYTHON'], '-u', ns['SCRIPTS'] + 'table_closed_loop_api_client.py',
                    '--api-url', ns['API'] + '/v1/ocr', '--set', 'random', '--count', '100',
                    '--shuffle-seed', '1', '--max-in-flight', str(batch),
                    '--client-label', f'b{batch}-{head}', '--output-dir', str(target)), run_dir / 'measured', 1200)
                result_dir = HISTORICAL / target
                summary = json.loads((result_dir / 'summary.json').read_text())
                ids = [json.loads(line)['request_id'] for line in (result_dir / 'tables.jsonl').read_text().splitlines()]
                assert hashlib.sha256('\n'.join(ids).encode()).hexdigest() == EXPECTED_ORDER
                assert summary['request_count'] == 100 and summary['failed_request_count'] == 0
                row = {'batch': batch, 'head': head, 'qps': summary['completion_qps'],
                       'latency_s': summary['latency_s'], 'artifact': str(target)}
                rows.append(row)
                (aggregate / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
                sweep.write('status.json', {'status': 'measured_complete', 'row': row})
                sweep.log('HEAD_AB RESULT ' + json.dumps(row))
            except BaseException as exc:
                sweep.write('status.json', {'status': 'failed', 'error': repr(exc)})
                raise
            finally:
                sweep.stop()
                sweep.ownership.close()
        time.sleep(3)
    (aggregate / 'status.json').write_text(json.dumps({'status': 'complete', 'runs': len(rows)}) + '\n')
    print('HEAD_AB COMPLETE; owned servers stopped and NPU released', flush=True)


if __name__ == '__main__':
    main()
