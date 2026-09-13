"""Validate the current product with the locked B8 / 6-QPS / 1000-table client.

Run on the direct 910B host. Only the launcher is adapted to today's CLI;
the historical client, request schedule, warmup and ownership checks stay fixed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HISTORICAL = Path('/data1/lukaiv/workspace/repos/table_step1_be691de1_20260910')
RUNTIME_REPO = Path('/data1/lukaiv/workspace/repos/paddle_ocr_vl_npu')
CONTAINER_RUNTIME = '/workspace/repos/paddle_ocr_vl_npu'
BASE = Path('tmp/19_table_ocr_serving/current_checkpoint_20260913')
CACHE = CONTAINER_RUNTIME + '/.runtime_cache/19_current_checkpoint_20260913'
LOCKED = 'be691de190ae099d1a9b0ba80865006b122ecc00'
REFERENCE = Path('tmp/19_table_ocr_serving/step3_b8qps6_0976fa33_20260911_cached')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--npu', type=int, required=True, choices=range(8))
    parser.add_argument('--output-dir', type=Path, default=BASE)
    parser.add_argument('--reuse-compiled-cache', action='store_true')
    args = parser.parse_args()
    commit = subprocess.check_output(['git', '-C', str(RUNTIME_REPO), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == args.expected_commit
    assert not subprocess.check_output(['git', '-C', str(RUNTIME_REPO), 'status', '--porcelain', '--', '19_table_ocr_serving'], text=True).strip()
    assert subprocess.check_output(['git', '-C', str(HISTORICAL), 'rev-parse', 'HEAD'], text=True).strip() == LOCKED
    if args.reuse_compiled_cache:
        assert (RUNTIME_REPO / '.runtime_cache/19_current_checkpoint_20260913').is_dir()
        for name in ('p04_paddle_ocr_vl_1_6_modeling.py','p05_vision_prefill.py','p06_text_prefill_and_decode.py'):
            relative = '19_table_ocr_serving/' + name
            assert (RUNTIME_REPO / relative).read_bytes() == subprocess.check_output(
                ['git','-C',str(RUNTIME_REPO),'show','ce7a92b1:' + relative])
        phases = ('cached',)
    else:
        # The original checkpoint protocol still requires a genuinely fresh root.
        assert not (RUNTIME_REPO / '.runtime_cache/19_current_checkpoint_20260913').exists()
        phases = ('compile','cached')
    root = HISTORICAL / args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    path = HISTORICAL / '09_persistent_page_engine/scripts/table_poisson_frontier.py'
    source = path.read_text()
    server = CONTAINER_RUNTIME + '/19_table_ocr_serving/p01_serve.py'
    replacements = {
        '"serve_crop_ocr_api.py" in cmd': '"19_table_ocr_serving/p01_serve.py" in cmd',
        'SCRIPTS + "serve_crop_ocr_api.py"': repr(server),
        'self.marker = str(relative / "service.json")': 'self.marker = str(relative / "service_logs")',
        ', "--min-pixels", "28224", "--max-pixels", "802816"': ', *SERVER_ARGS',
        '"--service-summary-output", self.marker': '"--log-folder", self.marker',
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    ns = {'__file__': str(path), '__name__': 'locked_validation_harness'}
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
    ns['MATRIX'] = {8: [6]}
    ns['SERVER_ARGS'] = [
        '--model-path', '/workspace/models/PaddleOCR-VL-1.6',
        '--device', 'npu:0', '--graph-cache-directory', CACHE,
        '--metrics-level', 'scheduling',
    ]
    plan = {
        'runtime_commit': commit, 'client_commit': LOCKED, 'physical_npu': args.npu,
        'batch': 8, 'target_qps': 6, 'requests': 1000, 'seed': 1, 'shuffle_all': True,
        'cache_root': CACHE, 'head_rows': 60416, 'metrics_level': 'scheduling',
        'phases': list(phases),
        'reference': str(REFERENCE),
        'comparison_note': 'Historical reference uses 16k vocabulary and former preprocessing/postprocessing. Report differences; do not claim an isolated refactor speedup.',
    }
    (root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    try:
        for phase in phases:
            sweep = ns['Sweep'](argparse.Namespace(npu=args.npu, count=1000, output_dir=args.output_dir / phase))
            sweep.write('runtime_identity.json', plan)
            try:
                sweep.log('CHECKPOINT phase=' + phase)
                if phase == 'compile':
                    folder, _ = sweep.start(8)
                    ready = json.loads((folder / 'ready.json').read_text())['configuration']
                    assert ready['batch_size'] == 8 and ready['cache_length'] == 4096
                    assert ready['decode_vocab']['selected_vocab_size'] == 60416
                    assert ready['decode_device_timing'] is False
                    assert ready['request_scheduling_metrics'] is True
                    sweep.write('status.json', {'status': 'compile_and_warmup_complete', 'measured_requests': 0})
                else:
                    sweep.run()
            finally:
                # run() already closes its ownership log after a measured run.
                if not sweep.ownership.closed:
                    sweep.stop()
                    sweep.ownership.close()
        relative = Path('b8/qps6/measured/schedule.jsonl')
        original = RUNTIME_REPO / REFERENCE / relative
        measured = root / 'cached' / relative
        assert original.read_bytes() == measured.read_bytes(), 'Saved schedule changed'
        rows = [json.loads(line) for line in measured.read_text().splitlines()]
        order_hash = hashlib.sha256(json.dumps([row['request_id'] for row in rows]).encode()).hexdigest()
        assert order_hash == '97a1f87dd18ace0833f6d66796ce6868845b04880292d0f632f3575c3693caa9'
        (root / 'status.json').write_text(json.dumps({'status': 'complete', 'schedule_byte_identical': True, 'ordered_ids_sha256': order_hash}) + '\n')
    except BaseException as exc:
        (root / 'status.json').write_text(json.dumps({'status': 'failed', 'error': repr(exc)}) + '\n')
        raise
    print('CHECKPOINT COMPLETE; owned servers stopped and NPU released', flush=True)


if __name__ == '__main__':
    main()
