"""Step-3 validation: frozen clients and scheduler harness, cleaned server.

Run compile-warm first, then measured in a new process using the cached graphs.
Only server path/removed CLI selectors and artifact ownership names are adapted.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess

HISTORICAL = Path('/data1/lukaiv/workspace/repos/table_step1_be691de1_20260910')
RUNTIME_REPO = Path('/data1/lukaiv/workspace/repos/paddle_ocr_vl_npu')
RUNTIME = RUNTIME_REPO / '19_table_ocr_serving'
SERVER = '/workspace/repos/paddle_ocr_vl_npu/19_table_ocr_serving/serve.py'
PIN = '0976fa33'
BASE = 'tmp/19_table_ocr_serving/step3_b8qps6_0976fa33_20260911'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--phase', choices=('compile-warm', 'measured'), required=True)
args = parser.parse_args()
assert subprocess.check_output(['git', '-C', str(RUNTIME_REPO), 'rev-parse', '--short=8', 'HEAD'], text=True).strip() == PIN
assert subprocess.check_output(['git', '-C', str(HISTORICAL), 'rev-parse', 'HEAD'], text=True).strip() == 'be691de190ae099d1a9b0ba80865006b122ecc00'
assert not subprocess.check_output(['git', '-C', str(RUNTIME_REPO), 'status', '--porcelain', '--', '19_table_ocr_serving'], text=True).strip()
path = HISTORICAL / '09_persistent_page_engine/scripts/table_poisson_frontier.py'
source = path.read_text()
replacements = {
    '"serve_crop_ocr_api.py" in cmd': '"19_table_ocr_serving/serve.py" in cmd',
    'SCRIPTS + "serve_crop_ocr_api.py"': repr(SERVER),
    ', "--min-pixels", "28224", "--max-pixels", "802816"': '',
    'cached production setup, then one full request warmup': 'production setup, then one full request warmup',
}
for old, new in replacements.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
ns = {'__file__': str(path), '__name__': 'historical_frontier'}
exec(compile(source, str(path), 'exec'), ns)
original_fingerprint = ns['fingerprint']


def fingerprint(repo):
    digest = hashlib.sha256(original_fingerprint(repo).encode())
    for item in sorted(RUNTIME.rglob('*')):
        if item.suffix in {'.py', '.json'}:
            digest.update(str(item.relative_to(RUNTIME)).encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()


ns['MATRIX'] = {8: [6]}
ns['CONTAINER_REPO'] = '/workspace/repos/table_step1_be691de1_20260910'
ns['fingerprint'] = fingerprint
artifact = Path(BASE + ('_compile' if args.phase == 'compile-warm' else '_cached'))
sweep = ns['Sweep'](argparse.Namespace(npu=6, count=1000, output_dir=artifact))
(sweep.root / 'driver.py').write_bytes(Path(__file__).read_bytes())
sweep.write('relocated_runtime.json', {
    'runtime_git_commit': subprocess.check_output(['git', '-C', str(RUNTIME_REPO), 'rev-parse', 'HEAD'], text=True).strip(),
    'historical_client_and_harness_commit': 'be691de190ae099d1a9b0ba80865006b122ecc00',
    'phase': args.phase, 'runtime_source': str(RUNTIME), 'server_entrypoint': SERVER,
    'expected_ordered_request_id_sha256': '97a1f87dd18ace0833f6d66796ce6868845b04880292d0f632f3575c3693caa9',
    'changes': 'Entrypoint/ownership names and fingerprint adapted; removed min/max CLI flags now hardcoded to the same pixel bounds. Frozen clients and warmup unchanged.'
})
sweep.log('STEP3 ' + args.phase + ': B8 / 6 QPS / 1000; physical NPU6')
if args.phase == 'compile-warm':
    try:
        sweep.start(8)
        sweep.write('status.json', {'status': 'compile_and_real_warmup_complete', 'measured_requests': 0})
    except BaseException as exc:
        sweep.write('status.json', {'status': 'failed', 'error': repr(exc)})
        raise
    finally:
        sweep.stop()
        sweep.ownership.close()
else:
    sweep.run()
