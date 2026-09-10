"""Host orchestration receipt: original clients/monitor, relocated server only."""
import argparse
import hashlib
from pathlib import Path
import subprocess

HISTORICAL = Path('/data1/lukaiv/workspace/repos/table_step1_be691de1_20260910')
RUNTIME_REPO = Path('/data1/lukaiv/workspace/repos/paddle_ocr_vl_npu')
RUNTIME = RUNTIME_REPO / '19_table_ocr_serving'
SERVER = '/workspace/repos/paddle_ocr_vl_npu/19_table_ocr_serving/serve.py'
PIN = 'dc755584'
ARTIFACT = Path('tmp/19_table_ocr_serving/step2_b8qps6_dc755584_20260910_cached')

assert subprocess.check_output(['git', '-C', str(RUNTIME_REPO), 'rev-parse', '--short=8', 'HEAD'], text=True).strip() == PIN
path = HISTORICAL / '09_persistent_page_engine/scripts/table_poisson_frontier.py'
source = path.read_text()
# The ownership matcher must recognize the new entrypoint, not another service.
assert source.count('"serve_crop_ocr_api.py" in cmd') == 1
source = source.replace('"serve_crop_ocr_api.py" in cmd', '"19_table_ocr_serving/serve.py" in cmd')
assert source.count('SCRIPTS + "serve_crop_ocr_api.py"') == 1
source = source.replace('SCRIPTS + "serve_crop_ocr_api.py"', repr(SERVER))
ns = {'__file__': str(path), '__name__': 'historical_frontier'}
exec(compile(source, str(path), 'exec'), ns)
original_fingerprint = ns['fingerprint']


def fingerprint(repo):
    digest = hashlib.sha256(original_fingerprint(repo).encode())
    for path in sorted(RUNTIME.rglob('*')):
        if path.suffix in {'.py', '.json'}:
            digest.update(str(path.relative_to(RUNTIME)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


ns['MATRIX'] = {8: [6]}
ns['CONTAINER_REPO'] = '/workspace/repos/table_step1_be691de1_20260910'
ns['fingerprint'] = fingerprint
sweep = ns['Sweep'](argparse.Namespace(npu=6, count=1000, output_dir=ARTIFACT))
sweep.write('relocated_runtime.json', {
    'runtime_git_commit': subprocess.check_output(['git', '-C', str(RUNTIME_REPO), 'rev-parse', 'HEAD'], text=True).strip(),
    'historical_client_and_harness_commit': 'be691de190ae099d1a9b0ba80865006b122ecc00',
    'runtime_source': str(RUNTIME), 'server_entrypoint': SERVER,
    'changes': 'Orchestration only: entrypoint/ownership name, working directory, selected point and source fingerprint. Historical clients and schedule generation unchanged.'
})
print('STEP2: unchanged historical benchmark clients, relocated experiment-19 server; B8 / 6 QPS / 1000.', flush=True)
sweep.run()
