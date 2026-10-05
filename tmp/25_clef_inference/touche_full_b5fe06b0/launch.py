"""Host-side coordinator: score on 0.23, evaluate with pinned MTEB on 0.21."""
import json
from pathlib import Path
import shlex
import subprocess
import time

runtime = Path('/data1/lukaiv/clef_vllm023_20261005/workspace/results/clef_touche_full/benchmark_b5fe06b0')
runtime.mkdir(parents=True, exist_ok=True)
repo = '/workspace/repos/clef-touche-b5fe06b0'
eval_repo = '/workspace/repos/clef-touche-f268a541'
root = '/workspace/results/clef_touche_full'
container = 'research_vllm_ascend_023_external_workspace'
evaluator = 'research_vllm_ascend_021_external_workspace'
output = root + '/benchmark_b5fe06b0/result.json'
argv = ['/usr/local/python3.12.13/bin/python3', '-u', '25_clef_inference/run_reranking_task.py',
        'score', '--fixture', root + '/fixture.json', '--model', '/workspace/models/clef-flash',
        '--cache-dir', root + '/bf16_storage', '--cache-manifest', root + '/bf16_storage-manifest.json',
        '--output', output]
command = ['docker', 'exec', container, 'bash', '-c',
           'source npu-setup || exit $?; cd ' + shlex.quote(repo) + ' && exec ' + shlex.join(argv)]
status = {'status': 'scoring', 'commit': 'b5fe06b02898561a691f324839185065ffa486d9',
          'started_unix': time.time(), 'score_command': command}

def save_status():
    partial = runtime / 'status.partial'
    partial.write_text(json.dumps(status, indent=2) + '\n')
    partial.replace(runtime / 'status.json')

def run(args, name):
    with (runtime / (name + '.log')).open('a') as log:
        p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        status[name + '_pid'] = p.pid
        save_status()
        code = p.wait()
    (runtime / (name + '-exit_code.txt')).write_text(str(code) + '\n')
    if code:
        raise RuntimeError(name + ' exited ' + str(code))

try:
    save_status()
    run(command, 'score')
    status['status'] = 'evaluating'
    save_status()
    target = eval_repo + '/tmp/25_clef_inference/touche_full_f268a541/restarted-result.json'
    subprocess.run(['docker', 'cp', str(runtime / 'result.json'), evaluator + ':' + target], check=True)
    eval_command = ['docker', 'exec', evaluator, 'bash', '-c',
                    'cd ' + shlex.quote(eval_repo) + ' && exec ' + shlex.join([
                        '/workspace/venvs/qwen3_embedding_eval_py312/bin/python',
                        '25_clef_inference/run_reranking_task.py', 'evaluate',
                        '--fixture', root + '/fixture.json', '--output', target])]
    status['evaluate_command'] = eval_command
    run(eval_command, 'evaluate')
    subprocess.run(['docker', 'cp', evaluator + ':' + target, str(runtime / 'evaluated.json')], check=True)
    Path(runtime / 'evaluated.json').replace(runtime / 'result.json')
    status['status'] = 'completed'
except BaseException as exc:
    status.update(status='failed', error=repr(exc))
    raise
finally:
    status['finished_unix'] = time.time()
    save_status()
