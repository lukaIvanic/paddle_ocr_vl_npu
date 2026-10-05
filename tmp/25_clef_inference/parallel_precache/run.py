"""Partition unfinished documents across explicitly available NPUs and merge."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
for name in ('fixture', 'manifest', 'model', 'work-dir'):
    parser.add_argument('--' + name, type=Path, required=True)
parser.add_argument('--devices', nargs='+', type=int, required=True)
args = parser.parse_args()
assert len(set(args.devices)) == len(args.devices) and 6 not in args.devices
repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / '25_clef_inference'))
from run_reranking_smoke import digest, save, sync_saved
from run_local_smoke import encode_document_prefix
from tokenizers import Tokenizer

args.work_dir.mkdir(parents=True, exist_ok=True)
plan_path = args.work_dir / 'plan.json'
seed_path = args.work_dir / 'seed.json'
fixture = json.loads(args.fixture.read_text())
if not plan_path.exists():
    seed = json.loads(args.manifest.read_text())
    assert seed['contract']['fixture_sha256'] == digest(args.fixture)
    assert seed['contract']['dtype'] == 'float32' and seed['contract']['storage_dtype'] == 'bfloat16'
    assert set(seed['documents']).issubset(fixture['documents'])
    cache_dir = Path(seed['contract']['cache_dir'])
    for row in seed['documents'].values():
        assert (cache_dir / (row['key'] + '.safetensors')).stat().st_size == row['file_bytes']
    # Identical token prefixes stay in one shard to avoid file-name collisions.
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    groups = {}
    for did, doc in fixture['documents'].items():
        if did not in seed['documents']:
            prefix = tuple(encode_document_prefix(tokenizer, doc['text']))
            groups.setdefault(prefix, []).append(did)
    shards = [[] for _ in args.devices]
    loads = [0 for _ in args.devices]
    for prefix, ids in sorted(groups.items(), key=lambda x: (-len(x[0]), x[1][0])):
        index = min(range(len(loads)), key=lambda i: (loads[i], i))
        shards[index].extend(ids)
        loads[index] += len(ids) * (len(prefix) + 128)
    flat = [d for shard in shards for d in shard]
    assert len(flat) == len(set(flat))
    assert set(flat) | set(seed['documents']) == set(fixture['documents'])
    assert not set(flat) & set(seed['documents'])
    save(seed_path, seed)
    sync_saved(seed_path)
    plan = {'fixture_sha256': digest(args.fixture), 'seed_sha256': digest(seed_path),
            'devices': args.devices, 'shards': shards, 'work_estimates': loads,
            'already_completed': len(seed['documents']), 'remaining': len(flat)}
    save(plan_path, plan)
    sync_saved(plan_path)
else:
    plan = json.loads(plan_path.read_text())
    assert plan['devices'] == args.devices and plan['fixture_sha256'] == digest(args.fixture)
    assert plan['seed_sha256'] == digest(seed_path)
seed = json.loads(seed_path.read_text())
print(json.dumps({k:v for k,v in plan.items() if k!='shards'}), flush=True)
status = {'status': 'running', 'started_unix': time.time(), 'devices': args.devices,
          'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), 'workers': []}
save(args.work_dir / 'status.json', status)
processes = []
for index, device in enumerate(args.devices):
    ids_path = args.work_dir / f'documents-{device}.json'
    save(ids_path, plan['shards'][index])
    output = args.work_dir / f'manifest-{device}.json'
    command = [sys.executable, '-u', str(repo / '25_clef_inference/run_reranking_smoke.py'),
        'precache-task', '--fixture', str(args.fixture), '--model', str(args.model),
        '--dtype', 'float32', '--storage-dtype', 'bfloat16',
        '--max-prefix-length', str(seed['contract']['max_prefix_length']),
        '--cache-dir', seed['contract']['cache_dir'], '--output', str(output),
        '--document-ids', str(ids_path)]
    save(args.work_dir / f'command-{device}.json', {'device': device, 'argv': command})
    env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(device))
    with (args.work_dir / f'worker-{device}.log').open('a') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    processes.append((device, process))
    status['workers'].append({'device': device, 'pid': process.pid, 'documents': len(plan['shards'][index])})
    save(args.work_dir / 'status.json', status)
failed = []
for device, process in processes:
    code = process.wait()
    (args.work_dir / f'exit-{device}.txt').write_text(str(code)+'\n')
    if code:
        failed.append(device)
if failed:
    status.update(status='failed', failed_devices=failed)
    save(args.work_dir / 'status.json', status)
    raise RuntimeError('Failed workers: ' + str(failed))
merged = dict(seed)
merged['documents'] = dict(seed['documents'])
for index, device in enumerate(args.devices):
    part = json.loads((args.work_dir / f'manifest-{device}.json').read_text())
    assert part['status'] == 'completed' and part['model_identity'] == seed['model_identity']
    assert set(part['documents']) == set(plan['shards'][index])
    assert not set(part['documents']) & set(merged['documents'])
    merged['documents'].update(part['documents'])
assert set(merged['documents']) == set(fixture['documents'])
for row in merged['documents'].values():
    assert (Path(seed['contract']['cache_dir'])/(row['key']+'.safetensors')).stat().st_size == row['file_bytes']
merged.pop('error', None)
merged.pop('physical_device', None)
merged.pop('elapsed_this_run_s', None)
merged.update(status='completed', physical_devices=args.devices, completed_documents=len(merged['documents']),
    parallel_run=str(args.work_dir), cache_bytes=sum(r['cache_bytes'] for r in merged['documents'].values()),
    file_bytes=sum(r['file_bytes'] for r in merged['documents'].values()))
save(args.manifest, merged)
sync_saved(args.manifest)
status.update(status='completed', finished_unix=time.time(), completed_documents=len(merged['documents']))
save(args.work_dir/'status.json', status)
sync_saved(args.work_dir/'status.json')
print(json.dumps(status), flush=True)
