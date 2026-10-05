"""Resume interrupted shard 3, retain other live shards, then validate and merge."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

repo=Path('/workspace/repos/clef-reranking-smoke')
sys.path.insert(0,str(repo/'25_clef_inference'))
from run_reranking_smoke import save,sync_saved
root=Path('/workspace/results/clef_touche_full')
work=root/'parallel_e8c66ee5'
plan=json.loads((work/'plan.json').read_text())
seed=json.loads((work/'seed.json').read_text())
fixture=json.loads((root/'fixture.json').read_text())
status=json.loads((work/'status.json').read_text())
status.update(status='running', resumed_after_profiling=True)
command=json.loads((work/'command-3.json').read_text())['argv']
with (work/'worker-3.log').open('a') as log:
    process=subprocess.Popen(command,env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES='3'),stdout=log,stderr=subprocess.STDOUT,cwd=repo)
status['workers']=[dict(r,pid=process.pid) if r['device']==3 else r for r in status['workers']]
save(work/'status.json',status)
code=process.wait()
(work/'exit-3.txt').write_text(str(code)+'\n')
try:
    assert code==0, f'Resumed shard failed: {code}'
    while True:
        parts=[json.loads((work/f'manifest-{i}.json').read_text()) for i in plan['devices']]
        assert not any(p['status']=='failed' for p in parts),'A cache shard failed'
        if all(p['status']=='completed' for p in parts): break
        time.sleep(10)
    merged=dict(seed)
    merged['documents']=dict(seed['documents'])
    for index,part in enumerate(parts):
        assert part['model_identity']==seed['model_identity']
        assert set(part['documents'])==set(plan['shards'][index])
        assert not set(part['documents']) & set(merged['documents'])
        merged['documents'].update(part['documents'])
    assert set(merged['documents'])==set(fixture['documents'])
    for row in merged['documents'].values():
        assert (Path(seed['contract']['cache_dir'])/(row['key']+'.safetensors')).stat().st_size==row['file_bytes']
    for key in ('error','physical_device','elapsed_this_run_s'): merged.pop(key,None)
    merged.update(status='completed',physical_devices=plan['devices'],completed_documents=len(merged['documents']),
        parallel_run=str(work),cache_bytes=sum(r['cache_bytes'] for r in merged['documents'].values()),
        file_bytes=sum(r['file_bytes'] for r in merged['documents'].values()))
    save(root/'bf16_storage-manifest.json',merged)
    sync_saved(root/'bf16_storage-manifest.json')
    status.update(status='completed',finished_unix=time.time(),completed_documents=len(merged['documents']))
except BaseException as exc:
    status.update(status='failed',error=repr(exc))
    raise
finally:
    save(work/'status.json',status)
    sync_saved(work/'status.json')
print(json.dumps(status),flush=True)
