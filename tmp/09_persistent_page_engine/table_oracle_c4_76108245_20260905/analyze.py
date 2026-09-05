"""Audit actual four-slot-server C2/C4 runs; no simulated timings."""
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime

ROOT=Path(__file__).resolve().parent
REFERENCE=ROOT.parent/'table_oracle_c2_a15a5b6d_20260905/t1024'
def rows(path):
    return {r['request_id']:r for r in map(json.loads,(path/'results.jsonl').read_text().splitlines())}
reference=rows(REFERENCE)
ref_summary=json.loads((REFERENCE/'summary.json').read_text())
digest=hashlib.sha256((REFERENCE/'tables.jsonl').read_bytes()).hexdigest()
results={}
server_log=(ROOT/'server.log').read_text()
assert 'Traceback (most recent call last)' not in server_log
for name,concurrency in [('c2',2),('c4',4)]:
    path=ROOT/name
    if not (path/'summary.json').exists():continue
    summary=json.loads((path/'summary.json').read_text()); data=rows(path)
    assert len(data)==100 and data.keys()==reference.keys()
    assert hashlib.sha256((path/'tables.jsonl').read_bytes()).hexdigest()==digest
    assert summary['dispatch_request_ids']==ref_summary['dispatch_request_ids']
    active=peak=0
    for _,delta in sorted([(r['dispatch_offset_s'],1) for r in data.values()]+[(r['completion_offset_s'],-1) for r in data.values()]):
        active+=delta; peak=max(peak,active)
        assert 0<=active<=concurrency
    assert active==0 and peak==concurrency
    admissions=list(map(int,re.findall(r'TABLE_PHASE preparing id=oracle-c4-cap-'+name+r'-\d{3}-\S+ admitted=(\d+)',server_log)))
    assert len(admissions)==100 and max(admissions)==concurrency and min(admissions)>=1
    details=[]
    for key,r in data.items():
        assert r['status']=='ok'
        s=r['service_result']['response']; old=reference[key]['service_result']['response']
        assert s['route_lane']==old['route_lane']
        assert s['runtime_metrics']['routing']==old['runtime_metrics']['routing']
        details.append(dict(request_id=key,route=s['route_lane'],latency_s=r['latency_s'],
            previous_c2_s=reference[key]['latency_s'],stop_reason=s['stop_reason'],
            output_tokens=len(s['token_ids']),old_output_tokens=len(old['token_ids']),
            native_token_identical=s['token_ids']==old['token_ids'],
            own_actions=s['runtime_metrics']['phase_accounting']['own_action_wall_s'],
            foreign_actions=s['runtime_metrics']['phase_accounting']['other_action_wait_s']))
    results[name]=dict(completion_qps=summary['completion_qps'],latency_s=summary['latency_s'],
        native_token_identical=sum(x['native_token_identical'] for x in details),
        all_eos=all(x['stop_reason']=='eos' for x in details),
        over_2s=sum(x['latency_s']>2 for x in details), spec_requests=sum(x['route']=='spec' for x in details),
        peak_outstanding=peak, selection_sha256=digest, per_table=details)
    results[name]['server_admission_records']=len(admissions)
    results[name]['peak_whole_pipeline_admission']=max(admissions)
if 'c2' in results and 'c4' in results:
    control={x['request_id']:x for x in results['c2']['per_table']}
    for x in results['c4']['per_table']:
        x['same_server_c2_s']=control[x['request_id']]['latency_s']
        x['delta_vs_same_server_c2_s']=x['latency_s']-x['same_server_c2_s']
if (ROOT/'host_npu6_monitor.log').exists():
    samples=[]
    for block in re.split(r'(?=^2026-\d\d-\d\dT)',(ROOT/'host_npu6_monitor.log').read_text(),flags=re.M):
        if 'Chip Count' not in block:continue
        samples.append((datetime.fromisoformat(block.splitlines()[0]).timestamp(),set(map(int,re.findall(r'Process id:(\d+)',block)))))
    for name,run in results.items():
        summary=json.loads((ROOT/name/'summary.json').read_text())
        begin=summary['actual_start_epoch_s'];end=begin+summary['run_wall_s']
        during=[p for t,p in samples if begin<=t<=end]
        assert samples[0][0]<=begin and samples[-1][0]>=end
        assert during and all(p=={1728658} for p in during)
        run['only_owned_npu_pid_in_samples']=True
        run['measured_npu_samples']=len(during)
(ROOT/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
for name,run in results.items():
    print(name,json.dumps({k:v for k,v in run.items() if k!='per_table'}))
    print('TAIL',[(x['request_id'],round(x['latency_s'],3)) for x in sorted(run['per_table'],key=lambda x:-x['latency_s'])[:10]])
