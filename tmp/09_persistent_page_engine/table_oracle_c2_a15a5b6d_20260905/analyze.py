"""Audit real C2 oracle runs and compare identical requests; no latency simulation."""
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
OLD = PARENT/'table_phase_reference_6099805a_20260905/async_c2'
B1 = PARENT/'table_closed_loop_random100_seed1_3a745ba_20260903/b1'
B2 = B1.parent/'b2'
COUNTS = json.loads((PARENT/'table_token_oracle_screen_20260905/oracle_counts.json').read_text())['token_counts']

def records(path):
    return {r['request_id']:r for r in map(json.loads,(path/'results.jsonl').read_text().splitlines())}

old, b1, b2 = records(OLD), records(B1), records(B2)
expected_hash = hashlib.sha256((OLD/'tables.jsonl').read_bytes()).hexdigest()
expected_order = json.loads((OLD/'summary.json').read_text())['dispatch_request_ids']
runs = {}
audits = {}
monitor = (ROOT/'host_npu6_monitor.log').read_text()
snapshots = []
for block in re.split(r'(?=^2026-\d\d-\d\dT)', monitor, flags=re.M):
    if 'Chip Count' not in block:
        continue
    stamp = datetime.fromisoformat(block.splitlines()[0]).timestamp()
    snapshots.append((stamp, set(map(int,re.findall(r'Process id:(\d+)',block)))))
owners = {'control':1689638, 't512':1704968, 't1024':1713251}
control_service = json.loads((ROOT/'control_service.json').read_text())
def graph_contract(graphs):
    return {name:{k:v for k,v in values.items() if k not in ('compile_first_call_s','compile_wrapper_s')}
            for name,values in graphs.items()}
for name, threshold in [('control', None), ('t512',512), ('t1024',1024)]:
    path = ROOT/name
    if not (path/'summary.json').exists():
        continue
    summary, data = json.loads((path/'summary.json').read_text()), records(path)
    service = json.loads((ROOT/(name+'_service.json')).read_text())
    assert graph_contract(service['summary']['graph_contracts']) == graph_contract(control_service['summary']['graph_contracts'])
    begin=summary['actual_start_epoch_s']; end=begin+summary['run_wall_s']
    during=[pids for stamp,pids in snapshots if begin <= stamp <= end]
    assert snapshots[0][0] <= begin and snapshots[-1][0] >= end
    assert during and all(pids == {owners[name]} for pids in during)
    log=(ROOT/(name+'_server.log')).read_text()
    assert not any(marker in log for marker in ('Traceback (most recent call last)', 'Skip cache'))
    admitted=list(map(int,re.findall(r'TABLE_PHASE preparing id=oracle-'+name+r'-\d{3}-\S+ admitted=(\d+)',log)))
    assert len(admitted)==100 and max(admitted)==2 and min(admitted)>=1
    audits[name]={'same_graph_contract_excluding_setup_durations':True,
                 'whole_pipeline_admission_records':len(admitted), 'peak_admitted':max(admitted),
                 'only_owned_npu_pid_in_measured_samples':True, 'measured_samples':len(during),
                 'host_worker_pid':owners[name], 'monitor_covers_measured_window':True}
    assert set(data) == set(old) and len(data)==100
    assert hashlib.sha256((path/'tables.jsonl').read_bytes()).hexdigest()==expected_hash
    assert summary['dispatch_request_ids']==expected_order
    events = sorted([(r['dispatch_offset_s'],1) for r in data.values()] +
                    [(r['completion_offset_s'],-1) for r in data.values()])
    active, peak = 0, 0
    for _, delta in events:
        active += delta
        assert 0 <= active <= 2
        peak=max(peak,active)
    assert active==0 and peak==2
    details=[]
    for k,r in data.items():
        assert r['status']=='ok' and r['service_result']['stop_reason']=='eos'
        response=r['service_result']['response']
        original_spec=old[k]['service_result']['response']['route_lane']=='spec'
        want_spec=original_spec and (threshold is None or COUNTS[k]>=threshold)
        assert response['route_lane']==('spec' if want_spec else 'b1')
        route=response['runtime_metrics']['routing']
        assert route['oracle_min_output_tokens']==threshold
        assert route['oracle_b1_output_tokens']==(None if threshold is None else COUNTS[k])
        if not want_spec:
            assert not response['runtime_metrics']['draft_rows']
            assert response['runtime_metrics']['row_preparation'] is None
        reference=old[k] if want_spec or not original_spec else b1[k]
        same_current = response['token_ids']==old[k]['service_result']['response']['token_ids']
        assert same_current
        details.append(dict(request_id=k, oracle_tokens=COUNTS[k], route=response['route_lane'],
            latency_s=r['latency_s'], previous_c2_s=old[k]['latency_s'],
            regular_b2_s=b2[k]['latency_s'],
            native_parity_current_c2=same_current,
            native_parity_route_reference=response['token_ids']==reference['service_result']['response']['token_ids']))
    runs[name]=dict(completion_qps=summary['completion_qps'], latency_s=summary['latency_s'],
        spec_requests=sum(x['route']=='spec' for x in details),
        over_2s=sum(x['latency_s']>2 for x in details),
        native_parity_route_reference=sum(x['native_parity_route_reference'] for x in details),
        native_parity_current_c2=sum(x['native_parity_current_c2'] for x in details),
        all_eos=True, peak_client_requests=peak, selection_sha256=expected_hash,
        per_table=details)

if 'control' in runs:
    control={x['request_id']:x for x in runs['control']['per_table']}
    for name,run in runs.items():
        for x in run['per_table']:
            x['control_c2_s']=control[x['request_id']]['latency_s']
            x['delta_vs_control_s']=x['latency_s']-x['control_c2_s']
        run['faster_than_control']=sum(x['delta_vs_control_s']<0 for x in run['per_table'])
        run['faster_by_over_50ms']=sum(x['delta_vs_control_s']<-.05 for x in run['per_table'])
        run['slower_by_over_50ms']=sum(x['delta_vs_control_s']>.05 for x in run['per_table'])
(ROOT/'comparison.json').write_text(json.dumps(runs,indent=2)+'\n')
(ROOT/'audit.json').write_text(json.dumps({'checks':audits,'all_checks_pass':True,
    'limitations':['NPU process sampling every five seconds, not continuous tracing.',
                   'Single measured run per threshold, not a repeatability guarantee.',
                   'Oracle uses saved historical B1 lengths, not a measured image estimator.']},indent=2)+'\n')
for name,run in runs.items():
    print(name,json.dumps({k:v for k,v in run.items() if k!='per_table'}))
    print('TAIL',json.dumps(sorted((x for x in run['per_table'] if x['latency_s']>2),key=lambda x:-x['latency_s'])))
