"""Summarize observed CPU-blocked prefill, never subtract it from E2E latency."""
import json
import math
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASELINE=ROOT.parent/'lm_head_60416_20260911'

def stats(values):
    v=sorted(values)
    def q(p):
        pos=(len(v)-1)*p; lo=int(pos); hi=math.ceil(pos)
        return v[lo]+(v[hi]-v[lo])*(pos-lo)
    return dict(count=len(v),positive=sum(x>0 for x in v),over_1ms=sum(x>.001 for x in v),
        mean=sum(v)/len(v),p50=q(.5),p90=q(.9),p95=q(.95),p99=q(.99),max=max(v),total=sum(v))

results=[]; per_request=[]
for b in (2,8):
    folder=ROOT/f'b{b}_expanded_measured'/f'b{b}'
    measured=folder/'measured/results'
    records=[json.loads(x) for x in (measured/'results.jsonl').read_text().splitlines()]
    control={r['sequence']:r for r in (json.loads(x) for x in
        (BASELINE/f'b{b}_expanded_measured'/f'b{b}'/'measured/results/results.jsonl').read_text().splitlines())}
    summary=json.loads((measured/'summary.json').read_text())
    by_metric={}; rows=[]; mismatches=[]; stops=Counter()
    assert len(records)==100
    for r in records:
        p=r['service_result']['response']; c=p['scheduling_metrics']['cpu_readiness']; t=p['timing_s']
        assert abs(c['prefill_blocked_s']-c['blocked_cpu_queue_s']-c['blocked_cpu_service_s'])<1e-6
        assert c['scheduler_idle_blocked_s']<=c['prefill_blocked_s']+1e-6
        for k,v in c.items():
            if k.endswith('_s') and not k.endswith('offset_s'): by_metric.setdefault(k,[]).append(v)
        for k in ('cpu_preprocess_background_service','cpu_preprocess_background_queue_wait',
                  'cpu_preprocess_background_consumer_wait'):
            by_metric.setdefault(k,[]).append(t[k])
        prev=control[r['sequence']]
        assert prev['request_id']==r['request_id']
        before=prev['service_result']['response']
        for k in ('input_tokens','projected_image_tokens','crop_size'): assert p[k]==before[k]
        assert p['stop_reason']==before['stop_reason']
        stops[p['stop_reason']]+=1
        if p['token_ids']!=before['token_ids']: mismatches.append(r['request_id'])
        row=dict(batch=b,sequence=r['sequence'],request_id=r['request_id'],latency_s=r['latency_s'],
            cpu_service_s=t['cpu_preprocess_background_service'],
            cpu_queue_s=t['cpu_preprocess_background_queue_wait'],**c)
        rows.append(row); per_request.append(row)
    affected=[r['prefill_blocked_s'] for r in rows if r['prefill_blocked_s']>.001]
    tail=sorted(rows,key=lambda r:r['latency_s'],reverse=True)[:5]
    own=[json.loads(x) for x in (folder.parent/'ownership.jsonl').read_text().splitlines()]
    assert all(set(r['pids'])<=set(r['owned']) for r in own)
    assert own[-1]['pids']==[]
    results.append(dict(batch=b,request_count=len(records),errors=summary['failed_request_count'],
        tables_s=summary['completion_qps'],latency_s=summary['latency_s'],
        metrics={k:stats(v) for k,v in by_metric.items()},
        over_1ms_only=stats(affected) if affected else None,
        slowest_e2e_five=tail,largest_cpu_blocks=sorted(rows,key=lambda r:r['prefill_blocked_s'],reverse=True)[:5],
        native_token_mismatches=mismatches,clean_ownership_snapshots=len(own),
        completion_reasons=dict(stops),
        slowest_e2e_five_cpu_block_s=stats([r['prefill_blocked_s'] for r in tail]),
        no_active_decode_cpu_wait_fraction=sum(r['scheduler_idle_blocked_s'] for r in rows)/summary['run_wall_s']))
(ROOT/'per_request.json').write_text(json.dumps(per_request,indent=2)+'\n')
(ROOT/'analysis.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
