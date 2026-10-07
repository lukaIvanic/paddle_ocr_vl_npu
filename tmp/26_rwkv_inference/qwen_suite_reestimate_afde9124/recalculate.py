from pathlib import Path
import json,gzip,collections,bisect,statistics,hashlib
root=Path('/home/luka/.codex/worktrees/rwkv-reference/paddle_ocr_vl_npu/tmp/26_rwkv_inference')
new=root/'nanobeir_large_bucketed_afde9124/probe'
old=json.loads((root/'qwen_suite_probe_67a5f3be/runtime_estimate.json').read_text())
audit=json.loads((root/'qwen_suite_probe_67a5f3be/prepared/audit.json').read_text())
lengths=json.loads(Path('/tmp/rwkv_suite_sample_lengths.json').read_text())
bylength=collections.defaultdict(list)
for p in new.glob('worker_*/batch_timings.jsonl.gz'):
 for line in gzip.open(p,'rt'):
  x=json.loads(line);bylength[x['length']].append(x['forward_and_transfer_seconds'])
report=json.loads((new/'result.json').read_text())
ratio=sum(w['scoring_wall_seconds'] for w in report['workers'])/sum(w['forward_and_transfer_seconds'] for w in report['workers'])
def lookup(L):
 points=sorted((n,statistics.median(v)) for n,v in bylength.items() if (n<=512)==(L<=512))
 xs=[p[0] for p in points];i=bisect.bisect_left(xs,L)
 if i==0:return points[0][1]*ratio
 if i==len(points):return points[-1][1]*ratio
 a,x=points[i-1];b,y=points[i]
 return (x+(y-x)*(L-a)/(b-a))*ratio
out=dict(method='Interpolate median measured B4 batch time by exact prepared length within each of T512/T2048; multiply by observed NanoBEIR scoring-loop/timed-batch ratio. Apply to two saved real top-100 queries per task and expand by full query count. This is a cross-workload estimate, not a new MTEB timing run.',source_run='afde9124',sample_source='67a5f3be',scoring_loop_ratio=ratio,tasks=[],suites={})
for t in old['tasks']:
 js=[j for j in lengths if j['task']==t['task']];assert len(js)==2
 per_query=[sum(lookup(L) for L in j['batch_lengths']) for j in js]
 device_s=sum(per_query)/2*t['pairs']/100
 fraction_long=sum(L>512 for j in js for L in j['batch_lengths'])/50
 row=dict(task=t['task'],suite=t['suite'],pairs=t['pairs'],estimated_scoring_seconds=device_s,cpu_prepare_seconds=t['estimated_cpu_prepare_seconds'],sample_query_projected_seconds=per_query,sample_fraction_t2048=fraction_long)
 out['tasks'].append(row)
for suite in ['english','chinese']:
 ts=[t for t in out['tasks'] if t['suite']==suite];s=sum(t['estimated_scoring_seconds'] for t in ts);cpu=sum(t['cpu_prepare_seconds'] for t in ts)
 au=sum(t['audit_seconds'] for t in audit['tasks'] if t['suite']==suite)
 setup=max(w['setup_and_preflight_seconds'] for w in report['workers'])
 # Conservative accounting: no CPU preparation overlap or two-device CPU speedup assumed.
 overhead=cpu+au+setup+60
 out['suites'][suite]=dict(pairs=sum(t['pairs'] for t in ts),one_npu_scoring_hours=s/3600,two_npu_scoring_hours=s/7200,estimated_overhead_seconds=overhead,one_npu_total_hours=(s+overhead)/3600,two_npu_total_hours=(s/2+overhead)/3600,two_npu_planning_30pct_hours=(s/2+overhead)/3600*1.3)
out['limitations']=['Only two sampled queries per task; 30% planning allowance is not a confidence interval.','Two-NPU estimate assumes balanced independent model replicas and ideal inference scaling.','Latest measured kernel/runtime timings come from NanoBEIR, mapped by shape; no new MTEB/C-MTEB inference was executed.','FP16 dense projections, FP32 recurrence, LayerNorm, B4, unchanged TorchAir configuration, T512/T2048. No document caching.','CPU preparation extrapolated from prior suite sample; audit from prior measured full audit, setup from latest run, plus an unmeasured 60-second output/metric allowance per suite. CPU work not assumed to overlap.','Released RWKV last2048 truncation and per-query logical B32 padding retained, unlike Qwen8192.']
Path('/tmp/rwkv_suite_reestimate.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
