from pathlib import Path
import json,gzip,hashlib,csv,io,math
root=Path('/home/luka/.codex/worktrees/rwkv-reference/paddle_ocr_vl_npu')
e=root/'tmp/26_rwkv_inference/reranker_largest_profile_b4_57c83b92'
assert (e/'exit_code.txt').read_text().strip()=='0'
p=e/'profile_comparison.json';d=json.loads(p.read_text())
d['observer_effect']='Shape/stack capture substantially inflates raw-eager host scopes/device gaps. Do not interpret those gaps as unprofiled serving stalls. Use clean synchronized timings for latency. Device kernel sums are checked against clean forward within 1%; CPU D2H markers include queued forward waits.'
d['launch_observation']='All four eager traces launch RwkvEndpointWkv7 with 48 vector blocks; all four TorchAir traces launch it with 24, with identical shape/dtype/ND format per length. Tiling source uses generic PlatformAscendC.GetCoreNum then SetBlockDim. Installed CANN exposes GetCoreNumAiv, but no kernel/tiling fix or causal intervention has been tested.'
for T,r in d['runs'].items():
 report=json.loads((e/('t'+T)/'result.json').read_text());assert report['all_checks_passed'] and report['source_commit']==d['source_commit']
 assert report['batch_size']==4 and report['physical_npu']=='1' and not report['shared_device']
 for label,v in r['profiles'].items():
  base=e/('t'+T)/('profile_'+label)
  for key in ['kernel_details','operator_details','step_trace_time','api_statistic']:
   raw=gzip.decompress((base/(key+'.csv.gz')).read_bytes())
   assert hashlib.sha256(raw).hexdigest()==v['trace_sha256'][key]
  rows=list(csv.DictReader(io.StringIO(gzip.decompress((base/'kernel_details.csv.gz').read_bytes()).decode())))
  assert len(rows)/2==v['kernel_count_per_batch']
  assert math.isclose(sum(float(x['Duration(us)']) for x in rows)/2000,v['kernel_sum_ms_per_batch'],abs_tol=1e-8)
  assert math.isclose(sum(x['ms_per_batch'] for x in v['groups'].values()),v['kernel_sum_ms_per_batch'],abs_tol=1e-8)
  endpoint=[x for x in v['shapes'] if x['type']=='RwkvEndpointWkv7'];assert len(endpoint)==1 and endpoint[0]['count_per_batch']==24
  assert endpoint[0]['block_num']==('48' if label.startswith('raw_eager') else '24')
  if label.endswith('_forward'):
   clean=r['timings_ms_per_batch']['eager_total' if label.startswith('raw_eager') else 'torchair_total']
   assert abs(v['kernel_sum_ms_per_batch']/clean-1)<.01
p.write_text(json.dumps(d,indent=2)+'\n')
plan=root/'26_rwkv_inference/PLAN.md';s=plan.read_text();anchor='Profiles show eager dispatch gaps and 2,441 kernels/score versus compiled 1,750;'
block='''**Largest B4 profiling**, source `57c83b92`, idle 910B2 NPU 1, FP32:
PyTorch CPU/NPU traces record shapes after five external + one profiler warmups,
two active calls; forward/pipeline captured separately and all parity checks pass.
T512/T2048 clean eager **190/694 ms**, TorchAir **214/810 ms**. Endpoint WKV
launches **48 eager / 24 compiled vector blocks**, same shapes/dtypes; device time
**42/170 ms eager / 84/340 ms compiled**. Core-count cause/fix remains untested;
no batch-size or precision conclusion. Shape/stack recording inflates eager host
trace time; use clean latency. [`Shapes, CSVs, hashes and accounting`](../tmp/26_rwkv_inference/reranker_largest_profile_b4_57c83b92/profile_comparison.json).

'''
assert s.count(anchor)==1 and '**Largest B4 profiling**' not in s
plan.write_text(s.replace(anchor,block+anchor))
print('Validated eight profiler CSV/hash sets, two-call accounting, 48/24 launch dimensions, and device sums versus clean timing.')
