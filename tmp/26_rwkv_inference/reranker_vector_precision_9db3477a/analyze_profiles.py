from pathlib import Path
from collections import defaultdict
import json,csv,gzip,hashlib,sys
root=Path(sys.argv[1])
summary=dict(hardware='Ascend910B2, CANN 9.0.1; same physical NPU1 for all full-model measurements',
 workload='Largest RWKV pair, FP16 dense projections, equivalent per-head LayerNorm; unchanged original TorchAir compiler helper. Real prepared NanoSCIDOCS pairs. Warm file read, tokenize/prepare, H2D, full backbone/readout, D2H, JSON write. Startup/compile/profiling excluded.',
 method='10 clean repeats after 3 warmups; separate PyTorch CPU/NPU full-forward and pipeline profiles, record_shapes/with_stack, 5 external plus 1 profiler warmups, 2 active calls. Synthetic checks are correctness-only, not speed tests.',
 limitation='All-FP16 recurrence fails existing numerical gates; diagnostic timings do not establish preserved NDCG. No full-suite quality run for these opt-ins. Original packages/defaults retained.',runs=[],arithmetic={})
for p in sorted(root.glob('*/result.json')):
 r=json.loads(p.read_text())
 if 'valid_tokens' not in r:continue
 row={k:r.get(k) for k in ['source_commit','source_sha256','physical_npu','batch_size','static_tokens','valid_tokens','checkpoint_sha256','reranker_sha256','all_checks_passed','vector_variant','vector_accuracy_passed','vector_accuracy_error','vector_vs_stock_states','vector_vs_stock_logits','vector_padding','vector_continuation','compiled_states_vs_eager','compiled_logits_vs_eager','batch_and_right_padding_vs_single','error']}
 row.update(run=p.parent.name,input_sha256=hashlib.sha256((p.parent/'inputs.json').read_bytes()).hexdigest(),
   timings_ms={k:v['host_median_seconds']*1000 for k,v in r['timings'].items()},profiles={})
 for q in sorted(p.parent.glob('profile_*/summary.json')):
  s=json.loads(q.read_text());n=s['profile_iterations'];kp=q.parent/(Path(s['runs'][0]['files']['kernel_details']).name+'.gz')
  types=defaultdict(lambda:[0,0.]);shapes=defaultdict(lambda:[0,0.])
  for x in csv.DictReader(gzip.open(kp,'rt')):
   us=float(x['Duration(us)'])
   for data,key in [(types,x['Type']),(shapes,(x['Type'],x['Input Shapes'],x['Input Data Types'],x['Accelerator Core'],x['Block Num']))]:
    data[key][0]+=1;data[key][1]+=us
  row['profiles'][q.parent.name]=dict(type_ms={k:v[1]/n/1000 for k,v in sorted(types.items(),key=lambda x:-x[1][1])},
    shapes=[dict(type=k[0],input_shapes=k[1],input_dtypes=k[2],core=k[3],block_num=k[4],calls=v[0]/n,ms=v[1]/n/1000) for k,v in sorted(shapes.items(),key=lambda x:-x[1][1])],
    kernel_details_sha256=hashlib.sha256(kp.read_bytes()).hexdigest(),source_files=s['runs'][0]['files'])
 summary['runs'].append(row)
for p in root.glob('*arithmetic.json'):summary['arithmetic'][p.stem]=json.loads(p.read_text())
assert len({r['source_sha256']['probe_reranker_endpoint.py'] for r in summary['runs']})==1
b4=[r for r in summary['runs'] if r['batch_size']==4]
assert len({r['input_sha256'] for r in b4})==1
assert len({r['checkpoint_sha256'] for r in b4})==1
assert len({r['reranker_sha256'] for r in b4})==1
assert len({r['physical_npu'] for r in summary['runs']})==1
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print([(r['run'],r['all_checks_passed'],r['timings_ms']) for r in summary['runs']])
