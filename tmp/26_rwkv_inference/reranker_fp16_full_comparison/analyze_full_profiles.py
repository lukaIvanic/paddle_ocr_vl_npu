from pathlib import Path
from collections import defaultdict
import csv,gzip,json,hashlib,math,shutil,sys
base=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent.parent
fp=base/'reranker_fp16_full_comparison'
out=json.loads((fp/'summary.json').read_text());out['runs']=[]
roots=[base/'reranker_matrix_full_comparison'/'reranker_matrix_full_3f3bd396',*sorted(p for p in fp.iterdir() if p.is_dir())]
for root in roots:
 for d in root.iterdir():
  if not d.is_dir() or not (d/'result.json').exists():continue
  r=json.loads((d/'result.json').read_text())
  if root.name=='reranker_matrix_full_3f3bd396' and not d.name.startswith('vector_'):continue
  row=dict(run=root.name+'/'+d.name,source_commit=r['source_commit'],dtype=r['dtype'],batch=r['batch_size'],static_tokens=r.get('static_tokens'),valid_tokens=r.get('valid_tokens'),physical_npu=r['physical_npu'],all_checks_passed=r['all_checks_passed'],recurrence=r.get('recurrence'),matrix_chunk_size=r.get('matrix_chunk_size') if r.get('recurrence')=='matrix' else None,matrix_compute_dtype=r.get('matrix_compute_dtype') if r.get('recurrence')=='matrix' else None,retain_dense_outputs=r.get('retain_dense_outputs',False),group_norm_impl=r.get('group_norm_impl','group_norm'),input_sha256=hashlib.sha256((d/'inputs.json').read_bytes()).hexdigest(),checkpoint_sha256=r['checkpoint_sha256'],reranker_sha256=r['reranker_sha256'],compiler_helper_sha256=r['source_sha256']['probe_reranker_endpoint.py'],clean_pipeline_ms={k:v['host_median_seconds']*1000 for k,v in r.get('timings',{}).items() if 'pipeline_total' in k},clean_scoring_ms={k:v['host_median_seconds']*1000 for k,v in r.get('timings',{}).items() if k in ['eager_total','torchair_total']},compile_and_first_call_seconds=r.get('compile_and_first_call_seconds'),compiled_state_errors=r.get('compiled_states_vs_eager'),compiled_score_error=r.get('compiled_logits_vs_eager'),padding_state_errors=r.get('padding_state_checks'),padding_score_error=r.get('batch_and_right_padding_vs_single'),matrix_score_error=r.get('matrix_vs_vector_logits',r.get('matrix_vs_vector_logit_diagnostics')),matrix_state_errors=r.get('matrix_vs_vector_states',r.get('matrix_vs_vector_state_diagnostics')),matrix_continuation=r.get('matrix_full_model_continuation'),implementation_vs_default_scores=r.get('implementation_vs_default_logit_diagnostics',r.get('retained_vs_default_logit_diagnostics')),implementation_vs_default_states=r.get('implementation_vs_default_state_diagnostics',r.get('retained_vs_default_state_diagnostics')),matrix_first_layer_finiteness=r.get('matrix_first_layer_finiteness'),cpu_anchor=r.get('cpu_gate'),source_sha256=r['source_sha256'],error=str(r.get('error') or '').splitlines()[:2],profiles={})
  for p in d.glob('profile_*/summary.json'):
   s=json.loads(p.read_text());n=s['profile_iterations'];parsed=s['runs'][0];kp=p.parent/(Path(parsed['files']['kernel_details']).name+'.gz');rows=list(csv.DictReader(gzip.open(kp,'rt')));types=defaultdict(lambda:[0,0.]);cores=defaultdict(lambda:[0,0.]);shapes=defaultdict(lambda:[0,0.])
   for item in rows:
    us=float(item['Duration(us)']);t=item['Type'];c=item['Accelerator Core']
    for data,k in [(types,t),(cores,c),(shapes,(t,item.get('Input Shapes',''),item.get('Input Data Types',''),c,item.get('Block Num','')))]:data[k][0]+=1;data[k][1]+=us
   markers=json.loads((p.parent/'cpu_rwkv_scopes.json').read_text());cpu=defaultdict(list)
   for marker in markers:cpu[marker['name']].append(marker['dur']/1000)
   row['profiles'][p.parent.name]=dict(kernel_count_per_call=len(rows)/n,kernel_sum_ms_per_call=sum(float(x['Duration(us)']) for x in rows)/n/1000,core_ms_per_call={k:v[1]/n/1000 for k,v in cores.items()},type_ms_per_call={k:v[1]/n/1000 for k,v in sorted(types.items(),key=lambda p:-p[1][1])},type_count_per_call={k:v[0]/n for k,v in types.items()},top_shapes=[dict(type=k[0],input_shapes=k[1],input_dtypes=k[2],core=k[3],block_num=k[4],calls_per_call=v[0]/n,ms_per_call=v[1]/n/1000) for k,v in sorted(shapes.items(),key=lambda p:-p[1][1])[:30]],record_shapes=s['record_shapes'],recorded_calls=n,cpu_marker_mean_ms={k:sum(v)/len(v) for k,v in cpu.items()},kernel_details_gzip_sha256=hashlib.sha256(kp.read_bytes()).hexdigest(),raw_profile_files=parsed['files'])
  out['runs'].append(row)
assert len({r['compiler_helper_sha256'] for r in out['runs']})==1
for batch in [1,4]:
 subset=[r for r in out['runs'] if r['batch']==batch]
 assert len({r['input_sha256'] for r in subset})==1
 assert len({r['checkpoint_sha256'] for r in subset})==1 and len({r['reranker_sha256'] for r in subset})==1
out['validated_comparisons']='Same input IDs/tokens/padding and checkpoints per batch; all runs use the identical original compiler-helper SHA256. B1/T512 valid257; B4/T2048 valid1038,2046,1039,2035. Matmul dtype comparisons do not change model order or candidate sets.'
out['interpretation']='FP16 reduces full-model matmul time. Retaining FP16 linear outputs reduces eager launch/cast overhead but compiled fusion costs offset the savings. Equivalent 64-channel LayerNorm plus original per-channel affine replaces slow GroupNorm; recurrence remains FP32 and dominant. Pure PyTorch FP16 matrix prototype is valid but slower at B1 chunk16, fails numerical gates at B4; optional upstream native helpers are not integrated. No precision/optimization-level changes in these comparisons.'
def clean(x):
 if isinstance(x,float) and not math.isfinite(x):return None
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
(fp/'summary.json').write_text(json.dumps(clean(out),indent=2,allow_nan=False)+'\n')
if Path(__file__).resolve() != (fp/'analyze_full_profiles.py').resolve():shutil.copy(__file__,fp/'analyze_full_profiles.py')
print('Summary valid:',len(out['runs']),'runs; matched inputs/checkpoints/compiler helper')
for r in out['runs']:print(r['run'],r['all_checks_passed'],r['clean_pipeline_ms'])
