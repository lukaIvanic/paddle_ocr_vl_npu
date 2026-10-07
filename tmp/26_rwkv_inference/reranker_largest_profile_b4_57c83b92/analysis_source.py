from pathlib import Path
from collections import defaultdict
import csv,json,hashlib
root=Path('/workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/reranker_largest_profile_b4_57c83b92')
result={'source_commit':'57c83b92198cf467088996680588dc7ab30a5460','scope':'Largest pair, FP32 states and dense math, idle 910B2 NPU 1, B4, validated right-padded endpoint inputs. Device forward and warm-file real-pair pipeline captured separately by PyTorch Ascend CPU/NPU profiler. Not a batch-size sweep, full-suite production pipeline, or cold-storage benchmark.', 'accounting':'Kernel time is sum of device task durations per recorded batch, not wall latency; interval union/span are separately reported. CPU scopes are nested and asynchronous and must not be summed or treated as pure transfer latency. Clean timings exclude profiling.', 'runs':{}}
for tokens in [512,2048]:
 r=root/('t'+str(tokens));report=json.loads((r/'result.json').read_text());assert report['all_checks_passed'] and report['dtype']=='fp32' and report['batch_size']==4
 out={'valid_lengths':report['valid_tokens'],'static_tokens':report['static_tokens'],'timings_ms_per_batch':{k:v['host_median_seconds']*1000 for k,v in report['timings'].items()},'setup_seconds':{k:report.get(k) for k in ['checkpoint_hash_load_convert_h2d_seconds','copy_private_graph_cache_seconds','compile_and_first_call_seconds','pipeline_setup_seconds']},'profiles':{}}
 for label,p in report['profiles'].items():
  summary=json.loads((Path(p['path'])/'summary.json').read_text());n=summary['profile_iterations'];assert n==2 and summary['record_shapes'] and len(summary['runs'])==1
  parsed=summary['runs'][0];rows=list(csv.DictReader(Path(parsed['files']['kernel_details']).open()))
  types=defaultdict(lambda:{'count':0,'duration_us':0.});shapes=defaultdict(lambda:{'count':0,'duration_us':0.});groups=defaultdict(lambda:{'count':0,'duration_us':0.});intervals=[]
  for row in rows:
   kind=row['Type'];duration=float(row['Duration(us)']);start=float(row['Start Time(us)'].strip());intervals.append((start,start+duration))
   group=('wkv_endpoint' if 'rwkvendpoint' in kind.lower() else 'wkv_reference' if 'rwkv' in kind.lower() else 'matmul' if 'matmul' in kind.lower() else 'cast' if kind.lower() in ['cast','bitcast'] else 'normalization' if 'norm' in kind.lower() else 'layout_gather_slice_broadcast' if any(x in kind.lower() for x in ['transpose','gather','slice','broadcast','transdata']) else 'other_pointwise_and_control')
   for table,key in [(groups,group),(types,kind),(shapes,(kind,row.get('Input Shapes',''),row.get('Output Shapes',''),row.get('Input Data Types',''),row.get('Output Data Types',''),row.get('Accelerator Core',''),row.get('Input Formats',''),row.get('Output Formats',''),row.get('Block Num','')))]:
    table[key]['count']+=1;table[key]['duration_us']+=duration
  merged=[]
  for a,b in sorted(intervals):
   if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
   else:merged.append([a,b])
  span=merged[-1][1]-merged[0][0];busy=sum(b-a for a,b in merged)
  trace=json.loads(Path(parsed['files']['trace_view']).read_text());events=trace['traceEvents'] if isinstance(trace,dict) else trace
  markers=defaultdict(lambda:{'count':0,'duration_us':0.})
  for event in events:
   if event.get('ph')=='X' and event.get('name','').startswith('rwkv.'):
    marker=markers[event['name']];marker['count']+=1;marker['duration_us']+=float(event.get('dur',0))
  assert markers['rwkv.'+label]['count']==n,(label,markers)
  normalized=lambda v:dict(count_per_batch=v['count']/n,ms_per_batch=v['duration_us']/n/1000)
  item={'warmup':{k:summary[k] for k in ['unprofiled_warmup_iterations','profiler_warmup_iterations','profile_iterations']},'kernel_count_per_batch':len(rows)/n,'kernel_sum_ms_per_batch':sum(float(x['Duration(us)']) for x in rows)/n/1000,'device_span_ms_per_batch':span/n/1000,'device_busy_union_ms_per_batch':busy/n/1000,'device_gap_ms_per_batch':(span-busy)/n/1000,
   'groups':{k:normalized(v) for k,v in groups.items()},'types':[dict(type=k,**normalized(v)) for k,v in sorted(types.items(),key=lambda kv:kv[1]['duration_us'],reverse=True)],
   'shapes':[dict(type=k[0],input_shapes=k[1],output_shapes=k[2],input_dtypes=k[3],output_dtypes=k[4],accelerator_core=k[5],input_formats=k[6],output_formats=k[7],block_num=k[8],**normalized(v)) for k,v in sorted(shapes.items(),key=lambda kv:kv[1]['duration_us'],reverse=True)],
   'cpu_markers':{k:dict(count=v['count'],ms_per_call=v['duration_us']/v['count']/1000) for k,v in markers.items()},
   'trace_sha256':{k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in parsed['files'].items()}}
  out['profiles'][label]=item
 result['runs'][str(tokens)]=out
(root/'profile_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
for T,r in result['runs'].items():
 print('TOKENS',T,'VALID',r['valid_lengths'],'CLEAN',r['timings_ms_per_batch'])
 for label,p in r['profiles'].items():
  print('PROFILE',label,'kernels',p['kernel_count_per_batch'],'sum_ms',round(p['kernel_sum_ms_per_batch'],3),'gap_ms',round(p['device_gap_ms_per_batch'],3));print('GROUPS',p['groups']);print('SHAPES',p['shapes'][:12]);print('CPU_MARKERS',p['cpu_markers'])
