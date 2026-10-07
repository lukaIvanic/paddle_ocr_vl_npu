from pathlib import Path
import json,csv,hashlib
from collections import defaultdict
root=Path('/workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/reranker_profile_b1_3ea75a78')
result={'aggregation_note':'BroadcastTo is a layout/broadcast operation, not a Cast; CPU d2h marker includes waiting for queued device work; Pipe utilization counter anomalies prevent utilization claims.', 'scope':'910B2 NPU 7 shared; middle pair B1/T257, two calls per profile. Kernel durations are summed across tasks and normalized per score; idle derived from interval union, not subtracting overlapping CPU scopes. Clean timing is measured outside profiler. Pipe counters are not used to claim HBM saturation.','runs':{}}
for dtype in ['fp32','fp16']:
 r=root/dtype;report=json.loads((r/'result.json').read_text())
 entry={'all_checks_passed':report['all_checks_passed'],'timings_ms':{k:v['host_median_seconds']*1000 for k,v in report['timings'].items()},'startup_seconds':{k:report.get(k) for k in ['checkpoint_hash_load_convert_h2d_seconds','copy_private_graph_cache_seconds','compile_and_first_call_seconds','pipeline_setup_seconds']},'profiles':{}}
 for label in ['forward','pipeline']:
  p=r/('profile_'+label)/'summary.json'
  summary=json.loads(p.read_text());assert summary['profile_iterations']==2 and len(summary['runs'])==1
  parsed=summary['runs'][0];csvpath=Path(parsed['files']['kernel_details']);rows=list(csv.DictReader(csvpath.open()))
  kinds=defaultdict(lambda:{'count':0,'duration_us':0.,'input_dtypes':set()});groups=defaultdict(lambda:{'count':0,'duration_us':0.})
  intervals=[]
  for row in rows:
   t=row['Type'];dur=float(row['Duration(us)']);start=float(row['Start Time(us)'].strip())
   intervals.append((start,start+dur));kinds[t]['count']+=1;kinds[t]['duration_us']+=dur;kinds[t]['input_dtypes'].add(row['Input Data Types'])
   lo=t.lower()
   group=('wkv' if 'rwkv' in lo else 'matmul' if 'matmul' in lo else 'cast' if lo in ['cast','bitcast'] else 'normalization' if 'norm' in lo else 'layout_gather_slice_broadcast' if any(n in lo for n in ['transpose','gather','slice','broadcast','transdata']) else 'other_pointwise_and_control')
   groups[group]['count']+=1;groups[group]['duration_us']+=dur
  merged=[]
  for a,b in sorted(intervals):
   if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
   else:merged.append([a,b])
  busy=sum(b-a for a,b in merged);span=merged[-1][1]-merged[0][0]
  trace=json.loads(Path(parsed['files']['trace_view']).read_text());events=trace['traceEvents'] if isinstance(trace,dict) else trace
  markers=defaultdict(lambda:{'count':0,'duration_us':0.})
  for e in events:
   if e.get('ph')=='X' and e.get('name','').startswith('rwkv.'):
    m=markers[e['name']];m['count']+=1;m['duration_us']+=float(e.get('dur',0))
  item={'kernel_count_per_score':len(rows)/2,'kernel_sum_ms_per_score':sum(float(x['Duration(us)']) for x in rows)/2000,
   'device_span_ms_per_score':span/2000,'device_busy_union_ms_per_score':busy/2000,'device_gap_ms_per_score':(span-busy)/2000,
   'groups':{k:{'count_per_score':v['count']/2,'ms_per_score':v['duration_us']/2000} for k,v in groups.items()},
   'kernel_types':[{ 'type':k,'count_per_score':v['count']/2,'ms_per_score':v['duration_us']/2000,'input_dtypes':sorted(v['input_dtypes'])} for k,v in sorted(kinds.items(),key=lambda kv:kv[1]['duration_us'],reverse=True)],
   'cpu_marker_ms_per_call':{k:{'count':v['count'],'ms_per_call':v['duration_us']/v['count']/1000} for k,v in markers.items()},
   'trace_sha256':{k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in parsed['files'].items()},
   'step_trace_totals_us':parsed.get('step_trace_time',{}).get('totals_us',{})}
  entry['profiles'][label]=item
 result['runs'][dtype]=entry
(root/'profile_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
for dtype,d in result['runs'].items():
 print(dtype,'clean_ms',d['timings_ms'],'startup_s',d['startup_seconds'])
 for label,p in d['profiles'].items():
  print(label,'kernels',p['kernel_count_per_score'],'sum_ms',p['kernel_sum_ms_per_score'],'busy_ms',p['device_busy_union_ms_per_score'],'gap_ms',p['device_gap_ms_per_score'])
  print('GROUPS',p['groups']);print('CPU_MARKERS',p['cpu_marker_ms_per_call'])
  print('TYPES',[(x['type'],x['count_per_score'],round(x['ms_per_score'],4),x['input_dtypes'][:2]) for x in p['kernel_types'][:12]])
