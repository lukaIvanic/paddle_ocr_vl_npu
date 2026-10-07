"""Pair exact full-vision ND/NZ configurations; retain failed and incomplete work."""
import collections,csv,json,sys
from pathlib import Path
root=Path(sys.argv[1])
pairs=[('PromptFA D80 compiled','baseline','pfa_nz_weights'),('PromptFA D128 compiled','pfa_d128','pfa_d128_nz_weights'),('Grouped QKV compiled','grouped_qkv','grouped_qkv_nz_weights'),('Grouped QKV+FC1 compiled','grouped_qkv_mlp_fc1','grouped_qkv_mlp_fc1_nz_weights'),('PromptFA D80 eager','eager_pfa','eager_pfa_nz_weights'),('Unpad D128 eager','unpad_d128','unpad_d128_nz_weights')]
records={}
for p in root.glob('matrix/*/result.json'):
 d=json.loads(p.read_text());receipt=p.parent.with_name(p.parent.name+'.receipt')
 exit_path=receipt/'exit.json'
 if exit_path.exists():
  outcome=json.loads(exit_path.read_text())
  if outcome['status']!='completed':d['status']=outcome['status']
 else:d['status']='incomplete'
 d['receipt']=str(receipt.relative_to(root));d['launch_environment']=json.loads((receipt/'command.json').read_text())['environment'];records[(d['variant'],d['route'])]=d
routes=['crop_0_bucket_768','crop_1_bucket_3072']
result={'scope':'Complete 32-block vision on the same two original real-crop captures, not page throughput','pairs':[]}
for name,nd,nz in pairs:
 for route in routes:
  a,b=records.get((nd,route)),records.get((nz,route))
  if not a or not b:
   result['pairs'].append(dict(name=name,route=route,nd_status=a['status'] if a else 'missing',nz_status=b['status'] if b else 'missing'))
   continue
  row={'name':name,'route':route,'nd_status':a['status'],'nz_status':b['status']}
  if a['status']==b['status']=='completed':
   for key in ['vision_config','source_capture_sha256','model_hashes','execution','tags','device','torch','torch_npu','commit','launch_environment']:
    assert a[key]==b[key],(name,route,'mismatched',key)
   assert a['vision_config']['allow_internal_format'] is True
   if nd=='baseline':assert a['full_encoder_parity']['exact']
   for d,fmt in [(a,2),(b,29)]:
    assert len(d['vision_weights'])==128 and {v['format'] for v in d['vision_weights']}=={fmt}
    assert d['timing_gate']==dict(new_graphs=0,recompile_warnings=0)
    assert d['repeat_parity']['exact'] and not d['full_encoder_parity']['nonfinite']
    assert d['timing']['wall_ms']['count']==30
   row.update(real_tokens=a['tags']['real_tokens'],nd_wall_samples_ms=a['timing']['wall_samples_ms'],nz_wall_samples_ms=b['timing']['wall_samples_ms'],nd_wall_ms=a['timing']['wall_ms']['mean'],nz_wall_ms=b['timing']['wall_ms']['mean'],nd_event_ms=a['timing']['device_ms']['mean'],nz_event_ms=b['timing']['device_ms']['mean'],nd_tok_s=a['wall_real_tok_s'],nz_tok_s=b['wall_real_tok_s'],nz_vs_capture_l2=b['full_encoder_parity']['relative_l2'],nd_vs_capture_l2=a['full_encoder_parity']['relative_l2'],nz_kernel_audit=b['weight_kernel_audit'],grouped_inputs=b.get('grouped_weight_inputs'))
   row['nz_format_scope']='all_matmul_inputs_nz' if b['weight_kernel_audit']['every_matmul_reports_nz_input'] else 'mixed_inputs_despite_nz_parameters'
   row['nz_wall_change_percent']=(row['nz_wall_ms']/row['nd_wall_ms']-1)*100
   row['small_eager_comparison_caution']=a['execution']=='raw_eager' and a['tags']['real_tokens']==720
  result['pairs'].append(row)
(root/'paired_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
