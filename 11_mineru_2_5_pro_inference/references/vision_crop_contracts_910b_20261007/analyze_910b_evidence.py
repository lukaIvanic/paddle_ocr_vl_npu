"""Validate the extracted twelve-lane 910B evidence; no inference is run."""
import json,csv,pathlib,sys,collections
root=pathlib.Path(sys.argv[1]); output=pathlib.Path(sys.argv[2])
manifest=json.loads((root/'capture/manifest.json').read_text()); records=[]
for matrix in ['matrix','padding_matrix']:
 for p in sorted((root/matrix).glob('*/result.json')):
  d=json.loads(p.read_text())
  assert d.get('status')=='completed',p
  assert d['device']=='Ascend910B2'
  assert not any(d['timing_gate'].values())
  assert d['timing']['wall_ms']['count']==30
  assert d['repeat_parity']['exact'] and not d['full_encoder_parity']['nonfinite']
  entry=manifest['routes'][d['route']]
  assert d['source_capture_sha256']==entry['sha256']
  assert d['model_hashes']==manifest['model_hashes']
  formats=collections.Counter(w['format'] for w in d['vision_weights'])
  assert len(d['vision_weights'])==128
  assert formats=={29:128} if d['variant'].endswith('_nz_weights') else formats=={2:128}
  if d['variant']=='baseline': assert d['full_encoder_parity']['exact']
  files=list(p.parent.rglob('kernel_details.csv'));assert len(files)==1,p
  rows=list(csv.DictReader(files[0].open())); totals=collections.defaultdict(lambda:dict(count=0,ms_per_forward=0.0))
  for row in rows:
   x=totals[row['Type']];x['count']+=1;x['ms_per_forward']+=float(row['Duration(us)'])/3000
  attention=[r for r in rows if 'Attention' in r['Type']]
  linear=[r for r in rows if r['Type'].startswith('MatMul')]
  assert len(attention)==96 and len(linear)==384,(p,len(attention),len(linear))
  kernel_ms=sum(float(r['Duration(us)']) for r in rows)/3000
  attn_ms=sum(float(r['Duration(us)']) for r in attention)/3000
  linear_ms=sum(float(r['Duration(us)']) for r in linear)/3000
  by_format=collections.Counter(r['Input Formats'] for r in linear)
  conversions=[r for r in rows if 'TransData' in r['Type']]
  records.append(dict(variant=d['variant'],route=d['route'],execution=d['execution'],real_tokens=d['tags']['real_tokens'],physical_tokens=d['tags']['physical_tokens'],wall_ms=d['timing']['wall_ms']['mean'],event_ms=d['timing']['device_ms']['mean'],wall_real_tok_s=d['wall_real_tok_s'],event_real_tok_s=d['real_tok_s'],first_call_s=d['first_call_s'],weight_conversion_s=d.get('vision_weight_conversion_s'),relative_l2=d['full_encoder_parity']['relative_l2'],cosine=d['full_encoder_parity']['cosine'],max_abs=d['full_encoder_parity']['max_abs'],nonfinite=d['full_encoder_parity']['nonfinite'],stored_weight_formats=dict(formats),profile_forwards=3,attention_calls=len(attention),linear_calls=len(linear),kernel_ms=kernel_ms,attention_ms=attn_ms,linear_ms=linear_ms,remaining_ms=kernel_ms-attn_ms-linear_ms,attention_share=attn_ms/kernel_ms,matmul_input_formats=dict(by_format),attention_input_formats=dict(collections.Counter(r['Input Formats'] for r in attention)),conversion_calls=len(conversions),conversion_directions=dict(collections.Counter(r['Input Formats']+' -> '+r['Output Formats'] for r in conversions)),kernel_types=dict(totals),raw_result=str(p.relative_to(root))))
expected={(v,n) for v in ['baseline','pfa_nz_weights','pfa_d128','eager_pfa','unpad_d128','unpad_d128_nz_weights'] for n in [720,3036]}
assert len(records)==12 and {(d['variant'],d['real_tokens']) for d in records}==expected
output.write_text(json.dumps(dict(scope='32-block real-crop vision on 910B2; not page or text decode throughput',source_commit=manifest['commit'],model_hashes=manifest['model_hashes'],profile_denominator='kernel Duration(us) / (1000 * 3 forwards); PMU engine counters not elapsed time',records=records),indent=2)+'\n')
print('validated',len(records),'complete lanes')
for d in records:
 print(d['variant'],d['real_tokens'],round(d['wall_ms'],3),round(d['wall_real_tok_s']),round(d['attention_ms'],3),round(d['linear_ms'],3),round(d['relative_l2'],6),d['matmul_input_formats'])
