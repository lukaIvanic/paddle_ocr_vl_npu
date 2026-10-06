import argparse, json, statistics
from pathlib import Path

def analyse(p):
 out={'status':p['status'],'commit':p['commit'],'phase':p['phase'],'environment':p['environment'],'summary':p['summary'],'partition_summaries':p.get('partition_summaries')}
 names=list(p['pairs'][0]['variants'])
 out['score_diagnostics']={}
 for source in sorted({r['source'] for r in p['pairs']}):
  rows=[r for r in p['pairs'] if r['source']==source]
  source_out={}
  for name in names:
   values=[r['variants'][name] for r in rows]
   baseline=[r['variants']['normal'] for r in rows]
   changes=[v['margin']-b['margin'] for v,b in zip(values,baseline)]
   source_out[name]={
    'mean_no_logit_change':statistics.mean(v['no_logit']-b['no_logit'] for v,b in zip(values,baseline)),
    'mean_yes_logit_change':statistics.mean(v['yes_logit']-b['yes_logit'] for v,b in zip(values,baseline)),
    'mean_margin_change':statistics.mean(changes),
    'mean_score':statistics.mean(v['score'] for v in values),
    'mean_absolute_score_change':statistics.mean(abs(v['score']-b['score']) for v,b in zip(values,baseline)),
    'margin_correlation':statistics.correlation([v['margin'] for v in values],[b['margin'] for b in baseline]),
    'grade_margin_changes':{str(g):statistics.mean(r['variants'][name]['margin']-r['variants']['normal']['margin'] for r in rows if r['grade']==g) for g in sorted({r['grade'] for r in rows})},
    'full_vocabulary_yes_no_mass_min':min(v['full_vocabulary_yes_probability']+v['full_vocabulary_no_probability'] for v in values),
    'full_vocabulary_yes_no_mass_max':max(v['full_vocabulary_yes_probability']+v['full_vocabulary_no_probability'] for v in values),
   }
  out['score_diagnostics'][source]=source_out
 errors=[abs(v['margin']-v['fp32_head_margin']) for r in p['pairs'] for v in r['variants'].values()]
 out['fp32_head_margin_error']={'mean':statistics.mean(errors),'max':max(errors),'scope':'only final projection recomputed; backbone and its hidden states remain BF16'}
 out['prompt_audit']={
  'all_suffix_ids_identical':all(v['input_ids'][-len(p['suffix_ids']):]==p['suffix_ids'] for r in p['pairs'] for v in r['variants'].values()),
  'all_prefix_ids_identical_except_declared_system_remap':all(v['input_ids'][:len(p['normal_prefix_ids'])]==p['normal_prefix_ids'] for r in p['pairs'] for name,v in r['variants'].items() if name not in ('swapped_contents_system_remapped','swapped_contents_joint_remap')),
  'max_input_tokens':max(v['tokens'] for r in p['pairs'] for v in r['variants'].values()),
 }
 if p['phase']=='calibration':out['normal_max_abs_score_change_from_prior']=max(abs(r['normal_score_change_from_prior']) for r in p['pairs'])
 out['pairs']=[{k:v for k,v in r.items() if k not in ('variants','document')}|{'variants':{name:{k:v[k] for k in ('score','margin','no_logit','yes_logit','fp32_head_margin','tokens')} for name,v in r['variants'].items()}} for r in p['pairs']]
 return out

parser=argparse.ArgumentParser();parser.add_argument('input',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
p=json.loads(args.input.read_text());assert p['status']=='completed'
s=analyse(p);args.output.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
for source in s['score_diagnostics']:
 print(source,s['summary'][source]['pairs'],s['summary'][source]['queries'])
 for name,v in s['summary'][source]['orderings'].items():
  d=s['score_diagnostics'][source][name]
  print(name,f"{v['correct']}/{v['total']}",'ties',v['ties'],'mean-margin-change',round(d['mean_margin_change'],3),'grade-shifts',d['grade_margin_changes'])
