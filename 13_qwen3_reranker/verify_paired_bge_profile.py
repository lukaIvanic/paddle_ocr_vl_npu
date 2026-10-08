"""Verify optimizer-disabled paired preparation before reviewing a training launch."""
import argparse,json,statistics,math
from pathlib import Path
from distill_runtime import read,digest,save

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
 c=read(r/'configuration.json');x=read(r/'profile/result.json');control=read(r/'paired_loss_control.json');baseline=read(Path(c['baseline_root'])/'student/result.json')
 assert control['passed']
 assert x['status']=='profile_completed_no_optimizer_updates' and len(x['updates'])==3 and not x['checkpoints']
 assert x['dataset_sha256']==c['dataset_sha256'] and x['teacher_sha256']==c['teacher_sha256']
 assert x['recipe']['order_weights']==c['order_weights']
 assert x['lengths_by_order']==c['lengths_by_order']
 assert x['evaluation_panels']['membership_sha256']==baseline['evaluation_panels']['membership_sha256']
 by=x['evaluations']['0']['by_order'];assert set(by)=={'query_first','document_first'}
 references={'query_first':x['query_first_baseline'],'document_first':baseline['evaluations']['0']}
 comparisons={}
 for order,item in by.items():
  ref=references[order]
  for panel in ('trajectory_108','baseline_endpoint_180'):
   expected=ref.get(panel,ref.get('benchmark')) if panel=='trajectory_108' else ref[panel]
   assert item[panel]['suite_macro_ndcg10']==expected['suite_macro_ndcg10'],(order,panel,'baseline metrics changed')
  a_scores=item['benchmark_scores'];b_scores=ref['benchmark_scores'];assert a_scores.keys()==b_scores.keys()
  max_diff=max(abs(s-t) for k in a_scores for s,t in zip(a_scores[k],b_scores[k]));assert max_diff<1e-5
  comparisons[order]={'trajectory_108':item['trajectory_108']['suite_macro_ndcg10'],'baseline_endpoint_180':item['baseline_endpoint_180']['suite_macro_ndcg10'],'benchmark_score_max_abs_difference':max_diff}
 for u in x['updates']:
  assert not u['optimizer_step_applied'] and u['order_count']==2 and u['pairs']==u['unique_candidate_pairs']*2==u['queries']*16
  assert set(u['loss_by_order'])==set(by) and all(math.isfinite(v) for v in u['loss_by_order'].values())
  assert abs(u['loss']-sum(u['loss_by_order'].values())/2)<1e-5
  assert math.isfinite(u['gradient_norm'])
 frequent=sum(v['seconds']['benchmark']+v['seconds']['validation'] for v in by.values());assert frequent<180
 seconds=statistics.mean(u['seconds'] for u in x['updates'][1:]);checkpoint_overhead=30*10
 eta=500*seconds+len(c['eval_steps'])*frequent+2*sum(v['seconds']['reserved_benchmark'] for v in by.values())+checkpoint_overhead
 out={'passed':True,'configuration_sha256':digest(r/'configuration.json'),'profile_sha256':digest(r/'profile/result.json'),'loss_control_sha256':digest(r/'paired_loss_control.json'),'baseline_comparisons':comparisons,'optimizer_updates_applied':0,'seconds_per_update': [u['seconds'] for u in x['updates']],'steady_mean_seconds_per_update':seconds,'paired_frequent_eval_seconds':frequent,'paired_baseline_eval_seconds':x['evaluations']['0']['seconds']['total'],'peak_allocated_gib':max(u['peak_allocated_gib'] for u in x['updates']),'estimated_full_run_seconds':eta,'estimate_basis':'last two optimizer-disabled profile updates; measured both-order evaluation; 30 seconds per saved checkpoint; excludes startup and optimizer-step overhead','training_launched':False}
 save(r/'verification.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':main()
