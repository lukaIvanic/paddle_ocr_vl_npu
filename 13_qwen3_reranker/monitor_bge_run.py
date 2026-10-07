"""Read-only progress summary for the expanded BGE experiment and its teacher."""
import argparse,datetime,json,statistics,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
 def read(path):return json.loads(path.read_text()) if path.exists() else None
 summary={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(r)}
 config=read(r/'run_configuration.json')
 if config:summary['configuration']={k:config[k] for k in ('steps','retained_groups','learning_rate_peak','warmup_updates','physical_npu','teacher_physical_npu')}
 prep=read(r/'prepared/preparation.json')
 if prep:summary['prepared']={'groups':prep['retained_groups'],'unique_queries':prep['sampling']['unique_query_hashes'],'dataset_sha256':prep['dataset_sha256']}
 producer=read(r/'teacher/progress.json')
 if producer:summary['teacher']={k:producer[k] for k in ('status','physical_npu','completed_groups','total_groups','seconds','full_token_audit_passed') if k in producer}
 failure=read(r/'teacher/failure.json')
 if failure:summary['teacher_failure']=failure
 student=r/'student';d=read(student/'result.json')
 if d:
  summary['student_status']=d['status'];updates=d['updates'];summary['updates_completed']=len(updates)
  if updates:
   summary['latest_update']=updates[-1];summary['mean_recent_update_seconds']=statistics.mean(x['seconds'] for x in updates[-100:])
   summary['consumed_query_groups']=sum(x['queries'] for x in updates)
  summary['evaluations']={step:{'trajectory_108':v['trajectory_108']['suite_macro_ndcg10'],'baseline_endpoint_180':v.get('baseline_endpoint_180',{}).get('suite_macro_ndcg10'),'validation':v.get('validation_objective'),'teacher_ordering_agreement':v['agreement']['pairwise_agreement'],'seconds':v['seconds']['total']} for step,v in d['evaluations'].items()}
  summary['checkpoint_steps']=[c['step'] for c in d['checkpoints']]
  summary['checkpoints_match_recorded_sizes']=all(Path(c['path']).exists() and Path(c['path']).stat().st_size==c['bytes'] for c in d['checkpoints'])
  summary['teacher_chunks_consumed']=len(d.get('teacher_stream_chunks_consumed',{}))
  for key in ('error','total_seconds','teacher_stream_full_token_audit_passed'):
   if key in d:summary[key]=d[key]
  if updates and d['status']=='running':
   remaining=max(0,d['config']['steps']-updates[-1]['step']);pending=sum(s>updates[-1]['step'] for s in d['evaluation_panels']['trajectory_108']['steps'])
   ckpt=statistics.mean(c['seconds'] for c in d['checkpoints']) if d['checkpoints'] else 20
   summary['estimated_remaining_seconds_from_recent_cadence']=remaining*summary['mean_recent_update_seconds']+pending*(75+ckpt)+46
 log=student/'run.log'
 if log.exists():
  summary['log_age_seconds']=time.time()-log.stat().st_mtime
  with log.open('rb') as f:
   f.seek(max(0,log.stat().st_size-20000));tail=f.read().decode(errors='replace')
  lines=tail.splitlines();summary['log_tail']=lines[-5:]
  for line in reversed(lines):
   if line.startswith('UPDATE '):summary['latest_logged_update']=json.loads(line[7:]);break
 for name in ('teacher.exit','student/exit_code.txt'):
  path=r/name
  if path.exists():summary[name]=path.read_text().strip()
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
