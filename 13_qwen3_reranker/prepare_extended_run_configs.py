"""Create hash-bound configurations for the two authorized 1800-update runs."""
import argparse,json,subprocess,collections
from pathlib import Path
from distill_runtime import read,digest,save,model_manifest
from bge_filtered_runtime import update_windows
from margin_distillation import lr_at

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
 prep=read(r/'prepared/preparation.json');data=read(r/'prepared/dataset.json.gz');teacher=read(r/'teacher/teacher.json');progress=read(r/'teacher/progress.json');audit=read(r/'audit_review.json')
 assert read(r/'resume_control.json')['passed']
 probe=Path('/workspace/results/qwen_bge_paired_filtered_500_20261008');assert read(probe/'verification.json')['passed'] and read(probe/'paired_loss_control.json')['passed']
 assert progress['status']=='completed' and progress['full_token_audit_passed']
 assert digest(r/'teacher/teacher.json')==progress['teacher_manifest_sha256']
 assert digest(r/'prepared/dataset.json.gz')==prep['dataset_sha256']==teacher['dataset_sha256']==audit['dataset_sha256']
 assert digest(r/'audit_review.json')==prep['audit_review_sha256']
 assert digest(Path(audit['append_audit_path'])/'summary.json')==audit['append_audit_summary_sha256']
 assert all(digest(Path(audit['append_audit_path'])/f'{t}.json')==h for t,h in audit['append_task_sha256'].items())
 assert all(x['truncated']==0 for d in prep['lengths'].values() for x in d.values())
 assert teacher['lengths']==prep['lengths']['query_first']
 base=Path('/workspace/results/qwen_bge_family_overlap_filtered_500_20261008');old=read(base/'prepared/dataset.json.gz');ot=read(base/'teacher/teacher.json')
 assert data['train'][:len(old['train'])]==old['train']
 for s in ('validation','benchmark','reserved_benchmark'):assert data[s]==old[s]
 for s in teacher['scores']:assert all(teacher['scores'][s][k]==v for k,v in ot['scores'][s].items())
 counts=list(map(len,update_windows(data['train'],1800,32,'retained_original_slots')));assert counts==prep['group_counts']
 source=Path(__file__).parent;checks=['run_bge_baseline.py','paired_bge_training.py','resume_bge_training.py','bge_baseline.py','distill_runtime.py','local_modeling_qwen3_reranker.py','training_smoke_data.py','transformers_rerank.py']
 cfg={'authorization':'User requested both runs through1800, append filtered data after existing500 prefix, launch and monitor both.','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'inputs':{'dataset':str(r/'prepared/dataset.json.gz'),'teacher':str(r/'teacher/teacher.json')},'dataset_sha256':prep['dataset_sha256'],'teacher_sha256':digest(r/'teacher/teacher.json'),'audit_review_sha256':digest(r/'audit_review.json'),'implementation_sha256':{f:digest(source/f) for f in checks},'model_sha256':model_manifest('/workspace/models/Qwen3-Reranker-0.6B'),'steps':1800,'schedule_steps':1800,'learning_rate':1e-5,'warmup':5,'schedule':'warmup_linear','final_lr':lr_at(1800,1800,1e-5,'warmup_linear'),'group_counts':counts,'groups':len(data['train']),'unique_queries':prep['unique_queries'],'source_counts':prep['source_counts'],'loss':'Supervised group CE + teacher CE, 1:1; raw yes-no margins; candidate group size8','eval_steps':[0,1,3,10,25,50]+list(range(100,1801,100)),'reserved_eval_steps':[500,1800],'evaluation':'Fixed108 throughout for each order; separate180 at baseline,500 and1800','maximum_runtime_seconds':129600,'first500_exact':True,'teacher_complete':True,'status':'prepared'}
 for mode in ('document_first','paired'):
  out=r/mode;out.mkdir(exist_ok=True)
  config=dict(cfg,mode=mode,orders=['query_first','document_first'] if mode=='paired' else ['document_first'],order_weights=[.5,.5] if mode=='paired' else [1.],initialization='released0.6B' if mode=='paired' else 'resume model+optimizer+RNG from original filtered checkpoint500',resume_parent_root=str(base) if mode=='document_first' else None)
  if (out/'configuration.json').exists():assert read(out/'configuration.json')==config
  else:save(out/'configuration.json',config)
 print('CONFIGURATIONS_READY',json.dumps({k:v for k,v in cfg.items() if k not in ('group_counts','source_counts','model_sha256','implementation_sha256')}),flush=True)
if __name__=='__main__':main()
