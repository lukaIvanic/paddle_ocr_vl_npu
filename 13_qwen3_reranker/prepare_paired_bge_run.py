"""Prepare a paired run manifest using the immutable, already audited inputs."""
import argparse,json,subprocess
from pathlib import Path
from distill_runtime import read,digest,save,model_manifest
from margin_distillation import lr_at
from bge_filtered_runtime import update_windows

def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 b=a.baseline_root;cfg=read(b/'run_configuration.json');prep=read(b/'prepared/preparation.json');data=read(b/'prepared/dataset.json.gz');teacher=read(b/'teacher/teacher.json');audit=read(b/'audit_review.json')
 assert digest(b/'prepared/dataset.json.gz')==cfg['dataset_sha256']==prep['dataset_sha256']==teacher['dataset_sha256']==audit['dataset_sha256']
 assert digest(b/'teacher/teacher.json')==cfg['teacher_sha256']==prep['teacher_sha256']
 assert digest(b/'audit/summary.json')==audit['audit_summary_sha256']
 assert all(digest(b/'audit'/f'{t}.json')==h for t,h in audit['task_audit_sha256'].items())
 assert audit['substantive_detected_matches']==0
 assert all(x['truncated']==0 for sections in prep['lengths'].values() for x in sections.values())
 groups=update_windows(data['train'],500,32,'retained_original_slots');assert list(map(len,groups))==cfg['group_counts']
 assert cfg['stop_after_updates']==500 and cfg['lr_schedule_horizon_updates']==1800 and cfg['peak_lr']==1e-5 and cfg['warmup_updates']==5
 source=Path(__file__).parent
 out={'status':'prepared_not_launched','baseline_root':str(b),'output_root':str(a.output),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'implementation_sha256':{f:digest(source/f) for f in ['run_bge_baseline.py','paired_bge_training.py','bge_baseline.py','distill_runtime.py','training_smoke_data.py','transformers_rerank.py','local_modeling_qwen3_reranker.py']},
 'inputs':{'dataset':str(b/'prepared/dataset.json.gz'),'teacher':str(b/'teacher/teacher.json'),'control':'/workspace/results/qwen_bge_baseline/reference_check.json','query_first_reference':'/workspace/results/qwen_bge_baseline/query_first_lr1e5/result.json','query_first_reference_dataset':'/workspace/results/qwen_bge_baseline/prepared/dataset.json.gz'},
 'dataset_sha256':cfg['dataset_sha256'],'teacher_sha256':cfg['teacher_sha256'],'audit_review_sha256':digest(b/'audit_review.json'),
 'model':'/workspace/models/Qwen3-Reranker-0.6B','model_sha256':model_manifest('/workspace/models/Qwen3-Reranker-0.6B'),
 'training_groups':len(data['train']),'unique_queries':prep['unique_queries'],'unique_candidate_pairs':8*len(data['train']),'sequence_presentations':16*len(data['train']),
 'group_counts':cfg['group_counts'],'source_counts':cfg['source_counts'],'lengths_by_order':prep['lengths'],
 'orders':['query_first','document_first'],'order_weights':{'query_first':0.5,'document_first':0.5},'loss':'Each order: supervised CE + teacher CE, 1:1; mean two orders and underlying query groups. Separate 8-candidate softmax per order.',
 'optimizer':'fresh NpuFusedAdamW; same parameters as baseline; clip once and step once per underlying query batch',
 'steps':500,'schedule_steps':1800,'peak_lr':1e-5,'warmup_updates':5,'schedule':'warmup_linear','lr_at_500':lr_at(500,1800,1e-5,'warmup_linear'),
 'eval_steps':cfg['eval_steps'],'evaluation':'Both orders on identical 108-query trajectory; both orders on separate 180-query baseline and endpoint; per-order validation components',
 'material_choices':[{'choice':'Both orderings of the same supplied query/candidates per update','basis':'User request'}, {'choice':'Equal 0.5/0.5 order weights; same existing loss and fixed 4B targets','basis':'Assistant proposal accepted by user'}, {'choice':'Identical filtered inputs, 500-update stop and 1800-update horizon','basis':'Accepted matched-comparison plan'}, {'choice':'No training launch during preparation','basis':'User requested preparation'}]}
 a.output.mkdir(parents=True,exist_ok=True)
 path=a.output/'configuration.json'
 if path.exists():assert read(path)==out,'Refuse changing existing prepared configuration'
 else:save(path,out)
 print(json.dumps({k:v for k,v in out.items() if k not in ['group_counts','source_counts','lengths_by_order','implementation_sha256','model_sha256']}),flush=True)
if __name__=='__main__':main()
