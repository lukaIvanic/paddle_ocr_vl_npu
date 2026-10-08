#!/usr/bin/env bash
set -eo pipefail
cd /workspace/repos/qwen-family-filter-500
source npu-setup
set -u
export ASCEND_RT_VISIBLE_DEVICES=2
export HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=8
root=/workspace/results/qwen_bge_family_excluded_500_20261008
/usr/local/python3.12.13/bin/python3 - <<'PY'
import sys,json,os,shutil,subprocess
from pathlib import Path
sys.path.insert(0,'13_qwen3_reranker')
from distill_runtime import read,digest,save
from margin_distillation import lr_at
from bge_filtered_runtime import update_windows
from prepare_bge_family_filter import EXCLUDED
r=Path('/workspace/results/qwen_bge_family_excluded_500_20261008');p=r/'prepared';data=read(p/'dataset.json.gz');prep=read(p/'preparation.json');teacher=read(r/'teacher/teacher.json');old=read('/workspace/results/qwen_bge_document_first_10h/student/result.json')
assert not (r/'student/result.json').exists(),'Refuse duplicate run'
assert digest(p/'dataset.json.gz')==prep['dataset_sha256']==teacher['dataset_sha256']
assert digest(r/'teacher/teacher.json')==prep['teacher_sha256']
assert not {g['source'] for g in data['train']}&EXCLUDED
windows=update_windows(data['train'],500,32,'retained_original_slots')
assert list(map(len,windows))==[x['queries'] for x in old['updates'][:500]]
assert all(lr_at(i,1800,1e-5,'warmup_linear')==old['updates'][i-1]['lr'] for i in range(1,501))
assert all(x['truncated']==0 for v in prep['lengths'].values() for x in v.values())
assert len(data['benchmark'])==108 and len(data['reserved_benchmark'])==72 and len(data['validation'])==62
assert shutil.disk_usage(r).free>110*1024**3
review=read(r/'audit_review.json');assert review['launch_authorized'] and review['dataset_sha256']==prep['dataset_sha256']
config={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'dataset_sha256':prep['dataset_sha256'],'teacher_sha256':prep['teacher_sha256'],'parent_dataset_sha256':prep['parent_sha256'],'stop_after_updates':500,'lr_schedule_horizon_updates':1800,'peak_lr':1e-5,'warmup_updates':5,'decay':'original warmup_linear; identical LR at every update 1..500','lr_at_500':lr_at(500,1800,1e-5,'warmup_linear'),'group_counts':list(map(len,windows)),'training_groups':len(data['train']),'unique_training_queries':prep['unique_queries'],'pairs':8*len(data['train']),'source_counts':prep['source_counts'],'excluded_sources':sorted(EXCLUDED),'initialization':'released Qwen3-Reranker-0.6B; fresh NpuFusedAdamW','student_order':'correctly labeled document_first','loss':'supervised group CE + teacher distribution CE, 1:1','teacher':'unchanged frozen query-first 4B; verified exact cache reuse','validation':'unchanged 62 held-out mixture groups, including excluded training families; no optimizer updates on these','evaluation':'same 108-query trajectory and separate 180-query baseline/endpoint','eval_steps':[0,1,3,10,25,50,100,200,300,400,500],'physical_npu':os.environ['ASCEND_RT_VISIBLE_DEVICES'],'audit_review':review,'maximum_runtime_seconds':36000}
save(r/'run_configuration.json',config)
print('PREFLIGHT',json.dumps({k:v for k,v in config.items() if k not in ('group_counts','audit_review')}),flush=True)
PY
exec /usr/local/python3.12.13/bin/python3 -u 13_qwen3_reranker/run_bge_baseline.py \
 --mode train --model /workspace/models/Qwen3-Reranker-0.6B \
 --dataset "$root/prepared/dataset.json.gz" --teacher "$root/teacher/teacher.json" \
 --control /workspace/results/qwen_bge_baseline/reference_check.json \
 --query-first-reference /workspace/results/qwen_bge_baseline/query_first_lr1e5/result.json \
 --query-first-reference-dataset /workspace/results/qwen_bge_baseline/prepared/dataset.json.gz \
 --output "$root/student" --steps 500 --schedule-steps 1800 \
 --queries-per-update 32 --batch-schedule retained_original_slots \
 --schedule warmup_linear --learning-rate 1e-5 --student-order document_first --wall-time-limit 36000 \
 --eval-steps 0 1 3 10 25 50 100 200 300 400 500
