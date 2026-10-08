#!/usr/bin/env bash
set -eo pipefail
mode=${1:?Specify profile or train}
[[ "$mode" == profile || "$mode" == train ]] || exit 2
cd /workspace/repos/qwen-paired-orders
source npu-setup
set -u
root=/workspace/results/qwen_bge_paired_filtered_500_20261008
npu-smi info > "$root/${mode}_npu_before.txt"
if ! grep -q "No running processes found in NPU ${ASCEND_RT_VISIBLE_DEVICES} " "$root/${mode}_npu_before.txt"; then
  echo "Selected device is no longer idle; refusing launch" >&2
  exit 75
fi
export HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=8
export QWEN_PAIRED_MODE="$mode"
/usr/local/python3.12.13/bin/python3 - <<'PY'
import os,sys,shutil
from pathlib import Path
sys.path.insert(0,'13_qwen3_reranker')
from distill_runtime import read,digest
r=Path('/workspace/results/qwen_bge_paired_filtered_500_20261008');c=read(r/'configuration.json');mode=os.environ['QWEN_PAIRED_MODE']
assert c['status']=='prepared_not_launched'
assert digest(c['inputs']['dataset'])==c['dataset_sha256']
assert digest(c['inputs']['teacher'])==c['teacher_sha256']
assert digest(Path(c['baseline_root'])/'audit_review.json')==c['audit_review_sha256']
assert all(digest(Path('13_qwen3_reranker')/f)==h for f,h in c['implementation_sha256'].items())
assert all(digest(Path(c['model'])/f)==h for f,h in c['model_sha256'].items())
assert not (r/('profile' if mode=='profile' else 'student')/'result.json').exists(),'Refuse duplicate run'
if mode=='train':
 v=read(r/'verification.json');assert v['passed'] and v['configuration_sha256']==digest(r/'configuration.json')
 assert shutil.disk_usage(r).free>110*1024**3
PY
if [[ "$mode" == profile ]]; then
 /usr/local/python3.12.13/bin/python3 -u 13_qwen3_reranker/check_paired_bge_training.py --output "$root/paired_loss_control.json"
 extra=(--profile-updates 3 --profile-eval)
 out="$root/profile"
else
 extra=()
 out="$root/student"
fi
base=/workspace/results/qwen_bge_family_overlap_filtered_500_20261008
exec /usr/local/python3.12.13/bin/python3 -u 13_qwen3_reranker/run_bge_baseline.py \
 --mode "$mode" --model /workspace/models/Qwen3-Reranker-0.6B \
 --dataset "$base/prepared/dataset.json.gz" --teacher "$base/teacher/teacher.json" \
 --control /workspace/results/qwen_bge_baseline/reference_check.json \
 --query-first-reference /workspace/results/qwen_bge_baseline/query_first_lr1e5/result.json \
 --query-first-reference-dataset /workspace/results/qwen_bge_baseline/prepared/dataset.json.gz \
 --output "$out" --steps 500 --schedule-steps 1800 \
 --queries-per-update 32 --batch-schedule retained_original_slots \
 --schedule warmup_linear --learning-rate 1e-5 --student-order document_first --paired-orders \
 --wall-time-limit 36000 --eval-steps 0 1 3 10 25 50 100 200 300 400 500 "${extra[@]}"
