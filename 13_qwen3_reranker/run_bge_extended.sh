#!/usr/bin/env bash
set -eo pipefail
mode=${1:?teacher, paired, or document_first}
[[ "$mode" == teacher || "$mode" == paired || "$mode" == document_first ]] || exit 2
cd /workspace/repos/qwen-paired-orders
source npu-setup
export ASCEND_RT_VISIBLE_DEVICES=${QWEN_EXTENDED_DEVICE:?Explicit verified idle NPU required}
[[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[1-3]$ ]] || exit 2
set -u
r=/workspace/results/qwen_bge_filtered_1800_20261008
npu-smi info > "$r/${mode}_npu_before.txt"
grep -q "No running processes found in NPU ${ASCEND_RT_VISIBLE_DEVICES} " "$r/${mode}_npu_before.txt" || exit 75
export HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=8
if [[ "$mode" == teacher ]]; then
 exec /usr/local/python3.12.13/bin/python3 -u 13_qwen3_reranker/produce_bge_extension_teacher.py --root "$r"
fi
export QWEN_EXTENDED_MODE="$mode"
/usr/local/python3.12.13/bin/python3 - <<'PY'
import sys,os,shutil
from pathlib import Path
sys.path.insert(0,'13_qwen3_reranker')
from distill_runtime import read,digest
r=Path('/workspace/results/qwen_bge_filtered_1800_20261008');mode=os.environ['QWEN_EXTENDED_MODE'];c=read(r/mode/'configuration.json')
assert not (r/mode/'student/result.json').exists(),'Refuse duplicate training'
assert digest(c['inputs']['dataset'])==c['dataset_sha256'] and digest(c['inputs']['teacher'])==c['teacher_sha256']
assert digest(r/'audit_review.json')==c['audit_review_sha256']
assert all(digest(Path('13_qwen3_reranker')/f)==h for f,h in c['implementation_sha256'].items())
assert all(digest(Path('/workspace/models/Qwen3-Reranker-0.6B')/f)==h for f,h in c['model_sha256'].items())
assert read(r/'teacher/progress.json')['status']=='completed'
assert shutil.disk_usage(r).free>500*1024**3
PY
extra=()
if [[ "$mode" == paired ]]; then
 extra=(--paired-orders)
else
 base=/workspace/results/qwen_bge_family_overlap_filtered_500_20261008
 [[ $(cat "$base/exit_code") == 0 ]] || exit 1
 extra=(--resume-checkpoint "$base/student/checkpoint_500.pt" --resume-parent-root "$base")
fi
exec /usr/local/python3.12.13/bin/python3 -u 13_qwen3_reranker/run_bge_baseline.py \
 --mode train --model /workspace/models/Qwen3-Reranker-0.6B \
 --dataset "$r/prepared/dataset.json.gz" --teacher "$r/teacher/teacher.json" \
 --control /workspace/results/qwen_bge_baseline/reference_check.json \
 --query-first-reference /workspace/results/qwen_bge_baseline/query_first_lr1e5/result.json \
 --query-first-reference-dataset /workspace/results/qwen_bge_baseline/prepared/dataset.json.gz \
 --output "$r/$mode/student" --steps 1800 --schedule-steps 1800 \
 --queries-per-update 32 --batch-schedule retained_original_slots \
 --schedule warmup_linear --learning-rate 1e-5 --student-order document_first \
 --wall-time-limit 129600 --eval-steps 0 1 3 10 25 50 $(seq 100 100 1800) \
 --reserved-eval-steps 500 1800 "${extra[@]}"
