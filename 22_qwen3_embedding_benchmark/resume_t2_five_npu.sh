#!/usr/bin/env bash
# One-time transition of this explicitly identified owned evaluation, not pkill.
set -euo pipefail
old_pid=${1:?Pass the verified evaluator PID, not a server or process-group PID}
[[ "$old_pid" =~ ^[0-9]+$ ]] || exit 2
old_run=tmp/22_qwen3_embedding_benchmark/reranker_cmteb_6npu_fc41475e
new_run=tmp/22_qwen3_embedding_benchmark/reranker_cmteb_5npu_resume
[[ -d "$old_run/evaluation/T2Retrieval" && ! -e "$new_run" ]] || exit 2
cmd=$(tr '\0' ' ' <"/proc/$old_pid/cmdline")
[[ "$cmd" == *"/python 22_qwen3_embedding_benchmark/run_reranker_evaluation.py "* && "$cmd" == *"--output $old_run/evaluation "* ]] || exit 2
mkdir "$new_run"
exec >"$new_run/run.log" 2>&1
printf 'Transition start: %s; old evaluator PID=%s\n' "$(date -Iseconds)" "$old_pid"
printf 'Verified command: %s\n' "$cmd"
kill -TERM "$old_pid"
for attempt in {1..90}; do
    [[ -e "/proc/$old_pid" ]] || break
    sleep 1
done
if [[ -e "/proc/$old_pid" ]]; then
    printf 'Old evaluator has not exited; refusing to overlap clients\n'
    exit 1
fi
for port in {18325..18330}; do
    if curl -sf --max-time 2 "http://127.0.0.1:$port/health"; then
        printf 'Old server on %s still alive; refusing to overlap\n' "$port"
        exit 1
    fi
done
printf 'Old evaluator and owned servers stopped: %s\n' "$(date -Iseconds)"
set +eu
source npu-setup
status=$?
set -eu
[[ "$status" == 0 ]] || exit "$status"
export ASCEND_RT_VISIBLE_DEVICES=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
exec /workspace/venvs/qwen3_embedding_eval_py312/bin/python 22_qwen3_embedding_benchmark/run_reranker_evaluation.py \
    --tasks T2Retrieval --devices 0 1 2 3 6 --output "$new_run/evaluation" \
    --resume-partial-run "$old_run/evaluation" \
    --completed-run tmp/22_qwen3_embedding_benchmark/reranker_ecom_e70e8473/evaluation \
    --prepared tmp/22_qwen3_embedding_benchmark/reranker_serving_cc6c6841/prepared \
    --candidates tmp/22_qwen3_embedding_benchmark/cmteb_910b_8ee60dc7/evaluation/mteb
