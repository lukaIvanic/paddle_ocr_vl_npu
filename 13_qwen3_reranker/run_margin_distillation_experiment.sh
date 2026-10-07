#!/usr/bin/env bash
# Source must have been fetched from a published commit before executing.
set -u
repo=$(git rev-parse --show-toplevel)
data=${1:?Pass prepared dataset.json.gz}
run_root=${2:?Pass a persistent evidence/checkpoint directory}
mkdir -p "$run_root"
python=/usr/local/python3.12.13/bin/python3
cd "$repo"

stage() (
    name=$1
    shift
    output="$run_root/$name"
    mkdir -p "$output"
    unset ASCEND_RT_VISIBLE_DEVICES
    # Vendor environment scripts reference optional unset variables.
    set +u
    source npu-setup || { echo 'NPU environment setup failed' >&2; exit 77; }
    set -u
    if [[ -z ${ASCEND_RT_VISIBLE_DEVICES:-} ]]; then
        echo 'No free NPU selected; stop without a CPU fallback.' >&2
        exit 77
    fi
    {
        git rev-parse HEAD
        hostname
        printf 'chip=Ascend 910B2 physical_npu=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
        printf '%q ' "$python" "$@"
        printf '\n'
    } > "$output/command.txt"
    "$python" "$@" > "$output/run.log" 2>&1
    code=$?
    printf '%s\n' "$code" > "$output/exit_code.txt"
    return "$code"
)

stage control 13_qwen3_reranker/check_margin_distillation.py \
    --model /workspace/models/Qwen3-Reranker-0.6B --dataset "$data" \
    --output "$run_root/control/control.json" || exit $?
stage teacher 13_qwen3_reranker/run_margin_distillation.py --mode teacher \
    --model /workspace/models/Qwen3-Reranker-4B --dataset "$data" \
    --output "$run_root/teacher" || exit $?
for schedule in constant warmup_linear; do
    stage "$schedule" 13_qwen3_reranker/run_margin_distillation.py --mode train \
        --model /workspace/models/Qwen3-Reranker-0.6B --dataset "$data" \
        --teacher "$run_root/teacher/teacher.json" --control "$run_root/control/control.json" \
        --output "$run_root/$schedule" --schedule "$schedule" \
        --steps 50 --queries-per-update 32 --learning-rate 1e-6 --wall-time-limit 2400 || exit $?
done
