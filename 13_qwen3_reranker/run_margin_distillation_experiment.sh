#!/usr/bin/env bash
# Source must have been fetched from a published commit before executing.
set -u
repo=$(git rev-parse --show-toplevel)
data=${1:?Pass prepared dataset.json.gz}
run_root=${2:?Pass a persistent evidence/checkpoint directory}
arm=${3:-all}
case "$arm" in all|constant|warmup_linear|contents_swapped) ;; *) echo 'Invalid training arm' >&2; exit 2 ;; esac
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

if [[ $arm == contents_swapped ]]; then
    reference_root=${4:?Pass the completed query-first run directory}
    peak_lr=${5:-1e-6}
    [[ -f $reference_root/teacher/teacher.json && -f $reference_root/warmup_linear/result.json ]] || exit 2
    stage control 13_qwen3_reranker/check_margin_distillation.py \
        --model /workspace/models/Qwen3-Reranker-0.6B --dataset "$data" \
        --student-order contents_swapped --output "$run_root/control/control.json" || exit $?
    stage warmup_linear 13_qwen3_reranker/run_margin_distillation.py --mode train \
        --model /workspace/models/Qwen3-Reranker-0.6B --dataset "$data" \
        --teacher "$reference_root/teacher/teacher.json" --control "$run_root/control/control.json" \
        --query-first-reference "$reference_root/warmup_linear/result.json" \
        --student-order contents_swapped --output "$run_root/warmup_linear" \
        --schedule warmup_linear --steps 50 --queries-per-update 32 \
        --learning-rate "$peak_lr" --wall-time-limit 2400 || exit $?
    exit 0
fi

if [[ $arm == all ]]; then
stage control 13_qwen3_reranker/check_margin_distillation.py \
    --model /workspace/models/Qwen3-Reranker-0.6B --dataset "$data" \
    --output "$run_root/control/control.json" || exit $?
stage teacher 13_qwen3_reranker/run_margin_distillation.py --mode teacher \
    --model /workspace/models/Qwen3-Reranker-4B --dataset "$data" \
    --output "$run_root/teacher" || exit $?
arms='constant warmup_linear'
else
    # Independent arms can run on separately selected free NPUs after setup.
    [[ -f $run_root/control/control.json && -f $run_root/teacher/teacher.json ]] || exit 2
    arms=$arm
fi
for schedule in $arms; do
    stage "$schedule" 13_qwen3_reranker/run_margin_distillation.py --mode train \
        --model /workspace/models/Qwen3-Reranker-0.6B --dataset "$data" \
        --teacher "$run_root/teacher/teacher.json" --control "$run_root/control/control.json" \
        --output "$run_root/$schedule" --schedule "$schedule" \
        --steps 50 --queries-per-update 32 --learning-rate 1e-6 --wall-time-limit 2400 || exit $?
done
