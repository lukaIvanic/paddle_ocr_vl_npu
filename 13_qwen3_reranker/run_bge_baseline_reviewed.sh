#!/usr/bin/env bash
# Run only after the user reviews the completed preparation and configuration.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source npu-setup
exec /usr/local/python3.12.13/bin/python3 -u 13_qwen3_reranker/run_bge_baseline.py \
  --mode train \
  --model /workspace/models/Qwen3-Reranker-0.6B \
  --dataset /workspace/results/qwen_bge_baseline/prepared/dataset.json.gz \
  --teacher /workspace/results/qwen_bge_baseline/teacher/teacher.json \
  --control /workspace/results/qwen_bge_baseline/reference_check.json \
  --output /workspace/results/qwen_bge_baseline/query_first_lr1e5 \
  --steps 50 --queries-per-update 32 --schedule warmup_linear \
  --learning-rate 1e-5 --student-order query_first --wall-time-limit 2400
