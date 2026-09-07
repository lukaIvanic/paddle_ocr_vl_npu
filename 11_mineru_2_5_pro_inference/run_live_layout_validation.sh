#!/usr/bin/env bash
# 910B validation: live layout, real page images, existing recognition caches.
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
source npu-setup
set -u
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mineru_python="${PYTHON:-/workspace/venvs/mineru_pro_vllm_py312/bin/python}"
mineru_root="${RUN_ROOT:?Set a new RUN_ROOT}"
test ! -e "$mineru_root/output"
mkdir -p "$mineru_root"
exec 9>.runtime_cache/11_mineru_2_5_pro_inference/serving_validation.lock
flock -n 9 || { echo 'MinerU cache owner is busy.' >&2; exit 2; }
mineru_args=(11_mineru_2_5_pro_inference/run_page_pipeline.py
  --dataset-json "${DATASET_JSON:-/workspace/datasets/OmniDocBench/OmniDocBench.json}"
  --images-dir /workspace/datasets/OmniDocBench/images
  --model /workspace/models/MinerU2.5-Pro-2605-1.2B
  --layout-model /workspace/models/PP-DocLayoutV3_safetensors
  --limit "${LIMIT:-7}" --output-dir "$mineru_root/output")
printf '%q ' "$mineru_python" "${mineru_args[@]}" > "$mineru_root/command.txt"
printf '\n' >> "$mineru_root/command.txt"
git rev-parse HEAD > "$mineru_root/commit.txt"
printf 'hostname=%s\nASCEND_RT_VISIBLE_DEVICES=%s\n' "$(hostname)" "$ASCEND_RT_VISIBLE_DEVICES" > "$mineru_root/environment.txt"
set +e
"$mineru_python" "${mineru_args[@]}" 2>&1 | tee "$mineru_root/run.log"
mineru_exit=${PIPESTATUS[0]}
printf '%s\n' "$mineru_exit" > "$mineru_root/exit_code.txt"
exit "$mineru_exit"
