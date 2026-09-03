#!/usr/bin/env bash
# Run only on the Blue Zone 910B. Reuse existing model graphs, never other jobs.
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
source npu-setup
set -u
mineru_status="$(npu-status)"
printf '%s\n' "$mineru_status"
case "$mineru_status" in
  *"NPU 4: free "*) export ASCEND_RT_VISIBLE_DEVICES=4 ;;
  *) echo 'NPU4 is not free; refusing to change the comparison device.' >&2; exit 2 ;;
esac
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mineru_python=/workspace/venvs/mineru_pro_vllm_py312/bin/python
mineru_mode="${MODE:-streaming}"
mineru_limit="${LIMIT:-384}"
mineru_root="${RUN_ROOT:-tmp/11_mineru_2_5_pro_inference/serving_${mineru_mode}_${mineru_limit}_$(date -u +%Y%m%dT%H%M%S)_$(git rev-parse --short HEAD)}"
test ! -e "$mineru_root/output"
mkdir -p "$mineru_root"
exec 9>.runtime_cache/11_mineru_2_5_pro_inference/serving_validation.lock
flock -n 9 || { echo 'A MinerU serving validation already owns these caches.' >&2; exit 2; }
mineru_args=(
  11_mineru_2_5_pro_inference/run_official_transformers_omnidocbench.py
  --backend local-continuous-client
  --model /workspace/models/MinerU2.5-Pro-2605-1.2B
  --dataset-json /workspace/datasets/OmniDocBench/OmniDocBench.json
  --images-dir /workspace/datasets/OmniDocBench/images
  --output-dir "$mineru_root/output"
  --offset 0 --limit "$mineru_limit" --warmup-pages 2 --no-resume --fail-fast
  --batch-size 32 --page-batch-size 32 --global-request-stream
  --layout-image-size 1036 1036 --processor-min-pixels 25088
  --local-dtype float16 --local-compiled-cache-length 4096
  --local-decode-attention increfa --local-decode-weight-format decode_nz
  --local-decode-rotary-impl npu_apply --local-prepare-prefetch-depth 64
  --local-prefill-metrics --local-text-backend torchair-packed
  --local-text-buckets 128,256,512,1024 --local-text-max-members 32
  --local-text-torchair-cache-dir .runtime_cache/11_mineru_2_5_pro_inference/text_prefill_packed_fp16
  --local-vision-attention prompt_flash_attention --local-vision-backend torchair
  --local-vision-buckets 384,512,768,1024,1536,2048,3072,4224,5632
  --local-vision-pack-target 768 --local-vision-lookahead 32
  --local-vision-torchair-cache-dir .runtime_cache/11_mineru_2_5_pro_inference/vision_prefill_b1_fp16_9511b2e
  --local-torchair-cache-dir .runtime_cache/11_mineru_2_5_pro_inference/production_increfa_real_nz_compile
  --token-trace --hash-model-files
)
case "$mineru_mode" in
  anchor) echo 'The unchanged anchor must run at trace-only commit 13061fc4.' >&2; exit 2 ;;
  stepping) mineru_args+=(--no-streaming-pages) ;;
  streaming) mineru_args+=(--streaming-pages --streaming-page-window "${PAGE_WINDOW:-32}") ;;
  *) echo "Unknown validation mode: $mineru_mode" >&2; exit 2 ;;
esac
printf '%q ' "$mineru_python" "${mineru_args[@]}" > "$mineru_root/command.txt"
printf '\n' >> "$mineru_root/command.txt"
git rev-parse HEAD > "$mineru_root/commit.txt"
printf 'RUN_ROOT=%s\n' "$mineru_root"
set +e
"$mineru_python" "${mineru_args[@]}" 2>&1 | tee "$mineru_root/run.log"
mineru_exit=${PIPESTATUS[0]}
printf '%s\n' "$mineru_exit" > "$mineru_root/exit_code.txt"
exit "$mineru_exit"
