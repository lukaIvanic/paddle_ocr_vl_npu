#!/usr/bin/env bash
set -eo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
source /usr/local/bin/npu-setup
set -u
if [[ -n ${PHYSICAL_NPU:-} ]]; then
  export ASCEND_RT_VISIBLE_DEVICES=$PHYSICAL_NPU
  echo "Overriding selected device with physical NPU $PHYSICAL_NPU"
fi

python_bin=${PYTHON_BIN:-/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python}
lab=09_persistent_page_engine/scripts/text_decode_lab.py
batch_size=${BATCH_SIZE:-4}
cache_length=${CACHE_LENGTH:-1024}
profile_position=${PROFILE_POSITION:-768}
warmup=${WARMUP:-3}
repeats=${REPEATS:-20}
cooldown_s=${COOLDOWN_S:-10}
cache_dir=${CACHE_DIR:-.runtime_cache/09_persistent_page_engine_torchair}
output_root=${OUTPUT_ROOT:-tmp/09_persistent_page_engine/text_decode_lab/optimization_matrix}
optimizations=${OPTIMIZATIONS:-"baseline mrope_hoist packed_qkv npu_rms_norm npu_apply_rotary npu_rotary_mul packed_mlp packed_mlp_swiglu npu_add_rms_norm"}

mkdir -p "$output_root"

echo "batch_size=$batch_size"
echo "cache_length=$cache_length"
echo "profile_position=$profile_position"
echo "warmup=$warmup"
echo "repeats=$repeats"
echo "cooldown_s=$cooldown_s"
echo "cache_dir=$cache_dir"
echo "optimizations=$optimizations"

for optimization in $optimizations; do
  echo "START optimization=$optimization"
  "$python_bin" "$lab" \
    --mode profile \
    --batch-size "$batch_size" \
    --cache-length "$cache_length" \
    --profile-position "$profile_position" \
    --warmup "$warmup" \
    --repeats "$repeats" \
    --cache-dir "$cache_dir" \
    --decode-optimization "$optimization" \
    --allow-compile \
    --output "$output_root/${optimization}_b${batch_size}_k${cache_length}.json"
  echo "DONE optimization=$optimization"
  sleep "$cooldown_s"
done

echo "OPTIMIZATION_MATRIX_COMPLETE"
