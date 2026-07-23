#!/usr/bin/env bash
set -eo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 CACHE_LENGTH PROFILE_POSITION PHYSICAL_NPU CACHE_DIR LOG_PATH" >&2
  exit 2
fi

cache_length=$1
profile_position=$2
physical_npu=$3
cache_dir=$4
log_path=$5

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
source /usr/local/bin/npu-setup
set -u
export ASCEND_RT_VISIBLE_DEVICES="$physical_npu"

mkdir -p "$(dirname "$log_path")"
exec >"$log_path" 2>&1

echo "matrix_context=$cache_length"
echo "profile_position=$profile_position"
echo "physical_npu=$physical_npu"
echo "cache_dir=$cache_dir"

python_bin=/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python
lab=09_persistent_page_engine/scripts/text_decode_lab.py
output_root=tmp/09_persistent_page_engine/text_decode_lab/decode_speed_matrix

for batch_size in 4 8 16 32 64; do
  echo "START batch_size=$batch_size cache_length=$cache_length"
  "$python_bin" "$lab" \
    --mode profile \
    --batch-size "$batch_size" \
    --cache-length "$cache_length" \
    --profile-position "$profile_position" \
    --warmup 3 \
    --repeats 20 \
    --cache-dir "$cache_dir" \
    --allow-compile \
    --output "$output_root/b${batch_size}_k${cache_length}.json"
  echo "DONE batch_size=$batch_size cache_length=$cache_length"
done

echo "MATRIX_COMPLETE cache_length=$cache_length"
