#!/usr/bin/env bash
set -euo pipefail

: "${CHIP:?Set CHIP=910B or CHIP=310P; chip labels are verified by the worker}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
commit="$(git -C "$repo_root" rev-parse HEAD)"
run_name="${RUN_NAME:-portable}"
if [[ ! "$run_name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo 'RUN_NAME must contain only letters, numbers, underscores and hyphens' >&2
  exit 1
fi
run_dir="$repo_root/tmp/11_mineru_2_5_pro_inference/kv_cache_${CHIP}_${run_name}_$(date -u +%Y%m%dT%H%M%SZ)_${commit:0:8}"
mkdir -p "$(dirname "$run_dir")"
mkdir "$run_dir"
command=("${PYTHON:-python3}" "$script_dir/probe_attention.py"
  --chip "$CHIP" --output "$run_dir/summary.json"
  --operators "${OPERATORS:-increfa,fia,fia2}"
  --batches "${BATCHES:-1,16}" --contexts "${CONTEXTS:-768}"
  --patterns "${PATTERNS:-ragged}" --cache-length "${CACHE_LENGTH:-4096}"
  --block-size "${BLOCK_SIZE:-128}" --warmup "${WARMUP:-5}"
  --samples "${SAMPLES:-30}" --calls-per-sample "${CALLS_PER_SAMPLE:-10}"
  --timeout "${CASE_TIMEOUT:-180}")
{
  echo "commit=$commit"
  echo "hostname=$(hostname)"
  echo "chip=$CHIP"
  echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-unset}"
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} > "$run_dir/command.txt"
echo "RUN_DIR=$run_dir"
set +e
"${command[@]}" 2>&1 | tee "$run_dir/run.log"
status=${PIPESTATUS[0]}
set -e
echo "$status" > "$run_dir/exit_code.txt"
echo "OUTPUT_JSON=$run_dir/summary.json"
exit "$status"
