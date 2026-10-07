#!/usr/bin/env bash
set -uo pipefail
: "${MODEL:?Set MODEL to the existing local MinerU2.5-Pro checkpoint}"
: "${PYTHON:?Set PYTHON to the working torch-npu/TorchAir interpreter}"
: "${CHIP:?Set CHIP=910B or CHIP=310P}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
commit="$(git -C "$repo" rev-parse HEAD)"
run_name="${RUN_NAME:-full_model_kv}"
if [[ ! "$run_name" =~ ^[a-zA-Z0-9_-]+$ ]]; then exit 1; fi
run_dir="$repo/tmp/11_mineru_2_5_pro_inference/${run_name}_${CHIP}_$(date -u +%Y%m%dT%H%M%SZ)_${commit:0:8}"
mkdir -p "$(dirname "$run_dir")"
mkdir "$run_dir" || exit 1
command=("$PYTHON" "$script_dir/bench_full_model_kv.py" --model "$MODEL"
  --manifest "${MANIFEST:-$repo/crops/hotswap_100_manifest.json}"
  --limit "${LIMIT:-8}" --batch-size "${BATCH_SIZE:-4}"
  --max-new-tokens "${MAX_NEW_TOKENS:-512}" --cache-length "${CACHE_LENGTH:-4096}"
  --block-size "${BLOCK_SIZE:-128}" --repeats "${REPEATS:-5}" --warmup-steps "${WARMUP_STEPS:-5}"
  --variants "${VARIANTS:-increfa_nd,fia_nd,fia_blocked_nd,increfa_nz,fia_nz}"
  --chip "$CHIP" --output "$run_dir/summary.json" --cache-dir "$run_dir/compile_cache")
if [[ "${PROFILE:-0}" == 1 ]]; then command+=(--profile); fi
{
  echo "commit=$commit"
  echo "hostname=$(hostname)"
  echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-unset}"
  printf 'command='; printf '%q ' "${command[@]}"; printf '\n'
} > "$run_dir/command.txt"
npu-smi info > "$run_dir/occupancy_before.txt" 2>&1
echo "RUN_DIR=$run_dir"
timeout --signal=TERM --kill-after=20s "${RUN_TIMEOUT:-1800}s" "${command[@]}" 2>&1 | tee "$run_dir/run.log"
status=${PIPESTATUS[0]}
echo "$status" > "$run_dir/exit_code.txt"
npu-smi info > "$run_dir/occupancy_after.txt" 2>&1
echo "OUTPUT_JSON=$run_dir/summary.json"
exit "$status"
