#!/usr/bin/env bash
# Run after build succeeds. Each selected package gets a fresh Python process.
set -euo pipefail
OUT=${1:?new output root}
CANDIDATE=${2:?candidate graph vendor}
BASELINE=${3:-/workspace/operators/bge-v2-graph-fixed/vendors/bge_v2_nn}
source npu-setup
export PATH=/usr/local/python3.12.13/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir "$OUT"
{
  git rev-parse HEAD
  hostname
  echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
  printf 'bash %q %q %q %q\n' "$0" "$OUT" "$CANDIDATE" "$BASELINE"
  npu-smi info
  python3 -c 'import sys,importlib.metadata as m; print(sys.version); print({n:m.version(n) for n in ("torch","torch-npu","transformers")})'
  readlink -f "$ASCEND_HOME_PATH"
} > "$OUT/command.txt" 2>&1
select_package() {
  export ASCEND_CUSTOM_OPP_PATH=$1
  export LD_LIBRARY_PATH=$1/op_api/lib:$ORIGINAL_LIBRARY_PATH
}
ORIGINAL_LIBRARY_PATH=$LD_LIBRARY_PATH
run() {
  local phase=$1 lane=$2 repeat=$3
  select_package "$BASELINE"
  if [[ $lane == candidate ]]; then select_package "$CANDIDATE"; fi
  local target="$OUT/${phase}_${lane}_${repeat}"
  local reference=()
  if [[ $lane != baseline || $repeat != 0 ]]; then
    reference=(--reference "$OUT/${phase}_baseline_0")
  fi
  python3 -u 26_bge_m3_inference/gamma_cache/validate_profile.py \
    --op-api "$ASCEND_CUSTOM_OPP_PATH/op_api/lib/libcust_opapi.so" \
    --phase "$phase" --output "$target" "${reference[@]}" > "$target.log" 2>&1
}
# Gated ABBA order on the same physical NPU; no overlap or package collision.
run direct baseline 0
run direct candidate 0
run direct candidate 1
run direct baseline 1
select_package "$CANDIDATE"
python3 -u 26_bge_m3_inference/probe_norm_v2_graph.py \
  --op-api "$ASCEND_CUSTOM_OPP_PATH/op_api/lib/libcust_opapi.so" \
  --rows 256 --output "$OUT/graph_probe" > "$OUT/graph_probe.log" 2>&1
run model baseline 0
run model candidate 0
run model candidate 1
run model baseline 1
echo 0 > "$OUT/exit_code.txt"
