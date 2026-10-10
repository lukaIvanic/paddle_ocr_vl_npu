#!/usr/bin/env bash
# Run after profiled processes have exited; debug logging stays out of profiles.
set -eo pipefail
OUT=${1:?existing run root}
CANDIDATE=${2:?candidate graph vendor}
BASELINE=${3:-/workspace/operators/bge-v2-graph-fixed/vendors/bge_v2_nn}
source npu-setup
set -u
export PATH=/usr/local/python3.12.13/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
ORIGINAL_LIBRARY_PATH=$LD_LIBRARY_PATH
{
  git rev-parse HEAD
  printf 'bash %q %q %q %q\n' "$0" "$OUT" "$CANDIDATE" "$BASELINE"
  echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
  cat "$ASCEND_HOME_PATH/aarch64-linux/data/platform_config/Ascend910B2.ini"
} > "$OUT/dispatch_command.txt"
for lane in baseline candidate; do
  vendor=$BASELINE
  if [[ $lane == candidate ]]; then vendor=$CANDIDATE; fi
  export ASCEND_CUSTOM_OPP_PATH=$vendor
  export LD_LIBRARY_PATH=$vendor/op_api/lib:$ORIGINAL_LIBRARY_PATH
  {
    echo "$vendor"
    sha256sum "$vendor/op_api/lib/libcust_opapi.so" \
      "$vendor/op_impl/ai_core/tbe/kernel/ascend910b/add_layer_norm_quant_v2/AddLayerNormQuantV2_fp16_high_performance.o" \
      "$vendor/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info.json"
    readelf -sW "$vendor/op_impl/ai_core/tbe/kernel/ascend910b/add_layer_norm_quant_v2/AddLayerNormQuantV2_fp16_high_performance.o" | grep -E 'FUNC.*performance_100[012]$'
    nm -D "$vendor/op_api/lib/libcust_opapi.so" | grep -E ' T aclnnAddLayerNormQuantV2(GetWorkspaceSize)?$'
    cat "$vendor/bge_graph_manifest.json"
  } > "$OUT/${lane}_binary_provenance.txt"
  ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1 \
    python3 -u 26_bge_m3_inference/gamma_cache/dispatch_probe.py \
    --op-api "$vendor/op_api/lib/libcust_opapi.so" > "$OUT/${lane}_dispatch.log" 2>&1
  grep -E '"pid"|maxUbSize|numCore =|rowPerTime =|tilingKey =|tilingType =|tiling_key|tiling key|TilingKey|AddLayerNormQuantV2_fp16_high_performance|bge_v2.*libcust|bge_v2.*liboptiling' \
    "$OUT/${lane}_dispatch.log" > "$OUT/${lane}_dispatch_excerpt.txt"
done
