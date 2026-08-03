#!/usr/bin/env bash
set -o pipefail

cd /workspace/repos/paddle_ocr_vl_npu
source npu-setup
set -u

PY=/usr/local/python3.12.13/bin/python3
LAB=09_persistent_page_engine/scripts/text_decode_lab.py
MODEL=/workspace/models/PaddleOCR-VL-1.6
COMMIT=$(git rev-parse --short HEAD)
ROOT=tmp/09_persistent_page_engine/910b_decode_length_modes_b16_128_k2048_${COMMIT}
CACHE=.runtime_cache/910b_decode_length_modes_k2048_${COMMIT}

if test -e "$ROOT"; then
  echo "ERROR artifact root exists: $ROOT"
  exit 2
fi
mkdir -p "$ROOT" "$CACHE"

{
  echo "commit=$(git rev-parse HEAD)"
  echo "host=$(hostname)"
  echo "physical_npu=$ASCEND_RT_VISIBLE_DEVICES"
  "$PY" -c 'import platform, torch, torch_npu; print("python="+platform.python_version()); print("torch="+torch.__version__); print("torch_npu="+torch_npu.__version__)'
  npu-smi info
} > "$ROOT/preflight.log" 2>&1

printf 'MATRIX_START root=%s cache=%s physical_npu=%s time=%s\n' \
  "$ROOT" "$CACHE" "$ASCEND_RT_VISIBLE_DEVICES" "$(date -Is)" \
  | tee "$ROOT/progress.log"

for B in 16 32 64 128; do
  for OPT in \
    combined_apply \
    combined_apply_static_actual \
    combined_apply_pse_sentinel
  do
    LANE=b${B}_${OPT}
    DIR="$ROOT/$LANE"
    mkdir -p "$DIR"
    printf 'LANE_START lane=%s time=%s\n' "$LANE" "$(date -Is)" \
      | tee -a "$ROOT/progress.log"
    printf '%q ' \
      timeout --signal=TERM --kill-after=15s 1800 \
      "$PY" "$LAB" \
      --mode profile --backend torchair --allow-compile \
      --model "$MODEL" --cache-dir "$CACHE" \
      --batch-size "$B" --active-slots "$B" --cache-length 2048 \
      --profile-position 1024 --warmup 3 --repeats 30 \
      --decode-optimization "$OPT" \
      --output "$DIR/result.json" \
      > "$DIR/command.sh"
    printf '\n' >> "$DIR/command.sh"

    set +e
    timeout --signal=TERM --kill-after=15s 1800 \
      "$PY" "$LAB" \
      --mode profile --backend torchair --allow-compile \
      --model "$MODEL" --cache-dir "$CACHE" \
      --batch-size "$B" --active-slots "$B" --cache-length 2048 \
      --profile-position 1024 --warmup 3 --repeats 30 \
      --decode-optimization "$OPT" \
      --output "$DIR/result.json" \
      2>&1 | tee "$DIR/run.log"
    RC=${PIPESTATUS[0]}
    set -e
    echo "$RC" > "$DIR/exit_code.txt"
    printf 'LANE_END lane=%s exit=%s time=%s\n' \
      "$LANE" "$RC" "$(date -Is)" | tee -a "$ROOT/progress.log"
    if test "$RC" -ne 0; then
      printf 'MATRIX_STOP lane=%s exit=%s\n' "$LANE" "$RC" \
        | tee -a "$ROOT/progress.log"
      exit "$RC"
    fi
  done
done

printf 'PROFILES_COMPLETE time=%s\n' "$(date -Is)" \
  | tee -a "$ROOT/progress.log"
