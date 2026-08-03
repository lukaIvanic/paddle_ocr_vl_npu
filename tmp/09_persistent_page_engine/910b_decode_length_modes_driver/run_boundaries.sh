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

test -d "$ROOT"
test -d "$CACHE"
printf 'BOUNDARIES_START physical_npu=%s time=%s\n' \
  "$ASCEND_RT_VISIBLE_DEVICES" "$(date -Is)" | tee -a "$ROOT/progress.log"

for B in 16 32 64 128; do
  for OPT in combined_apply_static_actual combined_apply_pse_sentinel; do
    LANE=boundary_b${B}_${OPT}
    DIR="$ROOT/$LANE"
    mkdir -p "$DIR"
    printf 'BOUNDARY_START lane=%s time=%s\n' "$LANE" "$(date -Is)" \
      | tee -a "$ROOT/progress.log"
    printf '%q ' \
      timeout --signal=TERM --kill-after=15s 300 \
      "$PY" "$LAB" \
      --mode boundary --backend torchair \
      --model "$MODEL" --cache-dir "$CACHE" \
      --batch-size "$B" --active-slots "$B" --cache-length 2048 \
      --profile-position 1279 \
      --decode-optimization "$OPT" \
      --output "$DIR/result.json" \
      > "$DIR/command.sh"
    printf '\n' >> "$DIR/command.sh"

    set +e
    timeout --signal=TERM --kill-after=15s 300 \
      "$PY" "$LAB" \
      --mode boundary --backend torchair \
      --model "$MODEL" --cache-dir "$CACHE" \
      --batch-size "$B" --active-slots "$B" --cache-length 2048 \
      --profile-position 1279 \
      --decode-optimization "$OPT" \
      --output "$DIR/result.json" \
      2>&1 | tee "$DIR/run.log"
    RC=${PIPESTATUS[0]}
    set -e
    echo "$RC" > "$DIR/exit_code.txt"
    printf 'BOUNDARY_END lane=%s exit=%s time=%s\n' \
      "$LANE" "$RC" "$(date -Is)" | tee -a "$ROOT/progress.log"
    if test "$RC" -ne 0; then
      printf 'BOUNDARIES_STOP lane=%s exit=%s\n' "$LANE" "$RC" \
        | tee -a "$ROOT/progress.log"
      exit "$RC"
    fi
  done
done

printf 'BOUNDARIES_COMPLETE time=%s\n' "$(date -Is)" \
  | tee -a "$ROOT/progress.log"
