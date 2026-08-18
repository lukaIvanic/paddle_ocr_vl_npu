#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ROOT:?set the active height A/B RUN_ROOT}"
: "${COMPILE_CACHE:?set the same compile-cache parent as the probe}"
: "${MONITOR_INTERVAL_S:=2}"
: "${FOCAL_REWRITE:=constant_grouped_all}"

test -s "$RUN_ROOT/pid.txt"
pid="$(cat "$RUN_ROOT/pid.txt")"
log="$RUN_ROOT/run.log"
monitor_log="$RUN_ROOT/live_cache_monitor.log"

snapshot() {
  printf 'UNIREC_HEIGHT_AB_MONITOR timestamp=%s pid=%s alive=%s elapsed_s=%s\n' \
    "$(date --iso-8601=ns)" "$pid" \
    "$(kill -0 "$pid" 2>/dev/null && echo yes || echo no)" \
    "$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  local key directory
  for key in 960x448_b1 960x512_b1; do
    while IFS= read -r directory; do
      printf 'UNIREC_HEIGHT_AB_CACHE bucket=%s directory=%s om_count=%s compiled_module_count=%s newest_files=' \
        "$key" "$directory" \
        "$(find "$directory" -type f -name '*.om' | wc -l)" \
        "$(find "$directory" -type f -name compiled_module | wc -l)"
      find "$directory" -type f -printf '%T@:%s:%P\n' \
        | sort -nr | head -n 3 | tr '\n' ','
      printf '\n'
    done < <(
      find "$COMPILE_CACHE" -type d \
        -name "vision_full_bucket_${key}_float16_*dw${FOCAL_REWRITE}_*wtorchair_internal*" \
        | sort
    )
  done
  if [[ -f "$log" ]]; then
    tail -n 80 "$log" \
      | rg 'UNIREC_VISION_K10_HEIGHT_AB_PHASE|UNIREC_VISION_GRAPH_DIAGNOSTIC|compile|cache' \
      | tail -n 8 || true
  fi
}

{
  while kill -0 "$pid" 2>/dev/null; do
    snapshot
    sleep "$MONITOR_INTERVAL_S"
  done
  snapshot
} | tee -a "$monitor_log"

printf 'MONITOR_LOG=%s\n' "$monitor_log"
