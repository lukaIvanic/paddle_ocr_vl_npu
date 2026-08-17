#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Accuracy-safe full production lane:
# - eager FP32 native layout, because the compiled FP16 layout changed crops;
# - optimized K10/L4 vision with all 45 focal weights prepacked;
# - B128 decode with cross-KV 1320 and self-KV 2048.
#
# The 310P container can expose only about 64 GiB through /dev/shm while the
# bare-metal host has more RAM. Record the memory state, but let the real run
# determine whether it fits instead of rejecting the host preflight.
export RUN_VARIANT=optimized_k10_l4
export ALLOW_LOW_HOST_MEMORY=1
export LAYOUT_CPU_THREADS="${LAYOUT_CPU_THREADS:-16}"
export CPUSET="${CPUSET:-0-63}"
export PROGRESS_EVERY_PAGES="${PROGRESS_EVERY_PAGES:-16}"
export MATCH_WORKERS="${MATCH_WORKERS:-64}"
export TEDS_WORKERS="${TEDS_WORKERS:-64}"
export CDM_WORKERS="${CDM_WORKERS:-64}"

exec "$SCRIPT_DIR/run_310p_full1651_w4t8_accuracy_background.sh" "$@"
