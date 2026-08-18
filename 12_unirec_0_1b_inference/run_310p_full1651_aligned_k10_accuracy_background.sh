#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Full accuracy lane using the corrected aligned K10 graphs. The latest
# first-128 gate must already have populated both direct-2D flat-global-context
# graphs and the exact B128 decode graph. This wrapper never repairs a cold
# vision cache implicitly.
: "${UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE:?export the passed B128 decode-cache parent}"
export RUN_VARIANT=optimized_k10_l4_aligned
export REQUIRE_WARM_VISION_CACHE=1
export DECODE_CACHE_GATE_ATTEMPTS=1
export ALLOW_LOW_HOST_MEMORY=1
export LAYOUT_CPU_THREADS="${LAYOUT_CPU_THREADS:-16}"
export CPUSET="${CPUSET:-0-63}"
export PROGRESS_EVERY_PAGES="${PROGRESS_EVERY_PAGES:-16}"
export MATCH_WORKERS="${MATCH_WORKERS:-64}"
export TEDS_WORKERS="${TEDS_WORKERS:-64}"
export CDM_WORKERS="${CDM_WORKERS:-64}"

exec "$SCRIPT_DIR/run_310p_full1651_w4t8_accuracy_background.sh" "$@"
