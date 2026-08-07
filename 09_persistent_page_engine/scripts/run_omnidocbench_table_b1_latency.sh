#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python}"
OUTPUT_DIR="${1:-$REPO_ROOT/tmp/09_persistent_page_engine/table_b1_latency}"
PORT="${PORT:-8767}"
DATASET_JSON="${DATASET_JSON:-/workspace/datasets/OmniDocBench/OmniDocBench.json}"
IMAGES_DIR="${IMAGES_DIR:-/workspace/datasets/OmniDocBench/images}"
EVALUATOR_ROOT="${EVALUATOR_ROOT:-/workspace/repos/OmniDocBench_eval}"
TEDS_WORKERS="${TEDS_WORKERS:-12}"
TEDS_TIMEOUT_S="${TEDS_TIMEOUT_S:-120}"
OFFSET="${OFFSET:-0}"
LIMIT_PAGES="${LIMIT_PAGES:-}"

mkdir -p "$OUTPUT_DIR"
SERVER_LOG="$OUTPUT_DIR/server.log"
CLIENT_LOG="$OUTPUT_DIR/client.log"

cd "$REPO_ROOT"

"$PYTHON" 09_persistent_page_engine/scripts/serve_crop_ocr_api.py \
  --host 127.0.0.1 \
  --port "$PORT" \
  --decode-batch-size 1 \
  --cache-length 4096 \
  --max-new-tokens 4096 \
  --decode-backend torchair \
  --decode-optimization combined_apply_pse_sentinel \
  --min-pixels 28224 \
  --max-pixels 802816 \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
    wait "$SERVER_PID" || true
  fi
}
trap cleanup EXIT INT TERM

"$PYTHON" - "$PORT" "$SERVER_PID" "$SERVER_LOG" <<'PY'
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

port, pid, log_path = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
deadline = time.monotonic() + 900
last_report = 0.0
while time.monotonic() < deadline:
    if not Path(f"/proc/{pid}").exists():
        raise SystemExit(f"API server exited before readiness; inspect {log_path}")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=2) as response:
            payload = json.load(response)
        if payload.get("ready"):
            print(f"B1 API ready worker_pid={payload.get('worker_pid')}", flush=True)
            break
    except Exception:
        pass
    now = time.monotonic()
    if now - last_report >= 15:
        print(f"Waiting for B1 API setup elapsed_s={900 - (deadline - now):.0f}", flush=True)
        last_report = now
    time.sleep(1)
else:
    raise SystemExit(f"Timed out waiting for B1 API; inspect {log_path}")
PY

CLIENT_ARGS=(
  --omnidocbench
  --crop-type table
  --dataset-json "$DATASET_JSON"
  --images-dir "$IMAGES_DIR"
  --evaluator-root "$EVALUATOR_ROOT"
  --api-url "http://127.0.0.1:$PORT/v1/ocr"
  --http-workers 1
  --teds-workers "$TEDS_WORKERS"
  --teds-timeout-s "$TEDS_TIMEOUT_S"
  --offset "$OFFSET"
  --no-drain-server
  --no-resume
  --output-dir "$OUTPUT_DIR/client"
)
if [[ -n "$LIMIT_PAGES" ]]; then
  CLIENT_ARGS+=(--limit-pages "$LIMIT_PAGES")
fi

"$PYTHON" 09_persistent_page_engine/scripts/run_omnidocbench_table_api.py \
  "${CLIENT_ARGS[@]}" \
  2>&1 | tee "$CLIENT_LOG"

echo "B1 table latency benchmark complete: $OUTPUT_DIR/client/summary.md"
