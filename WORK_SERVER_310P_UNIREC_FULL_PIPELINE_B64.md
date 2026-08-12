# 310P UniRec full pipeline, B64 decode

Pull the commit named by Luka and run this task only after the current prefill
A/B process has finished and its physical NPU is clean. This is the first
310P decode test, so the committed background runner gates the full run:

1. exact optimized pipeline on the first 32 pages;
2. require successful compiled IncreFA B64 decode and all crops completed;
3. require at least 40 GiB free in both `/dev/shm` and available system RAM;
4. run all 1,651 pages with the same B64 graph/cache.

The full pipeline is sequential by phase: optimized prefill first, retained
cross-KV in CPU shared memory, all eight workers shut down, then one coordinator
allocates the decode arena and runs continuous decoding. Prefill and decode do
not compete for NPU HBM.

## Decode configuration and memory

```text
batch:          64
self-KV:        1024
cross-KV:       512
self arena:     about 0.563 GiB
cross arena:    about 0.281 GiB
fixed arena:    about 0.844 GiB
attention:      compiled IncreFA
```

Model, graph workspace, and runtime allocations are additional. The fixed B64
arena is nevertheless far below the server's 21 GB HBM budget. The full CPU
cross-KV bank was 32.5 GiB on the matching 910B workload, which is why the
launcher checks CPU shared memory separately.

## Restrictions

- Pull only. Do not edit tracked files, branch, commit, or push.
- Use one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Do not stop another user's process.
- Reuse the exact successful optimized layout cache and all-focal vision cache
  from the current prefill A/B candidate.
- Do not enable per-device prefill event timing.
- Do not enable admission prefetch on this first decode test. It adds another
  NPU stream/event mechanism; direct arena admission is the validated 910B
  control.
- Launch in the background and send Luka the absolute log path immediately.
- Preserve a failed gate. Do not automatically fall back to B32 or alter an
  operator after failure; return the first causal error.

## Launch

Resolve the existing stock OpenOCR assets from the previous successful UniRec
commands on this server. They are required only for final page assembly.

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git status --short --branch
git pull --ff-only origin main
git status --short --branch

source npu-setup
test -n "${ASCEND_RT_VISIBLE_DEVICES:-}"

export PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
export MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
export LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
export OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
export IMAGES_DIR="${IMAGES_DIR:?set the existing OmniDocBench image directory}"
export STOCK_ENCODER="${STOCK_ENCODER:?set the existing UniRec encoder ONNX path}"
export STOCK_DECODER="${STOCK_DECODER:?set the existing UniRec decoder ONNX path}"
export STOCK_TOKENIZER_MAPPING="${STOCK_TOKENIZER_MAPPING:?set the existing tokenizer mapping path}"
export LAYOUT_CACHE="${LAYOUT_CACHE:?reuse the successful optimized-layout cache parent}"
export OPT_COMPILE_CACHE="${OPT_COMPILE_CACHE:?reuse the successful all-focal vision cache}"

launch_output="$(
  bash "$REPO/12_unirec_0_1b_inference/run_310p_full_pipeline_b64_background.sh" 2>&1
)"
printf '%s\n' "$launch_output"
RUN_ROOT="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_ROOT=//p')"
RUN_LOG="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_LOG=//p')"
PID="$(printf '%s\n' "$launch_output" | sed -n \
  's/^UNIREC_310P_FULL_PIPELINE_B64_STARTED pid=\([0-9][0-9]*\).*/\1/p')"
test -n "$RUN_ROOT"
test -n "$RUN_LOG"
test -n "$PID"
test -f "$RUN_LOG"
```

Immediately send Luka:

```text
310P FULL PIPELINE B64 STARTED - pid=<pid>; run_log=<absolute path>; tail_command=tail -f <absolute path>
```

Use `tail -f "$RUN_LOG"`. Prefill prints every completed page and a worker
heartbeat after 15 silent seconds. Decode prints every completed/written page
and a 15-second crop/page heartbeat.

## Completion report

```bash
test -f "$RUN_ROOT/exit_code.txt"
cat "$RUN_ROOT/gate_check.log" 2>/dev/null || true
cat "$RUN_ROOT/full_memory_check.log" 2>/dev/null || true
cat "$RUN_ROOT/full_summary.log" 2>/dev/null || true
```

Return all available `UNIREC_310P_B64_GATE`, `UNIREC_310P_FULL_MEMORY`, and
`UNIREC_310P_FULL_PIPELINE_B64` lines. Also state the commit, physical NPU,
runtime versions, process wall time, absolute run log, B64 decode cache path,
and both summary JSON paths.

If the gate fails, report the last completed phase and first causal error, then
stop. A B32 fallback is a subsequent experiment, not an automatic retry.
