# 310P UniRec production-faithful layout lab

Pull the commit containing this brief and run this task only. This measures the
same optimized PP-DocLayoutV2 B1 boundary used by the current UniRec page
producer. It is not an eager proxy and it does not include recognition.

## Exact contract

- OmniDocBench offset 0, first 128 pages;
- production Kornia-RS PNG / TorchVision non-PNG decode;
- production canonical contiguous RGB page with no full-page channel reversal;
- production adapter, 0.4 threshold, and OpenDoc result ordering;
- TorchAir static B1, FP16 model and FP16 reading-order head;
- `group16`, `torchair_internal`, and preformatted FrozenBN buffers;
- one excluded warmup call, then 128 measured pages;
- exact synchronized model-forward time plus surrounding layout stage times.

The lab rejects any conflicting model flag. Do not add profiling on this pass.

## Restrictions

- Pull only. Do not edit tracked files, branch, commit, or push.
- Use one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Do not stop another user's process.
- Reuse the warmed optimized-layout cache from the successful production run.
- Launch in the background and immediately send Luka the absolute log path.

## Launch

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main
source npu-setup

export PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
export LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
export OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
export IMAGES_DIR="${IMAGES_DIR:?reuse the existing OmniDocBench images directory}"
export LAYOUT_CACHE="${LAYOUT_CACHE:?reuse the warmed group16 internal buffer cache parent}"

launch_output="$(
  bash "$REPO/12_unirec_0_1b_inference/run_layout_production_lab_background.sh" 2>&1
)"
printf '%s\n' "$launch_output"
RUN_ROOT="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_ROOT=//p')"
RUN_LOG="$(printf '%s\n' "$launch_output" | sed -n 's/^RUN_LOG=//p')"
test -n "$RUN_ROOT"
test -n "$RUN_LOG"
test -f "$RUN_LOG"
```

Immediately send Luka:

```text
310P LAYOUT LAB STARTED - run_log=<absolute path>; tail_command=tail -f <absolute path>
```

The log prints every completed page, so a stall is visible directly.

## Completion report

```bash
test -f "$RUN_ROOT/exit_code.txt"
test "$(cat "$RUN_ROOT/exit_code.txt")" = 0
cat "$RUN_ROOT/report.log"
```

Return the three `UNIREC_LAYOUT_PRODUCTION_*` lines plus commit, physical NPU,
CANN, torch, torch_npu, absolute `run.log`, and `result.json`. Stop after this
lab. Do not start recognition, decode, or NPU profiling.
