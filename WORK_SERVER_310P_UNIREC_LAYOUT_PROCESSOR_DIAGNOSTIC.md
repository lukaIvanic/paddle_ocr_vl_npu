# 310P UniRec layout processor diagnostic

Use the commit containing this brief. This task confirms why the faithful W1
layout report attributed about 78 ms/page to `processor_preprocess_s`.

## Important finding from 910B2

The production processor is CPU-only. For every B1 page it performs:

1. wrap the contiguous RGB NumPy array as a Torch tensor;
2. create an HWC-to-CHW view;
3. copy that view into contiguous CHW storage;
4. resize uint8 BCHW to 800x800 with torchvision bicubic and
   `antialias=False`;
5. return the already-contiguous B1 tensor.

The new diagnostic preserves this exact math and reports each operation.

On 910B2, first 128 OmniDocBench pages, the faithful integrated layout lab
reported:

```text
torch intra-op / inter-op threads: 192 / 192
whole layout wall mean:             34.343 ms/page
compiled model forward:             12.659 ms/page
processor total:                     5.138 ms/page
  bicubic resize:                     3.289 ms/page
  CHW contiguous copy:                1.771 ms/page
  all other processor work:          ~0.078 ms/page
```

The CPU-only exact thread sweep reported:

```text
threads   processor   CHW copy   bicubic resize   exact vs native
native192   5.553 ms    1.664 ms       3.773 ms     true
1          71.617 ms   15.896 ms      55.601 ms     true
2          36.427 ms    7.922 ms      28.384 ms     true
4          20.332 ms    5.268 ms      14.932 ms     true
8          12.858 ms    5.441 ms       7.274 ms     true
```

The reported 310P processor time of about 78 ms is very close to the 910B2
one-thread result. The primary hypothesis is that the 310P process has one
PyTorch intra-op thread, or is equivalently CPU-constrained/contended. This
must be measured, not assumed.

## Restrictions

- Pull only. Do not edit tracked files, commit, push, or create a branch.
- Do not run while another CPU-heavy job is active. Record `uptime` and the top
  CPU processes first. If the host is busy, wait.
- Use one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Do not stop another user's process.
- Do not change the production thread count before the first integrated run.
- Run in the background and immediately send Luka the absolute log path.

## Environment

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main

uptime
ps -eo pid,pcpu,pmem,comm,args --sort=-pcpu | head -n 20

source npu-setup
case ",${ASCEND_RT_VISIBLE_DEVICES:-}," in
  *,5,*|*,6,*) echo "REFUSE_NPU_5_OR_6" >&2; exit 1 ;;
esac

export PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
export OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
export LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
export IMAGES_DIR="${IMAGES_DIR:?reuse the existing OmniDocBench images directory}"
export LAYOUT_CACHE="${LAYOUT_CACHE:?reuse the warmed optimized-layout cache parent}"

RUN_ROOT="$REPO/temp/12_unirec_0_1b_inference/layout_processor_diag_$(git rev-parse --short HEAD)_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$RUN_ROOT"
```

If the server uses a different passed Python/OpenOCR/model location, resolve it
from the latest successful UniRec W1 run. Do not search for ONNX exports.

## Background launch

Create a small run script in the untracked run directory, then launch it:

```bash
cat >"$RUN_ROOT/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO"

"$PYTHON_BIN" 12_unirec_0_1b_inference/layout_detector_lab.py \
  --contract current_production \
  --openocr-root "$OPENOCR_ROOT" \
  --model-path "$LAYOUT_MODEL" \
  --input "$IMAGES_DIR" \
  --output "$RUN_ROOT/integrated_native.json" \
  --device npu:0 \
  --compile-cache-dir "$LAYOUT_CACHE" \
  --offset 0 --limit 128 --warmup-pages 1

"$PYTHON_BIN" 12_unirec_0_1b_inference/layout_processor_lab.py \
  --openocr-root "$OPENOCR_ROOT" \
  --input "$IMAGES_DIR" \
  --output "$RUN_ROOT/processor_threads.json" \
  --offset 0 --limit 128 --warmup-calls 3 \
  --thread-counts native,1,2,4,8
EOF
chmod +x "$RUN_ROOT/run.sh"
export REPO PYTHON_BIN OPENOCR_ROOT LAYOUT_MODEL IMAGES_DIR LAYOUT_CACHE RUN_ROOT
nohup "$RUN_ROOT/run.sh" >"$RUN_ROOT/run.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" >"$RUN_ROOT/pid"
printf '310P LAYOUT PROCESSOR STARTED pid=%s log=%s\n' "$PID" "$RUN_ROOT/run.log"
printf 'tail -f %q\n' "$RUN_ROOT/run.log"
```

The heredoc is intentionally quoted. The exported variables are inherited by
the background process.

## Required report

After the owned PID exits successfully, return:

1. commit, physical NPU, CPU model, logical CPU count, CANN, torch, and
   torch_npu versions;
2. host load and top CPU processes before the run;
3. `integrated_native.json`:
   - `config.cpu_runtime`;
   - page wall mean and pages/s;
   - processor total and every `processor_*` substage mean/p90;
   - model-forward mean/p90;
4. every `LAYOUT_PROCESSOR_LAB_RESULT` line;
5. whether every thread lane is byte-exact versus native;
6. absolute `run.log`, `integrated_native.json`, and
   `processor_threads.json` paths.

Interpretation:

- If native PyTorch threads are 1 and the thread sweep improves, identify the
  best non-oversubscribed lane. Do not silently promote it yet.
- If native threads are already greater than 1 but the processor still takes
  about 78 ms, inspect CPU contention, affinity/cgroup quotas, and CPU frequency.
- If the integrated processor timer and CPU-only native timer disagree by more
  than 20%, report both and inspect contention or asynchronous attribution.
- `processor_preprocess_s` should approximately equal the sum of its new
  `processor_*` children. Small timer-call overhead is expected.

If one thread is the cause and a best lane is clear, run one additional
integrated diagnostic with only this added option:

```text
--torch-cpu-threads <best>
```

Report the new whole-layout wall time and exact result-digest comparison against
`integrated_native.json`. Do not modify production defaults in this task.
