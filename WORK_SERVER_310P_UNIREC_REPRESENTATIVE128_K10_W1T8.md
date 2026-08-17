# 310P UniRec representative-128 K10 W1/T8 follow-up

## Objective

Repeat the corrected K10 representative-128 prefill experiment with exactly one
process worker and eight recognition-preprocess threads. The earlier W1/T8 run
was only about 1% faster than W1/T1 and did not prove that eight crop tasks ran
concurrently. This rerun must collect direct native-thread and CPU evidence.
Compare it with the completed corrected W1/T1 run. Do not run decode or
evaluation.

Use the commit containing this brief or later. It must include the
`fd24c1b` eager-fallback warmup fix.

## Fixed contract

- committed `unirec_representative_128_v1` pages;
- W1/T8, layout B1, one-page vision lookahead;
- K10 `310p_k10_l1` vision buckets;
- optimized compiled vision: `constant_grouped_all` plus
  `torchair_internal` weights;
- optimized compiled layout without native MSDA;
- threshold 0.5, cross-KV 1320, self-KV/max length 2048;
- compact uint8 HWC recognition inputs;
- trace lane followed by clean lane;
- prefill only.

The eight-thread setting applies only to recognition crop preprocessing. The
launcher pins OpenMP, MKL, OpenBLAS, and NumExpr to one thread each. Do not
change those pins and do not let `nproc` select a thread count.

The trace now records every recognition crop task's native thread ID, start/end
CPU, thread CPU time, and monotonic interval. The report calculates the native
thread count, maximum concurrent tasks, CPUs observed, task interval union, and
average CPU cores used during crop-active windows. A CLI value of `8` without
this runtime evidence is not a valid W1/T8 result.

Do not use `nproc` to decide whether eight CPUs are available. GNU `nproc`
honors `OMP_NUM_THREADS=1` and can therefore print `1` even when the process has
many CPUs available. The runner uses `len(os.sched_getaffinity(0))` as the
authoritative check and refuses W1/T8 if fewer than eight CPUs are allowed.

## Work-server rules

- Pull only. Do not edit tracked files, create a branch, commit, or push.
- Use one free physical 310P from 0 through 3.
- Do not run `npu-setup`; it is not installed on this server.
- Preserve the validated `python_nosym` executable path. Do not apply
  `readlink -f` to it.
- Preserve the completed W1/T1 output and reuse its exact recognition and
  layout cache roots.
- Do not delete or rename caches. The ten K10 compiled modules and OMs must
  load without recompilation or new OMs.

## Pull and preflight

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main
git rev-parse HEAD

test -x "${PYTHON_BIN:?validated python_nosym executable}"
test "$(basename "$PYTHON_BIN")" = python_nosym
export MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
export LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
export OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/../OpenOCR}"
export IMAGES_DIR="${IMAGES_DIR:?OmniDocBench v1.6 images directory}"
export COMPILE_CACHE="${COMPILE_CACHE:?reuse corrected W1/T1 recognition cache}"
export LAYOUT_CACHE_ROOT="${LAYOUT_CACHE_ROOT:?reuse corrected W1/T1 layout cache}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:?one free device 0..3}"

case "$ASCEND_RT_VISIBLE_DEVICES" in
  0|1|2|3) ;;
  *) echo "Expected one 310P device in 0..3" >&2; exit 1 ;;
esac

test -f "$MODEL/model.pth"
test -d "$LAYOUT_MODEL"
test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
test -d "$IMAGES_DIR"
test -d "$COMPILE_CACHE"
test -d "$LAYOUT_CACHE_ROOT"
"$PYTHON_BIN" -c \
  'import os; a=sorted(os.sched_getaffinity(0)); print("CPU_AFFINITY", len(a), a)'
```

## Launch in background

```bash
export UNIREC_K10_CHIP_LABEL=310P
export UNIREC_K10_ALLOWED_DEVICES=0,1,2,3
export UNIREC_K10_RUN_MODE=both
export UNIREC_K10_THREADS=8
bash 12_unirec_0_1b_inference/run_910b_representative128_k10_l1_background.sh
```

Immediately return the printed absolute `RUN_ROOT`, `RUN_LOG`, PID, and
`tail -f` command. The cache source hash is unchanged. Every bucket must retain
exactly one `compiled_module` and one OM before and after the process restart.

## Required completion report

Wait for `exit_code.txt`; success requires zero. Paste back:

1. commit, physical NPU, CANN, torch, torch-npu, and the complete
   `CPU_AFFINITY` line;
2. complete `UNIREC_310P_K10_L1_RESULT` line; it must say `threads=8`;
3. complete `UNIREC_310P_K10_L1_STAGES`, `BUCKET_CALLS`, `LAYOUT`,
   `FALLBACK_WARMUP`, `CPU_EXECUTION`, and `CACHE` lines;
4. trace and clean setup, warmup, measured-prefill, and shutdown times;
5. all ten cache-open durations and confirmation that no new OM appeared;
6. crop/rejection/token counts and peak HBM;
7. absolute run root/log paths and final NPU state;
8. the corrected W1/T1 result line from the immediately preceding run, so the
   W1/T8 speedup is explicit.

`UNIREC_310P_K10_L1_CPU_EXECUTION` must report:

- `configured_threads=8`;
- worker affinity of at least eight CPUs;
- `native_threads=8`;
- measured overlap (`max_concurrent_tasks`);
- the actual CPUs observed at crop-task boundaries;
- summed thread CPU time and active-window wall time.

If the line does not prove eight native threads, do not interpret the timing as
a W1/T8 result. Report it as an invalid lane and stop.

The stage report includes aggregate layout time, layout model-forward time,
layout processor time, compiled vision, eager fallback, crop preprocessing,
text-prefill wall/device, and shared packing. Those fields are nested and must
not be added to reconstruct wall time.

Stop after this report.
