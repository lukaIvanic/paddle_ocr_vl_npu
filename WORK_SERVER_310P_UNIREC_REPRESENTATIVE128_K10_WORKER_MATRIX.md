# 310P UniRec representative-128 K10 W2/W4 CPU-worker matrix

## Objective

Measure production-faithful prefill scaling and NPU contention on one physical
Atlas 310P after the W1 layout-T16/crop-T8 run completes. Run two configurations
sequentially on the same NPU and warmed graph caches:

1. W2, layout CPU T8 per worker, crop preprocessing T8 per worker, clean only;
2. W4, layout CPU T4 per worker, crop preprocessing T8 per worker, trace then
   clean.

Do not run decode or evaluation. Do not change the pipeline, page lookahead,
layout batch size, bucket set, model formats, or graph sources.

Use the commit containing this brief or later. It must contain `4f98e7a`.

## Fixed production contract

- committed `unirec_representative_128_v1` pages;
- layout B1 and one-page vision lookahead;
- K10 `310p_k10_l1` vision buckets;
- optimized compiled vision: `constant_grouped_all` plus
  `torchair_internal` weights;
- optimized compiled layout without native MSDA;
- threshold 0.5, cross-KV 1320, self-KV/max length 2048;
- compact uint8 HWC recognition inputs;
- prefill only;
- all workers share the process's complete CPU affinity mask;
- no per-worker CPU partitioning in this experiment.

## 910B2 controls

All controls used physical NPU 7 and shared CPU affinity 0-63. Counts were
exact: 128 pages, 2,489 crops, 180,532 real source tokens.

| Workers | Layout CPU / worker | Crop threads / worker | Clean wall | Pages/s |
|---:|---:|---:|---:|---:|
| 1 | 16 | 8 | 29.104 s | 4.398 |
| 2 | 8 | 4 | 19.508 s | 6.561 |
| 2 | 8 | 8 | 15.892 s | 8.054 |
| 4 | 4 | 2 | 9.671 s | 13.236 |
| 4 | 4 | 8 | 8.982 s | 14.250 |
| 4 repeat | 4 | 8 | 9.281 s | 13.791 |

The two W4 layout-T4/crop-T8 clean results average 9.132 s and 14.02 pages/s.
The traced W4 lane was 10.633 s and 12.038 pages/s.

W4 trace service sums versus W1 showed contention:

- layout model forward: 1.641 to 2.263 s;
- vision graph: 9.885 to 12.067 s;
- text-prefill device: 1.108 to 1.474 s.

W4 CPU evidence: 32 native crop threads, maximum 24 concurrent tasks, 52 CPUs
observed, and 3.640 average CPU cores used during crop-active windows.

## Work-server rules

- Pull only. Do not edit tracked files, create a branch, commit, or push.
- Use one free physical 310P from 0 through 3 for both runs.
- Do not run `npu-setup`; it is not installed on this server.
- Preserve the validated `python_nosym` executable path. Do not apply
  `readlink -f` to it.
- Reuse the exact recognition and layout cache roots from the completed W1 run.
- Do not delete, rename, or rebuild caches. Each K10 cache directory must retain
  exactly one `compiled_module` and one OM.
- Preserve every earlier W1 artifact.
- Run W2 and W4 sequentially, never concurrently.

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
export COMPILE_CACHE="${COMPILE_CACHE:?reuse the completed W1 recognition cache}"
export LAYOUT_CACHE_ROOT="${LAYOUT_CACHE_ROOT:?reuse the completed W1 layout cache}"
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
  'import os; a=sorted(os.sched_getaffinity(0)); print("CPU_AFFINITY", len(a), a); assert len(a) >= 16'
```

Do not use `nproc`; it can report `1` because the launcher sets
`OMP_NUM_THREADS=1`. `os.sched_getaffinity(0)` is authoritative.

## Run 1: W2 layout-T8 / crop-T8 clean

```bash
export UNIREC_K10_CHIP_LABEL=310P
export UNIREC_K10_ALLOWED_DEVICES=0,1,2,3
export UNIREC_K10_WORKERS=2
export UNIREC_K10_LAYOUT_CPU_THREADS=8
export UNIREC_K10_THREADS=8
export UNIREC_K10_RUN_MODE=clean_only
bash 12_unirec_0_1b_inference/run_910b_representative128_k10_l1_background.sh
```

Immediately return the absolute run root/log/PID/tail command. Wait for exit
zero before starting W4.

## Run 2: W4 layout-T4 / crop-T8 trace and clean

```bash
export UNIREC_K10_WORKERS=4
export UNIREC_K10_LAYOUT_CPU_THREADS=4
export UNIREC_K10_THREADS=8
export UNIREC_K10_RUN_MODE=both
bash 12_unirec_0_1b_inference/run_910b_representative128_k10_l1_background.sh
```

Immediately return the second absolute run root/log/PID/tail command.

## Required report

For W2, paste complete `CLEAN_ONLY_RESULT`, `CLEAN_ONLY_STAGES`,
`FALLBACK_WARMUP`, and `CACHE` lines.

For W4, paste complete `RESULT`, `STAGES`, `LAYOUT`, `CPU_EXECUTION`,
`FALLBACK_WARMUP`, `BUCKET_CALLS`, and `CACHE` lines.

Also report for both runs:

- commit, physical NPU, CANN, torch, torch-npu, and full CPU affinity;
- setup, warmup, measured prefill, shutdown, and process-wall times;
- exact page/crop/rejection/token counts;
- per-worker page counts and busy times;
- peak HBM and final NPU state;
- confirmation that no new compiled module or OM appeared;
- absolute artifact and log paths.

The W4 trace must prove:

- four worker setup records;
- `layout_cpu_threads=4` in every worker;
- `configured_threads=8` and 32 native crop threads total;
- actual maximum concurrent tasks and observed CPUs;
- exact crop/token parity with W1 and W2.

Compute and report W2 and W4 clean speedups against the completed 310P W1
layout-T16/crop-T8 result. State whether W4 exceeds 10 pages/s on this
representative workload. Stop after this report.
