# 310P UniRec clean CPU/worker overlap test: W4 and W8

## Objective

Measure how far clean representative-128 prefill can scale through more CPU
capacity and more independent page workers before changing the H2D or model
implementation.

Run two clean-only lanes on the same free physical 310P and warmed caches:

1. W4, layout CPU T8, crop T8: isolate additional CPU capacity versus the
   completed W4/layout-T4/crop-T8 result near 6.87 pages/s.
2. W8, layout CPU T4, crop T8: test additional page/NPU overlap.

If W8 cannot fit in HBM, preserve the complete failure evidence and run W6,
layout CPU T4, crop T8 instead. Do not run trace, decode, or evaluation yet.

Use the commit containing this brief or later.

## Verified 910B2 controls

All controls use the exact representative-128 K10/B1/lookahead-1 production
contract and 2,489 crops with zero rejections.

| Workers | Layout CPU | Crop threads | Clean wall | Pages/s |
|---:|---:|---:|---:|---:|
| 4 | 4 | 8 | 8.982 / 9.281 s | 14.25 / 13.79 |
| 4 | 8 | 8 | 8.512 s | 15.04 |
| 8 | 4 | 8 | **5.907 s** | **21.67** |

The 910B W8 result passed on physical NPU 7 at commit `7cd0f82`:

- setup 70.296 s, warmup 0.421 s, measured prefill 5.906745 s;
- shutdown 22.033 s;
- worker page counts: 17, 17, 12, 16, 15, 19, 19, 13;
- worker busy times: 5.64--5.88 s;
- layout service sum 10.377 s;
- recognition-input service sum 4.022 s;
- recognition-prefill service sum 19.705 s;
- shared-pack service sum 4.450 s;
- direct-RGB-decode service sum 5.538 s;
- maximum per-worker NPU allocation snapshot 0.860 GB. This is not the
  aggregate device peak across eight processes.

Artifact:

`/workspace/repos/paddle_ocr_vl_npu/tmp/12_unirec_0_1b_inference/910b_rep128_k10_l1_7cd0f82_20260817T150836/clean/output/run_summary.json`

This validates the configuration, not its 310P scaling.

## Work-server rules

- Pull only. Do not edit tracked files, create a branch, commit, or push.
- Use one free physical 310P in `0..3` for all lanes.
- Do not run `npu-setup`; it is not installed.
- Preserve the validated `python_nosym` path. Do not apply `readlink -f` to it.
- Do not use `nproc`; use `os.sched_getaffinity(0)`.
- Reuse the exact model paths and warmed recognition/layout cache roots from
  the successful W4 run.
- Do not delete or rebuild caches. Multiple workers may read the same caches.
- Run lanes sequentially, never concurrently.
- Use the same expanded CPU affinity mechanism as the successful W4 run. The
  launched process must see at least 64 CPUs; prefer all 96 available CPUs.

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
export COMPILE_CACHE="${COMPILE_CACHE:?completed W4 recognition cache}"
export LAYOUT_CACHE_ROOT="${LAYOUT_CACHE_ROOT:?completed W4 layout cache}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:?free NPU 0..3}"
export UNIREC_CPUSET="${UNIREC_CPUSET:-0-95}"

case "$ASCEND_RT_VISIBLE_DEVICES" in
  0|1|2|3) ;;
  *) echo "Expected one physical 310P in 0..3" >&2; exit 1 ;;
esac

test -f "$MODEL/model.pth"
test -d "$LAYOUT_MODEL"
test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
test -d "$IMAGES_DIR"
test -d "$COMPILE_CACHE"
test -d "$LAYOUT_CACHE_ROOT"

taskset -c "$UNIREC_CPUSET" "$PYTHON_BIN" - <<'PY'
import os
affinity = sorted(os.sched_getaffinity(0))
print("UNIREC_CPU_AFFINITY", len(affinity), affinity)
assert len(affinity) >= 64
PY
```

If `0-95` is not the server's online CPU numbering, set `UNIREC_CPUSET` to the
exact known-good expanded CPU list. Do not silently proceed with one CPU.

## Lane A: W4 layout-T8 / crop-T8

```bash
export UNIREC_K10_CHIP_LABEL=310P
export UNIREC_K10_ALLOWED_DEVICES=0,1,2,3
export UNIREC_K10_WORKERS=4
export UNIREC_K10_LAYOUT_CPU_THREADS=8
export UNIREC_K10_THREADS=8
export UNIREC_K10_RUN_MODE=clean_only
taskset -c "$UNIREC_CPUSET" \
  bash 12_unirec_0_1b_inference/run_910b_representative128_k10_l1_background.sh
```

Return the absolute run root, log, PID, physical NPU, and tail command
immediately. Wait for exit zero before Lane B.

## Lane B: W8 layout-T4 / crop-T8

```bash
export UNIREC_K10_WORKERS=8
export UNIREC_K10_LAYOUT_CPU_THREADS=4
export UNIREC_K10_THREADS=8
export UNIREC_K10_RUN_MODE=clean_only
taskset -c "$UNIREC_CPUSET" \
  bash 12_unirec_0_1b_inference/run_910b_representative128_k10_l1_background.sh
```

Monitor aggregate device HBM externally during setup and measured prefill. The
per-worker JSON memory snapshot is not the aggregate across processes.

If W8 fails specifically from HBM or worker death, do not retry it. Preserve
the first causal log and run this fallback:

```bash
export UNIREC_K10_WORKERS=6
export UNIREC_K10_LAYOUT_CPU_THREADS=4
export UNIREC_K10_THREADS=8
export UNIREC_K10_RUN_MODE=clean_only
taskset -c "$UNIREC_CPUSET" \
  bash 12_unirec_0_1b_inference/run_910b_representative128_k10_l1_background.sh
```

## Cache and correctness gates

For every successful lane require:

- status `ok`, 128 pages, 2,489 crops, zero rejections;
- 180,532 real source tokens;
- layout B1, one-page lookahead, K10 bucket preset;
- exact requested worker/layout/crop-thread counts in worker diagnostics;
- every worker completed graph warmup;
- no new OM or `compiled_module` appeared;
- no HBM OOM, worker EOF, or process death;
- final NPU state returned to baseline.

## Required report

For the existing W4-T4/C8 baseline, Lane A, and Lane B/fallback, report:

- clean measured wall and pages/s;
- speedup versus W4-T4/C8 and versus W1;
- setup, warmup, shutdown, and lifecycle times separately;
- aggregate peak HBM from external sampling;
- worker page counts and busy times;
- service sums for layout, recognition input, recognition prefill, cross-KV
  D2H, shared pack, RGB decode, and IPC;
- CPU affinity visible to the parent and every worker;
- cache inventory before/after;
- absolute artifacts and logs.

State whether clean prefill reaches 8.0 pages/s. Stop after the clean report.
Do not trace the winning lane until Luka requests it.
