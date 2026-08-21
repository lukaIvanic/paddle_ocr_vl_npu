# 310P UniRec recognition-crop preprocessing replication

## Goal

Replicate the controlled 910B2 recognition-crop preprocessing benchmark on one
Atlas 310P. Use the same 32 difficult OmniDocBench pages, 1,564 crops, crop
geometry, padding plan, thread counts, warmups, and measured rounds.

This task measures three lanes:

1. original serial FP32 crop preprocessing;
2. compact uint8 HWC preprocessing with one CPU thread and NPU normalization;
3. compact uint8 HWC preprocessing with 16 CPU threads and NPU normalization.

It does not load UniRec weights, run layout inference, compile graphs, run the
vision encoder, or decode text. It should finish in a few minutes. If one phase
takes longer than two minutes, inspect the live log and running process before
waiting further.

The NPU benchmark explicitly calls
`torch_npu.npu.set_compile_mode(jit_compile=False)` before creating any NPU
tensor. Expected graph and operator compilation count is zero. If GE, TBE, ATC,
or TorchAir compilation appears, stop the owned run and report the first
compiler line. Do not wait for compilation to finish.

## Exact 910B2 reference

The matching run used physical Ascend 910B2 NPU 7 and source commit `f13d14d`.
It reconstructed 1,564 crops and retained 403,845,120 real uint8 values.

| Lane | CPU | NPU input | Combined | Crops/s |
|---|---:|---:|---:|---:|
| Original | 11.443914 s | 0.334572 s | 11.778485 s | 132.79 |
| Compact, one thread | 5.749040 s | 0.102798 s | 5.851837 s | 267.27 |
| Compact, 16 threads | 0.972264 s | 0.102798 s | 1.075062 s | 1,454.80 |

All compact outputs passed exact parity. The NPU lane covered 198 input calls:
147 fixed-bucket calls and 51 eager-fallback shapes. No graph compilation ran.

## Rules

- Pull only. Do not edit tracked files, create a branch, commit, or push.
- Use one free physical 310P device from 0 through 3.
- This server has no `npu-setup`. Do not call it.
- Use the validated `python_nosym` executable. Do not apply `readlink -f` to
  the executable because that resolves it outside the validated environment.
- Use at least 16 CPUs in the process affinity. Do not use `nproc` as the
  affinity check. The runner checks `os.sched_getaffinity(0)`.
- Do not run any other benchmark after this task.
- Do not alter round counts or use a different page set.
- Reject any result whose NPU JSON does not contain
  `"npu_jit_compile": false`.

## Launch

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

export PYTHON_BIN=/absolute/path/to/validated/venv/bin/python_nosym
export OPENOCR_ROOT=/absolute/path/to/matching/OpenOCR
export IMAGES_DIR=/absolute/path/to/OmniDocBench-v1.6/images
export ASCEND_RT_VISIBLE_DEVICES=0

"$PYTHON_BIN" -c \
  'import os; a=sorted(os.sched_getaffinity(0)); print("CPU_AFFINITY", len(a), a); assert len(a) >= 16'

bash 12_unirec_0_1b_inference/run_310p_recognition_preprocess_replication_background.sh
```

Choose an actually free device. The runner returns immediately. Send Luka the
printed absolute `RUN_LOG` and `TAIL_COMMAND` before doing anything else.

## Monitor

```bash
tail -f "$RUN_LOG"
```

The log prints start and end timestamps for `sequential`, `threaded`, and `npu`.
Expected work is about three serial original rounds, three compact serial
rounds, five compact 16-thread rounds, and three NPU rounds. There must be no
TorchAir, GE, ATC, or TBE compilation process. If the log is unchanged for more
than two minutes, report the current phase, elapsed time, owned PID state, and
last 30 log lines. Do not wait silently.

## Completion report

Wait for `exit_code.txt`, then return:

```bash
cat "$RUN_ROOT/exit_code.txt"
cat "$RUN_ROOT/process_wall_s.txt"
cat "$RUN_ROOT/final_report.txt"
```

Also provide:

- commit and physical NPU;
- CPU affinity count;
- Python, NumPy, Pillow, OpenCV, torch, and torch-npu versions from
  `preflight.log`;
- the three per-round arrays from `sequential.log`, `threaded.log`, and
  `npu.log`;
- absolute `RUN_ROOT`, `run.log`, and `final_report.json` paths;
- confirmation that no compilation occurred.

Success requires exactly 32 pages and 1,564 crops, exact compact model-input
parity, exact 16-thread bucket output, exact NPU normalization, and these lines:

```text
UNIREC_310P_RECOGNITION_PREPROCESS_PARITY: PASS
UNIREC_310P_RECOGNITION_PREPROCESS_REPLICATION: PASS
UNIREC_310P_RECOGNITION_PREPROCESS_WORKER_END status=0
```

Timing differences never fail the task. Report them and the automatic ratio
against the committed 910B2 reference.
