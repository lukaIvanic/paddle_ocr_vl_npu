# 310P UniRec dual-lane decoder lab

## Goal

Measure the two exact B128 decoder graphs needed by the new 3A:1B production
policy:

- lane A: cross-KV 256, self-KV 256;
- lane B: cross-KV 1320, self-KV 2048.

This is a decoder-only run. Do not run page prefill, layout, profiling, or
OmniDocBench evaluation. Stop after the matrix report so the measured 310P
step costs can be reviewed before a production run.

The completed 910B2 full run established the workload counts used by the report:

- A: 11,621 graph steps;
- B: 8,619 graph steps;
- prior single-B baseline: 19,388 graph steps.

The matrix prints graph-only and production-like sampled-token D2H projections.
They are forecasts, not production measurements.

The exact runner passed on 910B2 at commit `f0cad0b` with warm caches:

| Lane | Clean step | Clean raw tok/s | D2H step | D2H raw tok/s | Peak HBM |
|---|---:|---:|---:|---:|---:|
| A, C256/S256 | 1.8556 ms | 68,979.5 | 2.3055 ms | 55,520.2 | 2.62 GiB |
| B, C1320/S2048 | 5.7797 ms | 22,146.3 | 6.1174 ms | 20,924.1 | 15.18 GiB |

The 910B2 D2H projection was 79.52 seconds versus the measured full-production
dual graph time of 81.54 seconds, a 2.5% miss. Compare the 310P result directly
against these rows; do not transfer the 910B timings to 310P.

## Constraints

- Pull only. Do not edit tracked files, create a branch, commit, or push.
- Use one free physical 310P device from 0-3. There is no `npu-setup` here.
- Use the validated venv `python_nosym`. Never run `readlink -f` on it.
- Reuse the exact B graph from the completed accuracy-safe full run.
- Lane A may compile exactly once if its exact cache is absent. Lane B must
  never compile or change.
- The runner emits a heartbeat every ten seconds. If a phase exceeds 90
  seconds, report its last event, OM counts, and compiler-process count before
  continuing. Do not wait blindly.
- Do not delete, rename, repair, or copy any cache after a failure.

## Resolve the production cache parent

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

export PYTHON_BIN=/absolute/path/to/validated/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export FULL_RUN_ROOT=/absolute/path/to/completed/accuracy-safe/full1651/run
export ASCEND_RT_VISIBLE_DEVICES=0  # example; select a free device 0-3

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -s "$FULL_RUN_ROOT/output/run_summary.json"
export DECODE_CACHE_PARENT="$($PYTHON_BIN - "$FULL_RUN_ROOT/output/run_summary.json" <<'PY'
import json
import sys
from pathlib import Path

d = json.load(open(sys.argv[1]))
assert d["status"] == "ok"
assert d["decode_batch_size"] == 128
assert d["self_cache_length"] == 2048
assert d["cross_cache_length"] == 1320
path = Path(d["decode"]["compile"]["torchair_cache_dir"])
assert path.name == "decode_selfkv2048_cross1320_increfa_all_b128"
print(path.parent)
PY
)"
test -d "$DECODE_CACHE_PARENT/decode_selfkv2048_cross1320_increfa_all_b128"
printf 'DECODE_CACHE_PARENT=%s\n' "$DECODE_CACHE_PARENT"
```

If the selected full run used a dual decoder, resolve lane B instead:

```bash
export DECODE_CACHE_PARENT="$($PYTHON_BIN - "$FULL_RUN_ROOT/output/run_summary.json" <<'PY'
import json
import sys
from pathlib import Path

d = json.load(open(sys.argv[1]))
path = Path(d["decode"]["lanes"]["b"]["compile"]["torchair_cache_dir"])
assert path.name == "decode_selfkv2048_cross1320_increfa_all_b128"
print(path.parent)
PY
)"
```

## Launch and monitor

```bash
bash 12_unirec_0_1b_inference/run_310p_dual_lane_decode_lab_background.sh
```

Immediately give Luka the printed absolute `RUN_LOG` path. In a second terminal:

```bash
tail -f "$RUN_LOG"
```

The worker agent must check every 10-15 seconds:

```bash
while [[ ! -s "$RUN_ROOT/exit_code.txt" ]]; do
  date -Ins
  tail -n 12 "$RUN_LOG"
  sleep 10
done
```

Expected warm-cache wall time is about 1-2 minutes. A missing A graph can add
several minutes on 310P, but the heartbeat must identify that compilation
explicitly. There are two model-loading processes by design, one per static
self-KV length.

## Completion gate and report

Require:

- `exit_code.txt` is zero;
- `UNIREC_310P_DUAL_DECODE_LAB: PASS` exists;
- lane B has one `compiled_module` and one OM before and after;
- `b_om.diff` is empty;
- lane A has exactly one `compiled_module` and one OM after the run;
- no traceback, OOM, or NPU timeout;
- report whether A was already compiled or was compiled once in this run.

Paste back:

1. commit, physical NPU, `RUN_ROOT`, `RUN_LOG`, and total wall time;
2. the complete `UNIREC_310P_DUAL_DECODE_RESULT` line;
3. for both lanes: first-call seconds, clean step ms/raw tok/s,
   production-like D2H step ms/raw tok/s, and peak allocated HBM;
4. projected dual graph/D2H time, single-B graph/D2H time, and both speedups;
5. `a_om_before`, `b_om_before`, final exact cache counts, `b_om.diff`, and
   whether any compile/recompile text appeared.

Stop after this report. Do not launch the representative or full production
pipeline yet.
