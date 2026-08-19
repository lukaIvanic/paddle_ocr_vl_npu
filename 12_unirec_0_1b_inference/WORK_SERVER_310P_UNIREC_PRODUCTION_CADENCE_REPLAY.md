# 310P UniRec production-cadence decoder replay

## Goal

Reproduce the real B128 decoder throughput on 310P with the exact serving
contract. This is not the synthetic text-decode lab. It replays real retained
cross-KV through `ContinuousUniRecDecoder`, including:

- one preallocated production arena;
- stable long-lived device input buffers;
- tiny next-token/cache-position H2D copies;
- the unchanged compiled B128 C1320/S2048 GE graph;
- argmax, sampled-token D2H, EOS, slot refill, and output completion.

The matching 910B2 control used 957 real crops and measured:

- 2,213 graph iterations;
- 6.0735 ms mean decode step;
- 21,075.06 raw token slots/s;
- 3,153.55 effective tokens/s;
- 14.9634% slot efficiency;
- 0.0628 ms input build, 0.2998 ms asynchronous graph submission, and
  5.7724 ms graph + argmax + token-D2H wait per step.

The 910B2 step stayed near 6.02--6.09 ms from cache position 129 through 2048.
Its production replay was only 4.1% below the isolated B-lane clean result, so
this is the correct boundary for deciding whether the reported 310P 3.4k lab
number or the historical approximately 8.5k full-production number is real.

## Constraints

- Pull only. Do not edit tracked files, create a branch, commit, or push.
- Use one free physical 310P device from 0 through 3. There is no `npu-setup`.
- Use the validated venv `python_nosym`. Never apply `readlink -f` to it.
- Reuse the exact B graph cache reported by the completed accuracy-safe full
  run. There must be zero compilation and no OM changes.
- Reuse the existing canonical-native persistent first-128 artifact from the
  passed prefill/decode-factorization work. Do not rerun prefill.
- Run one traced production replay. The step callback executes after the
  measured decode step, so `decode_s` remains the serving timing boundary.
- If the process is quiet for 30 seconds, inspect it immediately. Do not wait
  blindly.

## Resolve inputs

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

export PYTHON_BIN=/absolute/path/to/validated/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export ARTIFACT_DIR=/absolute/path/to/passed/prefill_artifact_first128_canonical_native
export FULL_RUN_ROOT=/absolute/path/to/completed/accuracy-safe/full1651/run
export ASCEND_RT_VISIBLE_DEVICES=0  # example only; select a free device 0-3

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -s "$ARTIFACT_DIR/summary.json"
test -s "$ARTIFACT_DIR/crops.jsonl"
test -s "$ARTIFACT_DIR/cross_kv.bin"
test -s "$FULL_RUN_ROOT/output/run_summary.json"
test "$(wc -l < "$ARTIFACT_DIR/crops.jsonl")" -eq 957

readarray -t CACHE_PATHS < <(
  "$PYTHON_BIN" - "$FULL_RUN_ROOT/output/run_summary.json" <<'PY'
import json
import sys
from pathlib import Path

d = json.load(open(sys.argv[1]))
assert d["status"] == "ok"
assert d["page_count"] == 1651
assert d["decode_batch_size"] == 128
assert d["self_cache_length"] == 2048
assert d["cross_cache_length"] == 1320
if d.get("decode_lane_mode") == "dual":
    shape = Path(d["decode"]["lanes"]["b"]["compile"]["torchair_cache_dir"])
else:
    shape = Path(d["decode"]["compile"]["torchair_cache_dir"])
assert shape.name == "decode_selfkv2048_cross1320_increfa_all_b128"
print(shape.parent)
print(shape)
PY
)
test "${#CACHE_PATHS[@]}" -eq 2
export UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="${CACHE_PATHS[0]}"
export COMPILE_CACHE="${CACHE_PATHS[0]}"
SHAPE_CACHE="${CACHE_PATHS[1]}"
test "$(find "$SHAPE_CACHE" -type f -name compiled_module | wc -l)" -eq 1
test "$(find "$SHAPE_CACHE" -type f -name '*.om' | wc -l)" -eq 1

"$PYTHON_BIN" - "$ARTIFACT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.load(open(root / "summary.json"))
rows = [json.loads(line) for line in open(root / "crops.jsonl") if line.strip()]
assert summary["format"] == "unirec_cross_kv_v1"
assert summary["status"] == "ok"
assert len(rows) == 957
assert len({row["page_index"] for row in rows}) == 128
assert max(row["cross_kv"]["source_length"] for row in rows) <= 1320
print("UNIREC_310P_PRODUCTION_CADENCE_ARTIFACT: PASS")
PY
```

The selected full run must be the completed accuracy-safe result whose summary
reported the historical production decoder throughput. Do not substitute the
new synthetic dual-lane lab output.

## Launch

```bash
export BATCH_SIZE=128
export SELF_CACHE_LENGTH=2048
export CROSS_CACHE_LENGTH=1320
export MAX_LENGTH=2048
export STEP_TRACE=1
export PROGRESS_EVERY=100
export WARMUP_PASSES=2
export ADMISSION_PREFETCH_DEPTH=0
export REFERENCE_RUN_SUMMARY="$FULL_RUN_ROOT/output/run_summary.json"

SHORT="$(git rev-parse --short HEAD)"
STAMP="$(date +%Y%m%dT%H%M%S)"
export RUN_ROOT="$WORK_SERVER_REPO/tmp/12_unirec_0_1b_inference/310p_production_cadence_replay_${SHORT}_${STAMP}"
export RUN_LOG="$RUN_ROOT/run.log"

find "$SHAPE_CACHE" -type f \( -name compiled_module -o -name '*.om' \) \
  -printf '%p %s %T@\n' | sort > /tmp/unirec_310p_production_cadence_om_before.txt

bash 12_unirec_0_1b_inference/run_production_decode_replay_background.sh
```

Immediately give Luka the printed absolute `RUN_LOG` path. Follow it directly:

```bash
tail -f "$RUN_LOG"
```

The existing 957-crop 310P replay historically took several minutes because
of long outputs. Check every 10--15 seconds:

```bash
while [[ ! -s "$RUN_ROOT/exit_code.txt" ]]; do
  date -Ins
  ps -p "$(cat "$RUN_ROOT/pid.txt")" -o pid,etime,stat,%cpu,%mem --no-headers || true
  tail -n 12 "$RUN_LOG"
  ps -eo comm= | grep -E '^(ccec_compiler|op_compiler|atc)$' || true
  sleep 10
done
```

Progress must appear every 100 completed crops. If a compiler process or
compile/recompile message appears, stop interpreting throughput and report it.
Do not delete or repair any cache.

## Cache gate and report

```bash
find "$SHAPE_CACHE" -type f \( -name compiled_module -o -name '*.om' \) \
  -printf '%p %s %T@\n' | sort > "$RUN_ROOT/om_after.txt"
diff -u /tmp/unirec_310p_production_cadence_om_before.txt \
  "$RUN_ROOT/om_after.txt" > "$RUN_ROOT/om.diff"
test "$(cat "$RUN_ROOT/exit_code.txt")" -eq 0
! grep -Eqi 'recompil|skip cache|compile graph|start.*compil|Traceback|ERROR' \
  "$RUN_ROOT/run.log"

"$PYTHON_BIN" - "$RUN_ROOT/result.json" <<'PY' \
  | tee "$RUN_ROOT/final_report.txt"
import json
import sys

x = json.load(open(sys.argv[1]))
assert x["kind"] == "unirec_production_decode_replay"
assert x["status"] == "ok"
assert x["config"]["batch_size"] == 128
assert x["config"]["self_cache_length"] == 2048
assert x["config"]["cross_cache_length"] == 1320
assert x["config"]["self_attention_backend"] == "increfa_all"
assert x["config"]["qkv_fused"] is False
assert x["config"]["weights_nz"] is False
d = x["decode"]
t = x["step_trace"]

def ms(summary):
    return {
        key: 1000.0 * summary[key]
        for key in ("mean", "p50", "p90", "p95", "p99", "max")
    }

report = {
    "status": "pass",
    "physical_devices": x["physical_devices"],
    "selected_crops": x["workload"]["selected_crops"],
    "decode_iterations": d["decode_iterations"],
    "raw_token_slots": d["raw_decode_token_slots"],
    "effective_tokens": d["effective_decode_tokens"],
    "slot_efficiency": x["slot_efficiency"],
    "decode_s": d["decode_s"],
    "raw_tok_s": d["raw_decode_tokens_per_s"],
    "effective_tok_s": d["effective_decode_tokens_per_s"],
    "reference_full_run": x["reference_throughput"],
    "timing_ms": {
        field: ms(t["overall"][field])
        for field in (
            "input_build_s",
            "graph_submit_s",
            "token_select_d2h_wait_s",
            "decode_step_s",
            "scheduler_s",
            "iteration_wall_s",
        )
    },
    "by_cache_position_mean_step_ms": {
        key: 1000.0 * value["decode_step_s"]["mean"]
        for key, value in t["by_cache_position_max"].items()
    },
    "by_cross_length_mean_step_ms": {
        key: 1000.0 * value["decode_step_s"]["mean"]
        for key, value in t["by_cross_length_max"].items()
    },
    "ratio_to_910b_raw": d["raw_decode_tokens_per_s"] / 21075.057351810738,
}
print("UNIREC_310P_PRODUCTION_CADENCE_REPLAY: PASS")
print("UNIREC_310P_PRODUCTION_CADENCE_RESULT " + json.dumps(report, sort_keys=True))
PY
```

Paste back:

1. commit, physical NPU, `RUN_ROOT`, `RUN_LOG`, and process wall time;
2. the complete `UNIREC_310P_PRODUCTION_CADENCE_RESULT` line;
3. both warmup-pass times and whether any compilation text appeared;
4. `om.diff` status and final OM/compiled-module counts;
5. the ten slowest decode steps from `result.json`.

Stop after this report. Do not launch the dual-lane production pipeline yet.
