# 310P UniRec lane-A production-cadence replay

## Goal

Measure lane A with the exact production decoder cadence:

- batch 128;
- cross-KV 256;
- self-KV and generation cap 256;
- real retained crops whose source length is at most 256;
- stable production input buffers, argmax/D2H, EOS, refill, and completion;
- the already compiled `increfa_all` GE graph.

Do only this measurement. Do not profile, rerun prefill, run lane B, or launch
the dual-lane full pipeline.

The matching 910B2 control used 936 real crops and measured:

- 417 decode iterations;
- 57,416.68 raw token slots/s;
- 37,236.51 effective tokens/s;
- 64.8531% slot efficiency;
- 2.2293 ms mean decode step;
- 0.0674 ms input build, 0.3124 ms graph submission, and 1.9157 ms graph +
  argmax + token-D2H wait.

## Constraints

- Pull only. Do not edit tracked files, create a branch, commit, or push.
- Use the same validated `python_nosym`, model, canonical first-128 artifact,
  cache parent, and physical NPU as the passed lane-B production-cadence run.
- Never apply `readlink -f` to `python_nosym`.
- Use one physical 310P device from 0 through 3.
- The exact A graph must already exist: one `compiled_module` and one OM.
- There must be zero compilation and no cache change.
- Print progress every 100 completed crops. Inspect any 30-second silence.

## Prepare

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

export PYTHON_BIN=/absolute/path/to/validated/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export ARTIFACT_DIR=/same/canonical/first128/artifact/used/by/the/passed/B/replay
export DECODE_CACHE_PARENT=/same/cache/parent/used/by/the/passed/dual-decode/lab
export ASCEND_RT_VISIBLE_DEVICES=0  # example only; reuse/select a free 0-3

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -s "$ARTIFACT_DIR/summary.json"
test -s "$ARTIFACT_DIR/crops.jsonl"
test -s "$ARTIFACT_DIR/cross_kv.bin"
A_CACHE="$DECODE_CACHE_PARENT/decode_selfkv256_cross256_increfa_all_b128"
test "$(find "$A_CACHE" -type f -name compiled_module | wc -l)" -eq 1
test "$(find "$A_CACHE" -type f -name '*.om' | wc -l)" -eq 1

export REQUEST_IDS_FILE="$(mktemp /tmp/unirec_lane_a_request_ids.XXXXXX)"
"$PYTHON_BIN" - "$ARTIFACT_DIR/crops.jsonl" "$REQUEST_IDS_FILE" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
selected = [
    row for row in rows if int(row["cross_kv"]["source_length"]) <= 256
]
assert len(rows) == 957, len(rows)
assert len(selected) >= 128, len(selected)
with open(sys.argv[2], "w") as handle:
    for row in selected:
        handle.write(str(row["request_id"]) + "\n")
print(
    "UNIREC_310P_LANE_A_SELECTION: PASS "
    f"selected={len(selected)} min_cross="
    f"{min(row['cross_kv']['source_length'] for row in selected)} "
    f"max_cross={max(row['cross_kv']['source_length'] for row in selected)}"
)
PY
```

Use the exact cache parent from the completed dual-decode lab. Do not point the
runner at a fresh cache directory.

## Launch

```bash
export COMPILE_CACHE="$DECODE_CACHE_PARENT"
export UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE="$DECODE_CACHE_PARENT"
export BATCH_SIZE=128
export SELF_CACHE_LENGTH=256
export CROSS_CACHE_LENGTH=256
export MAX_LENGTH=256
export OFFSET_CROPS=0
export LIMIT_CROPS=0
export STEP_TRACE=1
export PROGRESS_EVERY=100
export WARMUP_PASSES=2
export ADMISSION_PREFETCH_DEPTH=0
export REFERENCE_RUN_SUMMARY=

SHORT="$(git rev-parse --short HEAD)"
STAMP="$(date +%Y%m%dT%H%M%S)"
export RUN_ROOT="$WORK_SERVER_REPO/tmp/12_unirec_0_1b_inference/310p_lane_a_production_cadence_${SHORT}_${STAMP}"
export RUN_LOG="$RUN_ROOT/run.log"

find "$A_CACHE" -type f \( -name compiled_module -o -name '*.om' \) \
  -printf '%p %s %T@\n' | sort > /tmp/unirec_310p_lane_a_om_before.txt

bash 12_unirec_0_1b_inference/run_production_decode_replay_background.sh
```

Immediately give Luka the printed absolute `RUN_LOG`. Monitor every 10--15
seconds:

```bash
while [[ ! -s "$RUN_ROOT/exit_code.txt" ]]; do
  date -Ins
  ps -p "$(cat "$RUN_ROOT/pid.txt")" -o pid,etime,stat,%cpu,%mem --no-headers || true
  tail -n 12 "$RUN_LOG"
  ps -eo comm= | grep -E '^(ccec_compiler|op_compiler|atc)$' || true
  sleep 10
done
```

Expected warm-cache process wall is approximately one to three minutes. A
compiler process, compile/recompile text, or missing progress invalidates the
timing. Report it; do not delete or repair the cache.

## Cache gate and concise report

```bash
find "$A_CACHE" -type f \( -name compiled_module -o -name '*.om' \) \
  -printf '%p %s %T@\n' | sort > "$RUN_ROOT/om_after.txt"
diff -u /tmp/unirec_310p_lane_a_om_before.txt "$RUN_ROOT/om_after.txt" \
  > "$RUN_ROOT/om.diff"
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
assert x["config"]["self_cache_length"] == 256
assert x["config"]["cross_cache_length"] == 256
assert x["config"]["max_length"] == 256
assert x["config"]["self_attention_backend"] == "increfa_all"
assert x["config"]["qkv_fused"] is False
assert x["config"]["weights_nz"] is False
d = x["decode"]
t = x["step_trace"]["overall"]

def ms(field):
    value = t[field]
    return {
        key: 1000.0 * value[key]
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
    "timing_ms": {
        field: ms(field)
        for field in (
            "input_build_s",
            "graph_submit_s",
            "token_select_d2h_wait_s",
            "decode_step_s",
            "scheduler_s",
            "iteration_wall_s",
        )
    },
    "ratio_to_910b_a_raw": (
        d["raw_decode_tokens_per_s"] / 57416.678719622425
    ),
    "approx_ratio_to_310p_b_raw_11400": (
        d["raw_decode_tokens_per_s"] / 11400.0
    ),
}
print("UNIREC_310P_LANE_A_PRODUCTION_CADENCE: PASS")
print("UNIREC_310P_LANE_A_RESULT " + json.dumps(report, sort_keys=True))
PY
```

Paste back:

1. commit, physical NPU, `RUN_ROOT`, `RUN_LOG`, and process wall time;
2. the complete `UNIREC_310P_LANE_A_RESULT` line;
3. both warmup-pass times and compile/recompile grep result;
4. `om.diff` status and final A-cache counts;
5. the five slowest decode steps.

Stop after this report.
