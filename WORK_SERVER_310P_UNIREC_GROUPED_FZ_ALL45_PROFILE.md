# 310P UniRec all-45 grouped-FZ profile

Pull the commit containing this brief. Profile only the compiled production
`960x64_b16` vision graph with `constant_grouped_all` and native vision-weight
handling. This is a narrow follow-up to the already completed five-bucket
`constant_grouped` A/B. Do not rerun page prefill or decode.

## Restrictions

- Pull only. Do not edit tracked files, branch, commit, or push.
- Use one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Reuse the native compiled `960x64_b16.pt` saved by the last passed five-bucket
  run. Do not rerun the native profile.
- Use a new cache directory for `constant_grouped_all`.
- Run in the background and immediately give Luka the absolute log path.
- Preserve the first causal error. Do not retry with changed settings.

## 910B2 evidence to compare against

Commit `b486ffd`, physical 910B2 NPU 7, CANN 9.0.0:

- all five production buckets were bit-exact to their saved native compiled
  outputs (`max_abs=0`, `mean_abs=0`);
- for `960x64_b16`, all 45 focal weights were rewritten;
- focal-weight TransData fell from 46 calls to zero;
- total TransData count fell from 259 to 215;
- warmed device median was 11.511460 ms.

The all-45 profile remains allowed to contain unrelated activation, attention,
and other-weight TransData operations. The required condition is zero matches
for all focal-weight native-to-FZ and FZ-to-grouped-FZ signatures.

## Resolve and launch

Reuse the interpreter, model, layout model, and layout cache from the previous
passed UniRec vision-profile task. Resolve the native reference directory from
the latest passed grouped-FZ bucket run; it must contain `960x64_b16.pt`.

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main
source npu-setup
case ",${ASCEND_RT_VISIBLE_DEVICES:-}," in
  *,5,*|*,6,*) echo "REJECTED_PHYSICAL_DEVICE_5_OR_6" >&2; exit 1 ;;
esac

export PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
export MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
export LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
export LAYOUT_CACHE="${LAYOUT_CACHE:?reuse the passed layout cache}"
export NATIVE_OUTPUTS="${NATIVE_OUTPUTS:?directory containing the passed native compiled outputs}"
test -f "$NATIVE_OUTPUTS/960x64_b16.pt"

STAMP="$(date +%Y%m%dT%H%M%S)"
RUN_ROOT="$REPO/tmp/12_unirec_0_1b_inference/310p_grouped_fz_all45_profile_$(git rev-parse --short HEAD)_$STAMP"
CACHE="$REPO/.runtime_cache/12_unirec_0_1b_inference/vision_grouped_fz_all45_$(git rev-parse --short HEAD)"
mkdir -p "$RUN_ROOT" "$CACHE"
export REPO RUN_ROOT CACHE

nohup bash -c '
  set -euo pipefail
  "$PYTHON_BIN" 12_unirec_0_1b_inference/profile_prefill_graph_suite.py \
    --model-path "$MODEL" \
    --layout-model "$LAYOUT_MODEL" \
    --layout-cache-dir "$LAYOUT_CACHE" \
    --recognition-cache-dir "$CACHE" \
    --output-dir "$RUN_ROOT/output" \
    --device npu:0 --lane vision --vision-bucket 960x64_b16 \
    --vision-depthwise-rewrite constant_grouped_all \
    --vision-weight-format native \
    --warmup 2 --control-repeats 10 --profile-steps 1 --parser-topn 100 \
    --allow-vision-parity-drift \
    --reference-vision-outputs-dir "$NATIVE_OUTPUTS"
  "$PYTHON_BIN" 12_unirec_0_1b_inference/audit_focal_weight_transdata.py \
    "$RUN_ROOT/output/profile_suite_summary.json" \
    --bucket 960x64_b16 --require-all-45 --require-exact-reference
' >"$RUN_ROOT/run.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" >"$RUN_ROOT/pid.txt"
printf '310P ALL45 PROFILE STARTED - pid=%s; run_log=%s; tail_command=tail -f %s\n' \
  "$PID" "$RUN_ROOT/run.log" "$RUN_ROOT/run.log"
```

## Report and stop

Return only:

1. commit, physical NPU, CANN, torch, and torch_npu;
2. the `UNIREC_PREFILL_PROFILE_LANE` line;
3. the `UNIREC_FOCAL_WEIGHT_TRANSDATA_AUDIT` line;
4. absolute run log, profile summary JSON, parsed profile JSON, and cache path.

The audit must say `rewritten=45`, `focal_weight_transdata=0`, and
`exact_reference=true`. If it fails, return the printed matching rows and the
first causal error. Do not run the full 1,651-page prefill in this task.
