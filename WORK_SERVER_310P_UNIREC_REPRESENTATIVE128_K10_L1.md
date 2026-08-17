# 310P UniRec representative-128 K10/L1 prefill

## Objective

Compile and measure the same K10 page-local vision-bucket prototype that is
being tested on 910B2. Run one traced lane and one clean warmed lane on the
fixed representative-128 manifest.

The cache-persistence production fix is commit `5710ab3` or later. Do not run
this handoff from an earlier commit.

This is one experiment, not an A/B matrix. Use:

- one process worker and one recognition-preprocess thread (`W1/T1`);
- layout batch size 1;
- one-page vision lookahead;
- vision preset `310p_k10_l1`;
- optimized compiled vision (`constant_grouped_all` plus
  `torchair_internal` weights);
- the current optimized layout path, without native MSDA;
- cross-KV 1320 and self-KV 2048;
- prefill only.

The runner compiles every corrected cache-stable K10 graph once, then runs trace
and clean lanes. Cold compile/cache-load/setup time is recorded but excluded
from measured prefill wall time.

This revision replaces the old dynamically generated per-bucket `forward`
methods. Those methods left only GE OMs and never persisted TorchAir's upper
`compiled_module` cache. Do not reuse an old dynamic K10 directory as evidence.
The corrected source hash creates new cache directories without deleting the
old ones.

## Work-server rules

- Pull only. Do not edit tracked files, create a branch, commit, or push.
- The 310P server has four physical NPUs: 0, 1, 2, and 3. Pick one free device.
- Do not run `npu-setup`; it is a 910B-server helper and is absent here.
- Use the already validated CANN/torch-npu shell environment.
- Use the real `python_nosym` executable from the validated venv. Do not apply
  `readlink -f` to it. The runner preserves the executable path.
- Do not run another benchmark on the selected NPU while this compiles.
- Leave correctness thresholds as warnings only where the existing production
  path already does so. Do not change code because of small tensor drift.

## Pull and preflight

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git status --short
git pull --ff-only
git rev-parse HEAD

test -x "${PYTHON_BIN:?set the existing validated python_nosym executable}"
test "$(basename "$PYTHON_BIN")" = python_nosym

export MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
export LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
export OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/../OpenOCR}"
export IMAGES_DIR="${IMAGES_DIR:?set the OmniDocBench v1.6 images directory}"
export COMPILE_CACHE="${COMPILE_CACHE:-$REPO/.runtime_cache/12_unirec_0_1b_inference/opendoc_batched_decode_a372dbf}"
export LAYOUT_CACHE_ROOT="${LAYOUT_CACHE_ROOT:?set the existing optimized-layout compile-cache directory}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:?select one free 310P device, 0 through 3}"

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
"$PYTHON_BIN" -c 'import torch, torch_npu, kornia_rs; print(torch.__version__, torch_npu.__version__)'
npu-smi info
```

If `OPENOCR_ROOT`, `IMAGES_DIR`, or `LAYOUT_CACHE_ROOT` differs on this server,
reuse the exact paths from the last successful representative-128 prefill run.
Do not search for ONNX exports; this experiment does not use one.

## Launch once in the background

```bash
cd "$REPO"
export UNIREC_K10_CHIP_LABEL=310P
export UNIREC_K10_ALLOWED_DEVICES=0,1,2,3
export UNIREC_K10_RUN_MODE=both
bash 12_unirec_0_1b_inference/run_910b_representative128_k10_l1_background.sh
```

The runner enables verbose graph diagnostics only for the trace lane. It
explicitly disables them for the clean lane. Confirm that the clean lane does
not print per-workload `bucket_call_begin` records; otherwise its throughput is
not clean.

The launcher prints an absolute `RUN_ROOT`, `RUN_LOG`, PID, and `tail -f`
command. Report those immediately so Luka can follow the log. Do not wait for
the full run before reporting the path.

The log prints one begin/end record for each compiled graph:

```text
UNIREC_VISION_GRAPH_DIAGNOSTIC ... warmup_graph_call_begin ...
UNIREC_VISION_GRAPH_DIAGNOSTIC ... warmup_graph_call_end ... synchronized_wall_s=...
```

Use these records to distinguish cold compilation from cache loading. The
corrected requirement is strict:

- after the trace lane, every one of the ten cache directories contains exactly
  one `compiled_module` and one GE OM;
- the clean fresh process keeps the same OM inventory;
- clean first-call times are cache-load times, not cold compile times;
- no second OM is created merely because the process restarted.

The 910B2 proof reduced a one-bucket fresh-process first call from 13--23 seconds
to approximately 0.60 seconds while preserving graph output and steady kernel
latency. If `compiled_module` is absent, stop and report the cache directory and
log. Do not continue to call the result warmed.

## Completion and report

Wait for `exit_code.txt`. Success requires `0`.

```bash
RUN_ROOT="<absolute path printed by launcher>"
cat "$RUN_ROOT/exit_code.txt"
cat "$RUN_ROOT/report.log"
grep -E 'warmup_graph_call_(begin|end)|UNIREC_310P_K10_L1_' "$RUN_ROOT/run.log"
npu-smi info
```

Paste back:

1. commit, physical NPU, CANN, torch, and torch-npu versions;
2. absolute `RUN_ROOT` and `RUN_LOG`;
3. all ten first-lane graph warmup durations, OM counts, and
   `compiled_module_count` values;
4. clean-lane first-open time and final OM/`compiled_module` inventory for each
   graph; confirm no new OM appeared across the process restart;
5. the complete `UNIREC_310P_K10_L1_RESULT`, `BUCKET_CALLS`, and `LAYOUT`
   lines;
6. trace and clean retained crop counts and source-token totals;
7. trace stage sums for layout, vision bucket graph, vision fallback graph,
   crop preprocessing, text-prefill pack, and text-prefill device;
8. peak HBM from the run summary or observed `npu-smi` sampling;
9. `exit_code.txt`, total process wall time, and final NPU state;
10. any warning, crash, fallback, cache anomaly, or mismatch.

Also run this final inventory command and paste its complete output:

```bash
find "$COMPILE_CACHE" \
  -path '*vision_full_bucket*' \
  \( -name compiled_module -o -name '*.om' \) \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' | sort
```

## Reference expectations

The 910B2 production-slot gate at `5710ab3` passed for two bucket variants in
one process and a fresh-process reload:

```text
448x64_b1 cold first call 33.9447 s; reload 0.6028 s
448x64_b4 cold first call 30.6936 s; reload 0.5671 s
compiled_module files: 2
OM files after reload: 2
new OMs during reload: 0
correctness warnings: 0
```

Both cold bucket slots logged `Saving cache`; both fresh-process calls logged
`Loading cache`. Steady graph latency remained approximately 9--11 ms. This is
the behavior the ten-bucket 310P run must reproduce structurally, although its
absolute load and graph times will differ.

The fixed CPU replay for this exact K10 planner predicts:

- 2,425 compiled crops from the prior representative-128 trace;
- 1,049 bucket calls;
- 68.3055% compiled pixel efficiency;
- 91.65% slot efficiency in the earlier planning replay;
- no change to crop geometry or model semantics; only padding canvas and call
  grouping change.

The exact expected call histogram is:

```text
448x64_b4   300
448x256_b2   82
448x384_b2   43
512x128_b4   58
960x64_b2    61
960x64_b4   150
960x128_b1  144
960x256_b1  120
960x384_b1   37
960x512_b1   54
```

The prior 310P five-bucket representative-128 trace reported approximately
26.34 s of compiled vision-bucket graph time plus 3.41 s of vision fallback
graph time. Compare K10 against those two components separately. Do not compare
total wall time without also reporting layout time because this K10 test uses
layout B1.
