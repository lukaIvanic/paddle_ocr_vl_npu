# 310P UniRec representative-128 K10/L1 prefill

## Objective

Compile and measure the same K10 page-local vision-bucket prototype that is
being tested on 910B2. Run one traced lane and one clean warmed lane on the
fixed representative-128 manifest.

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

The runner compiles every K10 graph once, then runs trace and clean lanes. Cold
compile/cache-load/setup time is recorded but excluded from measured prefill
wall time.

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
bash 12_unirec_0_1b_inference/run_910b_representative128_k10_l1_background.sh
```

The launcher prints an absolute `RUN_ROOT`, `RUN_LOG`, PID, and `tail -f`
command. Report those immediately so Luka can follow the log. Do not wait for
the full run before reporting the path.

The log prints one begin/end record for each compiled graph:

```text
UNIREC_VISION_GRAPH_DIAGNOSTIC ... warmup_graph_call_begin ...
UNIREC_VISION_GRAPH_DIAGNOSTIC ... warmup_graph_call_end ... synchronized_wall_s=...
```

Use these records to distinguish a long cold compile from first-process cache
loading and fast workload replay. Do not use `om_count` alone as the verdict.
On the matched 910B2 run, the cold lane created one OM per graph. Reopening the
cache in the clean process took about 20--23 seconds per graph and materialized
a second OM file. This is the previously observed TorchAir cache-load behavior,
not enough evidence of a full cold recompile. The decisive checks are that the
clean setup is much shorter than cold setup and that measured workload calls
after warmup are fast. Do not stop only because `om_count` changes from one to
two.

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
3. all ten first-lane graph warmup durations and OM counts;
4. clean-lane first-open time, OM-count change, and later workload-call time for
   each graph; label compile versus load only when the logs establish it;
5. the complete `UNIREC_310P_K10_L1_RESULT`, `BUCKET_CALLS`, and `LAYOUT`
   lines;
6. trace and clean retained crop counts and source-token totals;
7. trace stage sums for layout, vision bucket graph, vision fallback graph,
   crop preprocessing, text-prefill pack, and text-prefill device;
8. peak HBM from the run summary or observed `npu-smi` sampling;
9. `exit_code.txt`, total process wall time, and final NPU state;
10. any warning, crash, fallback, cache anomaly, or mismatch.

## Reference expectations

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
