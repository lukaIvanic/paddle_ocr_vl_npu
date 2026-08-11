# Work-server 310P UniRec prefill W1/T16 smoke

This is the next pull-only task for the agent on Luka's Atlas 310P server.
U2.5 is already complete. Do not repeat it. Run only the experiment in this
brief and stop.

The first Pass A attempt failed in eager PP-DocLayoutV2 reading-order input
construction. It established that Transformers 5.5.4's data-dependent indexed
writes into `input_ids` are not valid on this 310P stack. A follow-up source
audit found that the first local rewrite was incomplete: after replacing those
writes, it still called `create_bidirectional_mask`. In Transformers 5.5.4 that
helper reads `padding_mask[batch_idx, kv_idx]` with broadcast tensor indices,
then converts the mask with `torch.where(mask, zero_dimensional_tensor,
python_scalar)`. Both forms can dispatch to the same failing 310P AICPU Index
path.

The current source no longer calls that helper from reading order. It builds the
exact `[B, 1, S, S]` bidirectional key-padding bias directly from full-shape
zero and negative branches, using only `unsqueeze` and `expand` afterward. The
CPU compatibility test makes the upstream helper raise if it is called, and
checks every bias value for zero-, one-, and full-prediction boundary rows.

The audit also found variable-result tensor indexing in the official layout
postprocessor: `scatter_`, boolean selection, and tensor selection. The fixed
detector and reading-order forward remain on NPU. The adapter now copies only
`logits`, `pred_boxes`, and `order_logits` to CPU before official
postprocessing. This small CPU tail prevents those variable-size operations
from reaching the 310P AICPU Index path.

The eager fix intentionally does not install the broader attention, global
pointer, sine-position, or linear rewrites used by TorchAir fullgraph layout.
U2 and the failed Pass A crossed the detector's direct gather/take-along path;
those operations are not the broadcast advanced-index form removed here. If a
new first causal model-forward failure appears, stop and report its exact
operator and call site instead of enabling all compile rewrites speculatively.

A static audit of the later recognition-prefill path found no use of the two
failing forms inside the five compiled full-vision forwards or the packed S1024
cross-KV graph. The vision graph is convolution, linear, normalization, masking
arithmetic, reductions, reshapes, and basic fixed slices. Vision masks are
filled in NumPy on CPU before transfer. The packed cross-KV graph is six pairs
of fixed-shape linear projections plus reshapes. Cache segmentation and
cross-KV padding happen after the graph through fixed contiguous slices; no
decode/scatter path runs in this task. This audit does not prove TorchAir
lowering on 310P, so preserve and report the first graph-compile error if one
occurs.

Read `AGENTS.md` and `CLAUDE.md` first. Do not edit tracked files, create a
branch, commit, or push from the work server. If the run needs a source change,
report the smallest proposed change and the first causal failure. Do not apply
the change on the work server.

## Goal

Test the latest prefill-only UniRec pipeline on the first 16 OmniDocBench pages:

```text
1 process worker
16 persistent CPU crop-preprocessing threads
compact uint8 HWC CPU-to-NPU input
five compiled masked full-vision bucket graphs
compiled packed text/cross-KV prefill
cross-KV capacity 512
cross-KV discarded after validation
no text decode
```

This is a 310P compatibility and warm-performance smoke. It is not a
representative full-dataset benchmark.

Run the exact configuration twice with the same TorchAir cache. The first run
creates or loads the cache and warms all five vision graphs. The second run
must reuse that cache. The worker still executes one warmup call for every graph
at startup on both runs; compare setup time as well as the measured producer
window.

## Hard constraints

- Use the same passed NPU activation, Python environment, OpenOCR checkout,
  UniRec checkpoint, PP-DocLayoutV2 Transformers checkpoint, and dataset from
  U2.
- Use one free physical NPU. Never stop another user's process. Do not use
  physical device 5 because the current UniRec runner rejects it.
- Keep layout eager FP32. The experiment is testing the latest recognition
  prefill path, not compiled layout.
- Use one process worker and exactly 16 recognition preprocessing threads.
- Do not enable decode, artifact persistence, multiple process workers, or a
  different cache length.
- Do not retry a failed compiled graph through an eager-only replacement.
- Do not compare a cold setup time with a warm producer time as one number.

## 1. Pull and resolve the passed environment

Run with Bash from the existing checkout:

```sh
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git status --short --branch
git pull --ff-only origin main
git status --short --branch

PROJECT_COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/venvs/unirec_full_npu_310p_py312/bin/python}"
MODEL="${MODEL:-$HOME/models/unirec-0.1b}"
LAYOUT_MODEL="${LAYOUT_MODEL:-$HOME/models/PP-DocLayoutV2_safetensors}"
IMAGES_DIR="${IMAGES_DIR:-/home/lukaiv/datasets/OmniDocBench/images}"
OPENOCR_ROOT="${OPENOCR_ROOT:-$HOME/deps/OpenOCR_0d522801}"

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -f "$LAYOUT_MODEL/model.safetensors"
test -d "$IMAGES_DIR"
test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
test "$(git -C "$OPENOCR_ROOT" rev-parse HEAD)" = \
  "0d522801ec6dc1df852c6b6d4ed6a08f5127ed97"
test -z "$(git -C "$OPENOCR_ROOT" status --short)"
```

If the pull is blocked by tracked changes, stop. Do not discard them.

Activate the same CANN/torch-npu environment and free-device selection that
passed U2. `run_prefill_export.py` requires `ASCEND_RT_VISIBLE_DEVICES` to name
the selected physical device. Confirm that it contains exactly one free device
and that it is not device 5:

```sh
test -n "${ASCEND_RT_VISIBLE_DEVICES:-}"
printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
npu-smi info
```

Create the evidence and persistent 310P-only cache roots:

```sh
OUT="$REPO/tmp/12_unirec_0_1b_inference/310p_prefill_w1_t16_$COMMIT_SHORT"
CACHE_ROOT="$REPO/.runtime_cache/12_unirec_0_1b_inference/310p_prefill_w1_t16_$COMMIT_SHORT"
mkdir -p "$OUT" "$CACHE_ROOT/layout" "$CACHE_ROOT/recognition"
```

Run the committed CPU-only compatibility checks before loading either model:

```sh
PYTHONPYCACHEPREFIX="$OUT/pycache" \
  "$PYTHON_BIN" -m unittest \
  "$REPO/12_unirec_0_1b_inference/test_layout_npu_compat.py"
```

All four tests must pass. In particular, the reading-order test must complete
while its fake Transformers `create_bidirectional_mask` raises on any call.
If the tests fail, stop and report that failure before starting Pass A.

Do not delete an existing cache. Record whether it already contains files:

```sh
find "$CACHE_ROOT" -type f -printf '%P %s bytes\n' 2>/dev/null \
  | sort >"$OUT/cache_files_before.txt"
```

## 2. Runtime gate

```sh
"$PYTHON_BIN" - <<'PY' | tee "$OUT/runtime_gate.log"
import importlib.metadata as metadata
import os
import platform
import sys

import torch
import torch_npu
import transformers

print("python=", sys.executable)
print("machine=", platform.machine())
print("torch=", torch.__version__)
print("torch_npu=", torch_npu.__version__)
print("transformers=", transformers.__version__)
print("visible_devices=", os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
print("npu_available=", torch.npu.is_available())
print("device_0=", torch.npu.get_device_name(0))
assert torch.npu.is_available()
torch.npu.set_compile_mode(jit_compile=False)
x = torch.arange(8, dtype=torch.float16, device="npu:0")
print("npu_result=", (x + 1).cpu().tolist())
PY
```

Record the U2.5 report path and classification in
`$OUT/u2_5_reference.txt`. Do not rerun U1 or U2.

## 3. Exact shared command

Both passes use these arguments:

```sh
common_args=(
  "$REPO/12_unirec_0_1b_inference/run_prefill_export.py"
  --openocr-root "$OPENOCR_ROOT"
  --model-path "$MODEL"
  --layout-model "$LAYOUT_MODEL"
  --input "$IMAGES_DIR"
  --dtype float16
  --offset 0
  --limit 16
  --workers 1
  --warmup-pages 2
  --warmup-repeats 1
  --layout-threshold 0.4
  --layout-execution eager
  --cross-cache-length 512
  --layout-cache-dir "$CACHE_ROOT/layout"
  --recognition-cache-dir "$CACHE_ROOT/recognition"
  --vision-full-batches
  --recognition-input-contract compact_uint8_hwc
  --recognition-preprocess-threads 16
  --vision-page-lookahead 4
  --artifact-storage discard
  --profile-prefill-device-stages
)
```

The current runner automatically warms all five fixed full-vision graphs during
worker setup. The two warmup pages then exercise the complete page-to-cross-KV
path outside the measured producer window. The measured window processes all
16 pages.

## 4. Pass A: cold/cache-populating run

```sh
PASS_A="$OUT/pass_a_cold"
mkdir -p "$PASS_A/output"
command_a=("$PYTHON_BIN" "${common_args[@]}" --output-dir "$PASS_A/output")

{
  printf 'project_commit=%s\n' "$PROJECT_COMMIT"
  printf 'physical_devices=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  printf 'command='
  printf '%q ' "${command_a[@]}"
  printf '\n'
} >"$PASS_A/command.txt"

export PYTHONUNBUFFERED=1
set -o pipefail
SECONDS=0
"${command_a[@]}" 2>&1 | tee "$PASS_A/run.log"
PASS_A_STATUS="${PIPESTATUS[0]}"
printf '%s\n' "$PASS_A_STATUS" >"$PASS_A/exit_code.txt"
printf '%s\n' "$SECONDS" >"$PASS_A/wall_seconds.txt"
test "$PASS_A_STATUS" = 0
test -f "$PASS_A/output/summary.json"
```

The first graph compilation can be quiet and substantially slower than a 910B
cache load. Do not classify five minutes of quiet output as a hang. While it is
running, check only the owned process and selected-device activity. If neither
the log nor CPU/NPU activity changes for 15 minutes, report the state before
stopping the owned process. Never stop an unowned process.

If Pass A fails, stop. Preserve the complete traceback and CANN/TorchAir error.
Do not run Pass B and do not change the graph shapes or execution mode.

Record the populated cache inventory:

```sh
find "$CACHE_ROOT" -type f -printf '%P %s bytes\n' 2>/dev/null \
  | sort >"$OUT/cache_files_after_pass_a.txt"
du -sh "$CACHE_ROOT" | tee "$OUT/cache_size_after_pass_a.txt"
```

## 5. Pass B: warm repeat

Run the exact same configuration in a fresh process and reuse the same cache:

```sh
PASS_B="$OUT/pass_b_warm"
mkdir -p "$PASS_B/output"
command_b=("$PYTHON_BIN" "${common_args[@]}" --output-dir "$PASS_B/output")

{
  printf 'project_commit=%s\n' "$PROJECT_COMMIT"
  printf 'physical_devices=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  printf 'command='
  printf '%q ' "${command_b[@]}"
  printf '\n'
} >"$PASS_B/command.txt"

SECONDS=0
"${command_b[@]}" 2>&1 | tee "$PASS_B/run.log"
PASS_B_STATUS="${PIPESTATUS[0]}"
printf '%s\n' "$PASS_B_STATUS" >"$PASS_B/exit_code.txt"
printf '%s\n' "$SECONDS" >"$PASS_B/wall_seconds.txt"
test "$PASS_B_STATUS" = 0
test -f "$PASS_B/output/summary.json"
```

Do not run a third performance pass unless Luka asks for it.

## 6. Required checks and report

For both `summary.json` files verify and report:

- `status == "ok"`;
- `offset == 0`, `limit == 16`, and `workers == 1`;
- `artifact_storage == "discard"`;
- `cross_cache_length == 512`;
- `vision_full_batches == true`;
- `recognition_input_contract == "compact_uint8_hwc"`;
- `recognition_preprocess_threads == 16`;
- `use_chart_recognition == true`;
- validation passed and emitted crop count is nonzero;
- worker graph warmup reports all five compiled full-vision buckets;
- no first-call graph appeared after the explicit graph warmup;
- compiled real rows, physical rows, padding rows, slot efficiency, and eager
  fallback count;
- skipped/rejected crop count and reasons;
- `setup_s`, warmup wall, `producer_stream_wall_s`, `producer_wall_s`,
  `shutdown_s`, and `total_wall_s`;
- pages/s, crops/s, and real source tokens/s;
- device-stage timing for full vision and cross-KV/text prefill;
- main-process maximum RSS;
- selected physical NPU and observed HBM before, during, and after the run.

Compare Pass A and Pass B, but use Pass B for the warm-performance result.
Clearly separate graph setup/warmup from the measured 16-page producer window.
Do not call `total_wall_s` steady-state prefill throughput.

Write the final report to:

```text
tmp/12_unirec_0_1b_inference/310p_prefill_w1_t16_<commit>/agent_report.md
```

Classify the task as:

- `PASS_WARM`: both passes succeeded, all five graphs warmed, and the warm pass
  emitted valid descriptors for all completed pages;
- `FAIL_COMPILE`: the first causal failure was compilation or compiled graph
  execution;
- `FAIL_INTEGRATION`: another failure prevented a valid warm result.

Report back to Luka with this compact line followed by the report path:

```text
UNIREC_310P_PREFILL_W1_T16: <classification> — pages=16 crops=<n> warm_producer=<s> warm_pg_s=<n> setup_cold=<s> setup_warm=<s> five_graphs=<yes/no> fallback=<n> skipped=<n> peak_hbm=<MiB>; report=<path>
```

Then stop. Do not edit the source or start a larger page count.
