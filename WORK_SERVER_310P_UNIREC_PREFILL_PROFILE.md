# 310P UniRec prefill profile against the matched 910B reference

Profile the large 310P-versus-910B gaps in the first-128 W1/T16 prefill-only
producer. Do not rerun decode or OmniDocBench evaluation. The repository now
contains the exact 910B graph and 128-page layout references, so no benchmark
data needs to be copied from Luka.

The profiler covers these already-warmed production graphs independently:

- compiled FP32 PP-DocLayoutV2 B1 800x800;
- all five compiled FP16 full-vision buckets;
- compiled FP16 packed B1 S1024 cross-KV projection.

It also runs the existing full 128-page layout lab to separate the compiled
forward from image processing, H2D, D2H, box decode, and postprocessing. The
analyzer weights every graph by the exact first-128 W1 call histogram and joins
the result to the existing 310P W1 producer summary.

## Restrictions

- Pull only. Do not edit tracked files, branch, commit, or push.
- Do not create an agent report. Normal profiler artifacts under `tmp/` are
  required and may remain local.
- Use one genuinely free physical 310P. Never use physical device 5.
- Do not stop another user's process.
- Use the existing project-local `./venv`, `./deps`, `./models`, and graph
  caches. If their names differ, set the variables below; do not move them.
- Do not copy any 910B graph cache to 310P.
- Do not silently fall back to eager execution.
- Warmup, cache loading, and native-profiler export are diagnostic setup. The
  before/after NPU-event controls are the latency authority.
- Do not profile the whole 128-page producer. That would create a huge trace and
  obscure the fixed-graph comparison.

## 1. Pull and resolve the passed environment

Run with Bash after activating the existing CANN/torch-npu environment and
selecting one free physical device:

```sh
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
set -euo pipefail
git status --short --branch
git pull --ff-only origin main
git status --short --branch

COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
IMAGES_DIR="${IMAGES_DIR:?set IMAGES_DIR to the existing 1651-page image directory}"
RECOGNITION_CACHE="${RECOGNITION_CACHE:?set RECOGNITION_CACHE to the passed six-graph 310P cache root}"
LAYOUT_CACHE="${LAYOUT_CACHE:?set LAYOUT_CACHE to the passed compiled-layout cache root}"
W1_SUMMARY="${W1_SUMMARY:-$REPO/tmp/12_unirec_0_1b_inference/310p_compiled_layout_prefill_128_w1_w8_7821ad5/w1_t16/output/summary.json}"

for path in "$PYTHON_BIN" "$MODEL" "$LAYOUT_MODEL" "$OPENOCR_ROOT" \
  "$IMAGES_DIR" "$RECOGNITION_CACHE" "$LAYOUT_CACHE" "$W1_SUMMARY"
do
  test -e "$path"
done
test -n "${ASCEND_RT_VISIBLE_DEVICES:-}"
case ",$ASCEND_RT_VISIBLE_DEVICES," in
  *,5,*) printf 'REJECTED_PHYSICAL_DEVICE_5\n'; exit 1 ;;
esac
test "$(printf '%s' "$ASCEND_RT_VISIBLE_DEVICES" | awk -F, '{print NF}')" = 1
```

If project-local paths have different names, set the corresponding variables.
Do not redownload or move a passed artifact merely to match a default.

Verify the runtime and exact immutable inputs:

```sh
"$PYTHON_BIN" - <<'PY'
import torch
import torch_npu
import transformers

print("torch=", torch.__version__)
print("torch_npu=", torch_npu.__version__)
print("transformers=", transformers.__version__)
print("device=", torch.npu.get_device_name(0))
assert torch.npu.is_available()
assert torch.npu.device_count() == 1
PY

test "$(git -C "$OPENOCR_ROOT" rev-parse HEAD)" = \
  0d522801ec6dc1df852c6b6d4ed6a08f5127ed97
test "$(stat -c %s "$MODEL/model.pth")" = 535901578
test "$(stat -c %s "$LAYOUT_MODEL/model.safetensors")" = 214798436
printf '%s  %s\n' \
  b253951f80c6c2299768332b72845a5c3f52e73713a4ee2165a4bad1dfac7bef \
  "$MODEL/model.pth" | sha256sum -c -
printf '%s  %s\n' \
  e60f3725aeedc88fd319416ef166bda79171a41516a301c27cab9132dc2739d2 \
  "$LAYOUT_MODEL/model.safetensors" | sha256sum -c -
```

## 2. Require the exact current graph caches

```sh
HASHES="$REPO/tmp/12_unirec_0_1b_inference/profile_graph_hashes_$COMMIT_SHORT.txt"
mkdir -p "$(dirname "$HASHES")"
(
  cd "$REPO/12_unirec_0_1b_inference"
  "$PYTHON_BIN" - <<'PY'
import hashlib
from pathlib import Path
import text_packed_prefill
import vision_full_batch

print("LAYOUT_SOURCE_HASH=" + hashlib.sha256(
    Path("layout_torchair.py").read_bytes()
).hexdigest()[:12])
print("VISION_SOURCE_HASH=" + vision_full_batch._source_hash())
print("TEXT_SOURCE_HASH=" + text_packed_prefill._source_hash())
PY
) | tee "$HASHES"

LAYOUT_SOURCE_HASH="$(sed -n 's/^LAYOUT_SOURCE_HASH=//p' "$HASHES")"
VISION_SOURCE_HASH="$(sed -n 's/^VISION_SOURCE_HASH=//p' "$HASHES")"
TEXT_SOURCE_HASH="$(sed -n 's/^TEXT_SOURCE_HASH=//p' "$HASHES")"

for graph in \
  "layout_b1_800x800_float32_src$LAYOUT_SOURCE_HASH"
do
  test -d "$LAYOUT_CACHE/$graph"
  test -n "$(find "$LAYOUT_CACHE/$graph" -type f -print -quit)"
done

for graph in \
  "vision_full_bucket_960x64_b16_float16_src$VISION_SOURCE_HASH" \
  "vision_full_bucket_512x256_b16_float16_src$VISION_SOURCE_HASH" \
  "vision_full_bucket_960x256_b4_float16_src$VISION_SOURCE_HASH" \
  "vision_full_bucket_512x512_b8_float16_src$VISION_SOURCE_HASH" \
  "vision_full_bucket_960x512_b4_float16_src$VISION_SOURCE_HASH" \
  "text_prefill_packed_b1_s1024_float16_src$TEXT_SOURCE_HASH"
do
  test -d "$RECOGNITION_CACHE/$graph"
  test -n "$(find "$RECOGNITION_CACHE/$graph" -type f -print -quit)"
done
```

Stop if any graph is missing. Do not compile during this diagnostic comparison;
the successful W1/W8 run established that these caches already exist.

## 3. Run local analyzer tests

```sh
OUT="$REPO/tmp/12_unirec_0_1b_inference/310p_prefill_profile_$COMMIT_SHORT"
test ! -e "$OUT"
mkdir -p "$OUT"
PYTHONPYCACHEPREFIX="$OUT/pycache" \
  "$PYTHON_BIN" -m unittest \
  "$REPO/12_unirec_0_1b_inference/test_analyze_prefill_profiles.py" \
  -v 2>&1 | tee "$OUT/analyzer_tests.log"
```

## 4. Profile the seven warmed fixed graphs

```sh
GRAPH_OUT="$OUT/graph_suite"
graph_command=(
  "$PYTHON_BIN"
  "$REPO/12_unirec_0_1b_inference/profile_prefill_graph_suite.py"
  --model-path "$MODEL"
  --layout-model "$LAYOUT_MODEL"
  --layout-cache-dir "$LAYOUT_CACHE"
  --recognition-cache-dir "$RECOGNITION_CACHE"
  --output-dir "$GRAPH_OUT"
  --device npu:0
  --warmup 2
  --control-repeats 10
  --profile-steps 1
  --profile-metric pipe
  --parser-topn 50
)
printf '%q ' "${graph_command[@]}" >"$OUT/graph_command.txt"
printf '\n' >>"$OUT/graph_command.txt"
set -o pipefail
"${graph_command[@]}" 2>&1 | tee "$OUT/graph_suite.log"
test "${PIPESTATUS[0]}" = 0
test -f "$GRAPH_OUT/profile_suite_summary.json"
```

The suite creates seven native-profiler directories. Keep them. Do not compare
the profiler's host wall against production throughput; compare the steady NPU
event controls and use kernel CSV totals only to attribute the difference.

## 5. Run the same full 128-page layout stage lab

```sh
LAYOUT_OUT="$OUT/layout_first128.json"
layout_command=(
  "$PYTHON_BIN"
  "$REPO/12_unirec_0_1b_inference/layout_detector_lab.py"
  --openocr-root "$OPENOCR_ROOT"
  --model-path "$LAYOUT_MODEL"
  --input "$IMAGES_DIR"
  --output "$LAYOUT_OUT"
  --device npu:0
  --execution torchair
  --compile-cache-dir "$LAYOUT_CACHE"
  --dtype float32
  --threshold 0.4
  --offset 0
  --limit 128
  --warmup-pages 2
)
printf '%q ' "${layout_command[@]}" >"$OUT/layout_command.txt"
printf '\n' >>"$OUT/layout_command.txt"
"${layout_command[@]}" 2>&1 | tee "$OUT/layout_lab.log"
test "${PIPESTATUS[0]}" = 0
test -f "$LAYOUT_OUT"
```

## 6. Analyze the 310P/910B difference automatically

```sh
ANALYSIS="$OUT/gap_analysis.json"
"$PYTHON_BIN" \
  "$REPO/12_unirec_0_1b_inference/analyze_prefill_profiles.py" \
  --npu310-graph-profile "$GRAPH_OUT/profile_suite_summary.json" \
  --npu310-layout-lab "$LAYOUT_OUT" \
  --npu310-producer-w1 "$W1_SUMMARY" \
  --output "$ANALYSIS" \
  2>&1 | tee "$OUT/gap_analysis.log"
test "${PIPESTATUS[0]}" = 0
test -f "$ANALYSIS"
```

The analyzer prints:

- whole-stage versus compiled-graph versus surrounding-work ratios;
- the percentage of each producer-stage gap explained by graph replay;
- all full-layout substage ratios;
- kernel-type and shape deltas for layout and the dominant vision bucket;
- one ranked first optimization target.

Inspect `gap_analysis.json` before concluding. In particular:

- Same kernel counts but slower kernels indicate a kernel/hardware efficiency
  gap, not extra Python/model operations.
- More kernels or much more `TransData` time indicates poorer graph lowering or
  format churn on 310P.
- A large `Index`/AI-CPU delta in layout indicates that a shape-static indexing
  operation still falls to AI CPU and should be rewritten.
- A much lower cube-utilization value plus slow MatMul/Conv indicates poor 310P
  utilization for these shapes.
- A graph ratio near 1 but a large surrounding ratio points to H2D,
  normalization, output compaction, synchronization, or host postprocessing.
- Do not propose cross-KV optimization merely because its ratio is high. It is
  only important if its weighted absolute gap is material.

## Return only three short lines

Do not make Luka transcribe tables. Return exactly:

```text
Gap location: layout graph <ratio>x (<share>% of gap); vision graphs <ratio>x (<share>% of recognition gap); cross-KV <ratio>x; surrounding layout/recognition <ratio>x/<ratio>x.
Kernel cause: layout <largest type or shape>; vision <largest type or shape>; kernel counts <same/different>; cube utilization <310P versus 910B>.
First fix: <one concrete code/runtime target>, because it contributes <weighted seconds> of the first-128 gap.
```

If any gate fails, return one short failure line with the failing phase and the
first causal error instead.
