# Work-server 310P compiled-layout UniRec prefill, first 128 pages

This is a pull-only Atlas 310P experiment for the work-server agent. It tests
only the page-to-cross-KV producer. Do not run text decode, retain artifacts,
assemble pages, or evaluate OmniDocBench accuracy.

The matched 910B2 reference used commit `4747d8e`, physical NPU 6, the first
128 sorted OmniDocBench images, and the exact W8/T8 configuration below. Its
measured producer result was:

```text
producer_wall_s             4.6798825581
pages_per_s                27.3511137111
crops                     950
real_source_tokens      61,520
physical_source_tokens 129,024
real_source_tokens_per_s 13,145.6290274
vision real/physical rows 950 / 1,456
vision slot efficiency      0.6524725275
vision fallback crops       1
cross-KV-512 rejected crops 6
setup_s                   109.7331936117
shutdown_s                 15.9984996249
```

The 910B rate is the producer window only. It excludes graph/model setup,
warmup, shutdown, decode, page assembly, output writing, and evaluation.

## Goal and execution order

1. Pull the current `main` and preserve all previous 310P evidence and caches.
2. Reuse the recognition cache from the successful 16-page W1/T16 310P run.
3. Use one new 310P-only layout-cache root.
4. Run an eight-page W1/T16 compiled-layout gate. This is the only process
   allowed to populate the new layout graph cache.
5. Only if that gate passes, run the matched first-128 W8/T8 producer.
6. Report the matched producer result and the 310P/910B ratios, then stop.

This ordering is mandatory. Eight cold workers must not write the same layout
cache concurrently.

## Restrictions

- Read `CLAUDE.md` and `AGENTS.md` first.
- Do not edit tracked files, create a branch, commit, or push on the work server.
- Use one genuinely free physical 310P. Never stop another user's process.
- Do not use physical device 5.
- Keep layout FP32 and recognition FP16.
- Do not copy any 910B graph cache to 310P.
- Do not silently replace a failed compiled graph with eager execution.
- Preserve a complete first causal traceback and CANN/TorchAir error.
- Setup and graph warmup are not producer throughput.

## 1. Pull and recover the passed environment

Run from the existing checkout with Bash:

```sh
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git status --short --branch
git pull --ff-only origin main
git status --short --branch

COMMIT="$(git rev-parse HEAD)"
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

Activate the same Python/CANN/torch-npu environment and free-device selection
that passed the prior 16-page W1/T16 run. Verify that exactly one free physical
device is exposed and it is not device 5:

```sh
test -n "${ASCEND_RT_VISIBLE_DEVICES:-}"
printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
npu-smi info
```

## 2. Resolve caches and output roots

Set `PRIOR_CACHE_ROOT` to the exact cache root recorded by the successful
16-page W1/T16 report. That root already contains the six validated 310P
recognition-prefill graphs. Do not guess and do not select a failed run:

```sh
test -n "${PRIOR_CACHE_ROOT:-}"
PRIOR_CACHE_ROOT="$(readlink -f "$PRIOR_CACHE_ROOT")"
RECOGNITION_CACHE="$PRIOR_CACHE_ROOT/recognition"
test -d "$RECOGNITION_CACHE"

OUT="$REPO/tmp/12_unirec_0_1b_inference/310p_compiled_prefill_first128_$COMMIT_SHORT"
LAYOUT_CACHE="$REPO/.runtime_cache/12_unirec_0_1b_inference/310p_compiled_prefill_first128_$COMMIT_SHORT/layout"
mkdir -p "$OUT" "$LAYOUT_CACHE"
```

Before running, calculate the current source hashes and record the recognition
graph directories. Require the exact five current full-vision bucket directories
and the exact current packed S1024 text-prefill directory. If any exact directory
is missing, stop and report `CACHE_SOURCE_MISMATCH` rather than launching cold
recognition graphs:

```sh
find "$RECOGNITION_CACHE" -maxdepth 1 -type d -printf '%f\n' \
  | sort >"$OUT/recognition_cache_dirs.txt"

HASHES="$OUT/current_recognition_source_hashes.txt"
(
  cd "$REPO/12_unirec_0_1b_inference"
  "$PYTHON_BIN" - <<'PY'
import text_packed_prefill
import vision_full_batch

print(f"VISION_SOURCE_HASH={vision_full_batch._source_hash()}")
print(f"TEXT_SOURCE_HASH={text_packed_prefill._source_hash()}")
PY
) | tee "$HASHES"

VISION_SOURCE_HASH="$(sed -n 's/^VISION_SOURCE_HASH=//p' "$HASHES")"
TEXT_SOURCE_HASH="$(sed -n 's/^TEXT_SOURCE_HASH=//p' "$HASHES")"
test -n "$VISION_SOURCE_HASH"
test -n "$TEXT_SOURCE_HASH"

for graph in \
  "vision_full_bucket_960x64_b16_float16_src$VISION_SOURCE_HASH" \
  "vision_full_bucket_512x256_b16_float16_src$VISION_SOURCE_HASH" \
  "vision_full_bucket_960x256_b4_float16_src$VISION_SOURCE_HASH" \
  "vision_full_bucket_512x512_b8_float16_src$VISION_SOURCE_HASH" \
  "vision_full_bucket_960x512_b4_float16_src$VISION_SOURCE_HASH" \
  "text_prefill_packed_b1_s1024_float16_src$TEXT_SOURCE_HASH"
do
  test -d "$RECOGNITION_CACHE/$graph" || {
    printf 'CACHE_SOURCE_MISMATCH missing=%s\n' "$graph"
    exit 1
  }
done
```

Run the committed CPU compatibility tests:

```sh
PYTHONPYCACHEPREFIX="$OUT/pycache" \
  "$PYTHON_BIN" -m unittest \
  "$REPO/12_unirec_0_1b_inference/test_layout_npu_compat.py" \
  | tee "$OUT/cpu_compatibility_tests.log"
```

All four tests must pass.

## 3. Shared arguments

```sh
common_args=(
  "$REPO/12_unirec_0_1b_inference/run_prefill_export.py"
  --openocr-root "$OPENOCR_ROOT"
  --model-path "$MODEL"
  --layout-model "$LAYOUT_MODEL"
  --input "$IMAGES_DIR"
  --dtype float16
  --offset 0
  --warmup-repeats 1
  --layout-threshold 0.4
  --layout-execution torchair
  --layout-batch-size 1
  --cross-cache-length 512
  --layout-cache-dir "$LAYOUT_CACHE"
  --recognition-cache-dir "$RECOGNITION_CACHE"
  --vision-full-batches
  --recognition-input-contract compact_uint8_hwc
  --vision-page-lookahead 4
  --artifact-storage discard
  --profile-prefill-device-stages
)
```

## 4. Pass A: single-worker compiled-layout gate

This pass is allowed to compile/populate the single B1 800x800 FP32 layout
graph. Recognition graphs must load from the passed 310P cache.

```sh
PASS_A="$OUT/pass_a_w1_t16_gate"
mkdir -p "$PASS_A/output"
command_a=(
  "$PYTHON_BIN" "${common_args[@]}"
  --output-dir "$PASS_A/output"
  --limit 8
  --workers 1
  --warmup-pages 2
  --recognition-preprocess-threads 16
)

{
  printf 'commit=%s\n' "$COMMIT"
  printf 'physical_devices=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  printf 'command='
  printf '%q ' "${command_a[@]}"
  printf '\n'
} >"$PASS_A/command.txt"

export PYTHONUNBUFFERED=1
set -o pipefail
SECONDS=0
"${command_a[@]}" 2>&1 | tee "$PASS_A/run.log"
STATUS_A="${PIPESTATUS[0]}"
printf '%s\n' "$STATUS_A" >"$PASS_A/exit_code.txt"
printf '%s\n' "$SECONDS" >"$PASS_A/wall_seconds.txt"
test "$STATUS_A" = 0
test -f "$PASS_A/output/summary.json"
```

Require `status=ok`, `layout_execution=torchair`, eight pages, nonzero crops,
validation passed, five vision graphs, no unexpected recognition first-call
graph, and no eager-only layout retry. If Pass A fails, stop. Do not run Pass B.

## 5. Pass B: matched first-128 W8/T8 producer

Run only after Pass A populated and validated the layout cache:

```sh
PASS_B="$OUT/pass_b_w8_t8_first128"
mkdir -p "$PASS_B/output"
command_b=(
  "$PYTHON_BIN" "${common_args[@]}"
  --output-dir "$PASS_B/output"
  --limit 128
  --workers 8
  --warmup-pages 8
  --recognition-preprocess-threads 8
)

{
  printf 'commit=%s\n' "$COMMIT"
  printf 'physical_devices=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  printf 'command='
  printf '%q ' "${command_b[@]}"
  printf '\n'
} >"$PASS_B/command.txt"

SECONDS=0
"${command_b[@]}" 2>&1 | tee "$PASS_B/run.log"
STATUS_B="${PIPESTATUS[0]}"
printf '%s\n' "$STATUS_B" >"$PASS_B/exit_code.txt"
printf '%s\n' "$SECONDS" >"$PASS_B/wall_seconds.txt"
test "$STATUS_B" = 0
test -f "$PASS_B/output/summary.json"
```

The graph-cache import and per-worker graph warmups can take much longer than
the measured 128-page producer. Do not include them in pages/s. Do not classify
quiet cache loading as a hang while the owned process or selected NPU remains
active. Never terminate an unowned process.

## 6. Required result and compact report

For Pass B verify:

- `status == "ok"`, `offset == 0`, `limit == 128`, `workers == 8`;
- `layout_execution == "torchair"` and `layout_batch_size == 1`;
- compiled full-vision buckets and compiled packed S1024 cross-KV;
- `artifact_storage == "discard"` and validation passed;
- page count 128 and nonzero crops;
- all eight workers completed pages;
- no unexpected post-warmup graph first calls;
- real/physical source tokens, vision real/physical rows, slot efficiency;
- fallback and cross-KV rejection counts;
- producer wall and pages/s;
- stage totals for layout, CPU crop preparation, recognition prefill, D2H,
  shared packing, file/decode, and IPC delivery;
- setup, warmup, shutdown, total wall, peak HBM, and maximum RSS.

Compute these matched ratios against the 910B reference above:

```text
310P / 910B pages_per_s
310P / 910B real_source_tokens_per_s
910B / 310P producer_wall_s
```

Write:

```text
tmp/12_unirec_0_1b_inference/310p_compiled_prefill_first128_<commit>/agent_report.md
```

Report back in one compact line:

```text
UNIREC_310P_COMPILED_PREFILL_128: <PASS|FAIL_COMPILE|FAIL_INTEGRATION|OOM> — producer=<s> pg_s=<n> real_tok_s=<n> ratio_vs_910b=<n> crops=<n> slot_eff=<n> fallback=<n> rejected=<n> setup=<s> peak_hbm=<MiB>; report=<path>
```

Then stop. Do not start decode, a larger run, or a worker-count sweep.
