# Work-server 310P eager-layout UniRec prefill, first 128 pages

This is the fallback after the compiled PP-DocLayoutV2 Pass A failed during
worker-setup graph warmup with an `IndexByTensor` exception. The traceback
reached `layout_process_pool` warmup, `_predict_batch`, and
`torch.npu.synchronize()` before any page-processing progress event. Preserve
that failed run. Do not retry compiled layout in this task.

Run the page-to-cross-KV producer only. Use eager FP32 layout on NPU, while
retaining the six already validated compiled FP16 recognition-prefill graphs:
five masked full-vision buckets and packed S1024 cross-KV projection. Do not
run text decode, retain cross-KV artifacts, assemble pages, or score accuracy.

## Comparisons

The successful 310P W1/T16 eager-layout 16-page warm run reported:

```text
pages_per_s                         2.78
layout_s                            3.00  (51%)
npu_vision_and_cross_kv_prefill_s   1.35
cpu_crop_preprocess_s               0.77
pack_and_ipc_delivery_s             0.41
file_io_s                           0.22
d2h_cache_transfer_s                0.07
real / physical vision rows         91 / 136
vision slot efficiency              0.669
fallback / skipped                  0 / 0
peak HBM                            981 MiB
producer HBM increment               44.6 MiB
```

The current 910B2 first-128 W8/T8 best-setting reference used compiled layout
and reported:

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
```

The 310P/910B producer comparison is useful, but it is not a pure chip ratio:
the 310P layout is eager and the 910B layout is compiled. Keep that difference
explicit in the report.

## Restrictions

- Read `CLAUDE.md` and `AGENTS.md` first.
- Pull only. Do not edit tracked files, branch, commit, or push on the server.
- Use exactly one genuinely free physical 310P. Never use device 5.
- Never stop another user's process.
- Keep layout eager FP32 and recognition FP16.
- Reuse the passed 310P recognition graph cache. Do not copy a 910B cache.
- If eight-worker setup OOMs, preserve the first failure and stop. Do not
  silently reduce the worker count.
- Setup and warmup are outside producer throughput.

## 1. Pull and resolve the passed environment

Run with Bash:

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
test -n "${ASCEND_RT_VISIBLE_DEVICES:-}"
printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
npu-smi info
```

If pull is blocked by tracked changes, stop. Do not discard them. Use the same
Python/CANN/torch-npu activation and free-device selection that passed the
W1/T16 run.

## 2. Reuse and verify the current recognition cache

Set `PRIOR_CACHE_ROOT` to the exact successful 310P W1/T16 cache root. Do not
select the failed compiled-layout cache:

```sh
test -n "${PRIOR_CACHE_ROOT:-}"
PRIOR_CACHE_ROOT="$(readlink -f "$PRIOR_CACHE_ROOT")"
RECOGNITION_CACHE="$PRIOR_CACHE_ROOT/recognition"
test -d "$RECOGNITION_CACHE"

OUT="$REPO/tmp/12_unirec_0_1b_inference/310p_eager_layout_prefill_first128_$COMMIT_SHORT"
UNUSED_LAYOUT_CACHE="$OUT/unused_layout_cache"
mkdir -p "$OUT" "$UNUSED_LAYOUT_CACHE"

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

PYTHONPYCACHEPREFIX="$OUT/pycache" \
  "$PYTHON_BIN" -m unittest \
  "$REPO/12_unirec_0_1b_inference/test_layout_npu_compat.py" \
  | tee "$OUT/cpu_compatibility_tests.log"
```

All four compatibility tests must pass.

## 3. Run first-128 W8/T8 with eager layout

```sh
RUN="$OUT/w8_t8_first128"
mkdir -p "$RUN/output"
command=(
  "$PYTHON_BIN"
  "$REPO/12_unirec_0_1b_inference/run_prefill_export.py"
  --openocr-root "$OPENOCR_ROOT"
  --model-path "$MODEL"
  --layout-model "$LAYOUT_MODEL"
  --input "$IMAGES_DIR"
  --output-dir "$RUN/output"
  --dtype float16
  --offset 0
  --limit 128
  --workers 8
  --warmup-pages 8
  --warmup-repeats 1
  --layout-threshold 0.4
  --layout-execution eager
  --layout-batch-size 1
  --cross-cache-length 512
  --layout-cache-dir "$UNUSED_LAYOUT_CACHE"
  --recognition-cache-dir "$RECOGNITION_CACHE"
  --vision-full-batches
  --recognition-input-contract compact_uint8_hwc
  --recognition-preprocess-threads 8
  --vision-page-lookahead 4
  --artifact-storage discard
  --profile-prefill-device-stages
)

{
  printf 'commit=%s\n' "$COMMIT"
  printf 'physical_devices=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} >"$RUN/command.txt"

export PYTHONUNBUFFERED=1
set -o pipefail
SECONDS=0
"${command[@]}" 2>&1 | tee "$RUN/run.log"
STATUS="${PIPESTATUS[0]}"
printf '%s\n' "$STATUS" >"$RUN/exit_code.txt"
printf '%s\n' "$SECONDS" >"$RUN/wall_seconds.txt"
test "$STATUS" = 0
test -f "$RUN/output/summary.json"
```

Graph-cache import and eight per-worker recognition warmups can be much longer
than the producer window. Do not treat quiet cache loading as a hang while the
owned process or selected NPU remains active.

## 4. Validate and report

Require:

- `status == "ok"`, 128 pages, offset 0, eight workers, eight threads each;
- `layout_execution == "eager"` and `layout_batch_size == 1`;
- five compiled full-vision graphs and compiled packed S1024 cross-KV;
- compact uint8 HWC input, cross-KV 512, discard storage;
- all eight workers completed pages and validation passed;
- no unexpected recognition graph first calls after worker warmup.

Report producer wall, pages/s, crops, real source tokens/s, real/physical vision
rows, slot efficiency, fallback and rejected counts, worker page distribution,
layout time, CPU crop preparation, NPU recognition prefill, D2H, packing, IPC,
file I/O, setup, warmup, shutdown, total wall, peak HBM, and maximum RSS.

Compute:

```text
310P W8/T8 pages_per_s / 2.78                  # scaling vs 310P W1/T16
310P W8/T8 pages_per_s / 27.3511137111         # producer ratio vs 910B
310P real_source_tokens_per_s / 13145.6290274  # token ratio vs 910B
```

Write the report under `$OUT/agent_report.md`, then report one compact line:

```text
UNIREC_310P_EAGER_LAYOUT_PREFILL_128: <PASS|OOM|FAIL_INTEGRATION> — producer=<s> pg_s=<n> real_tok_s=<n> scale_vs_310p_w1=<n> ratio_vs_910b=<n> crops=<n> slot_eff=<n> fallback=<n> rejected=<n> layout=<s> setup=<s> peak_hbm=<MiB>; report=<path>
```

Then stop. Do not retry compiled layout, run decode, increase the page count,
or sweep worker counts.
