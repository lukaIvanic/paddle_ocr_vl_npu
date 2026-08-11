# 310P UniRec compiled-layout prefill: first 128 pages, W1 and W8

Run only the page-to-cross-KV producer. Do not run decode, page assembly, or
OmniDocBench evaluation. Run the same first 128 sorted pages twice:

1. W1/T16;
2. W8/T8.

Both runs use compiled FP32 PP-DocLayoutV2, the five compiled FP16 full-vision
graphs, compiled packed S1024 cross-KV prefill, cross-KV capacity 512, compact
uint8 HWC crop transfer, and discard storage.

The matched 910B2 W8/T8 producer reference is:

```text
producer_wall_s              4.6798825581
pages_per_s                 27.3511137111
real_source_tokens_per_s 13145.6290274
crops                         950
real_source_tokens          61520
physical_source_tokens     129024
vision real/physical rows 950 / 1456
vision slot efficiency         0.6524725275
vision fallback crops          1
cross-KV-512 rejected crops     6
```

This is a producer-window comparison. Setup, graph loading, warmup, shutdown,
decode, and evaluation are excluded from pages/s.

The prior 310P eager-layout results were 2.79 pages/s for the reported W1/T16
run and 5.11 pages/s for the first-128 W8/T8 run. The W8 comparison is matched;
the earlier W1 run used only 16 pages, so treat its ratio as context only.

## Restrictions

- Pull only. Do not edit tracked files, branch, commit, or push.
- Print the final compact result to stdout. Do not write an agent report.
- Use one genuinely free physical 310P. Never use physical device 5.
- Reuse the exact recognition cache from the successful rebuilt W1/W8 runs.
- Reuse the exact layout cache from the successful compiled-layout probe.
- Do not copy any 910B cache to 310P.
- Do not silently fall back from compiled layout or recognition graphs.
- Run W1 first. Run W8 only after W1 passes all checks.
- If W8 OOMs, preserve the first causal error and report `OOM_W8`.
- Setup and warmup are not producer throughput.

## 1. Pull and resolve the project-local environment

Run with Bash from the existing checkout:

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
RECOGNITION_CACHE="${RECOGNITION_CACHE:?set RECOGNITION_CACHE to the successful six-graph 310P cache root}"
LAYOUT_CACHE="${LAYOUT_CACHE:?set LAYOUT_CACHE to the successful compiled-layout probe cache root}"

PYTHON_BIN="$(readlink -f "$PYTHON_BIN")"
MODEL="$(readlink -f "$MODEL")"
LAYOUT_MODEL="$(readlink -f "$LAYOUT_MODEL")"
OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
IMAGES_DIR="$(readlink -f "$IMAGES_DIR")"
RECOGNITION_CACHE="$(readlink -f "$RECOGNITION_CACHE")"
LAYOUT_CACHE="$(readlink -f "$LAYOUT_CACHE")"

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -f "$LAYOUT_MODEL/model.safetensors"
test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
test -d "$IMAGES_DIR"
test -d "$RECOGNITION_CACHE"
test -d "$LAYOUT_CACHE"
test -n "${ASCEND_RT_VISIBLE_DEVICES:-}"
case ",$ASCEND_RT_VISIBLE_DEVICES," in
  *,5,*) printf 'REJECTED_PHYSICAL_DEVICE_5\n'; exit 1 ;;
esac
printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
```

If local project directories have different names, set the corresponding
variables. Do not move or redownload passed artifacts merely to match these
defaults. If `git pull` is blocked by tracked changes, stop without discarding
them.

Verify the exact model and dependency identities:

```sh
test "$(git -C "$OPENOCR_ROOT" rev-parse HEAD)" = \
  0d522801ec6dc1df852c6b6d4ed6a08f5127ed97
test -z "$(git -C "$OPENOCR_ROOT" status --short)"
test "$(stat -c %s "$MODEL/model.pth")" = 535901578
test "$(stat -c %s "$LAYOUT_MODEL/model.safetensors")" = 214798436
printf '%s  %s\n' \
  b253951f80c6c2299768332b72845a5c3f52e73713a4ee2165a4bad1dfac7bef \
  "$MODEL/model.pth" | sha256sum -c -
printf '%s  %s\n' \
  e60f3725aeedc88fd319416ef166bda79171a41516a301c27cab9132dc2739d2 \
  "$LAYOUT_MODEL/model.safetensors" | sha256sum -c -
```

## 2. Verify the passed graph caches

Require the layout graph that just passed the standalone 310P probe:

```sh
LAYOUT_SOURCE_HASH="$(sha256sum \
  "$REPO/12_unirec_0_1b_inference/layout_torchair.py" \
  | awk '{print substr($1, 1, 12)}')"
LAYOUT_GRAPH="layout_b1_800x800_float32_src$LAYOUT_SOURCE_HASH"
test -d "$LAYOUT_CACHE/$LAYOUT_GRAPH"
test -n "$(find "$LAYOUT_CACHE/$LAYOUT_GRAPH" -type f -print -quit)"
printf 'LAYOUT_GRAPH=%s\n' "$LAYOUT_GRAPH"
```

Require the exact current five full-vision graphs and packed cross-KV graph:

```sh
HASHES="$REPO/tmp/12_unirec_0_1b_inference/current_310p_prefill_hashes_$COMMIT_SHORT.txt"
mkdir -p "$(dirname "$HASHES")"
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
  test -d "$RECOGNITION_CACHE/$graph"
  test -n "$(find "$RECOGNITION_CACHE/$graph" -type f -print -quit)"
done
```

Run the six CPU compatibility tests:

```sh
OUT="$REPO/tmp/12_unirec_0_1b_inference/310p_compiled_layout_prefill_128_w1_w8_$COMMIT_SHORT"
test ! -e "$OUT"
mkdir -p "$OUT"
PYTHONPYCACHEPREFIX="$OUT/pycache" \
  "$PYTHON_BIN" -m unittest \
  "$REPO/12_unirec_0_1b_inference/test_layout_npu_compat.py" \
  2>&1 | tee "$OUT/cpu_tests.log"
```

## 3. Shared producer arguments

```sh
common_args=(
  "$REPO/12_unirec_0_1b_inference/run_prefill_export.py"
  --openocr-root "$OPENOCR_ROOT"
  --model-path "$MODEL"
  --layout-model "$LAYOUT_MODEL"
  --input "$IMAGES_DIR"
  --dtype float16
  --offset 0
  --limit 128
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
export PYTHONUNBUFFERED=1
set -o pipefail
```

## 4. First-128 W1/T16

```sh
W1="$OUT/w1_t16"
mkdir -p "$W1/output"
w1_command=(
  "$PYTHON_BIN" "${common_args[@]}"
  --output-dir "$W1/output"
  --workers 1
  --warmup-pages 4
  --recognition-preprocess-threads 16
)
printf '%q ' "${w1_command[@]}" >"$W1/command.txt"
printf '\n' >>"$W1/command.txt"
SECONDS=0
"${w1_command[@]}" 2>&1 | tee "$W1/run.log"
W1_STATUS="${PIPESTATUS[0]}"
printf '%s\n' "$W1_STATUS" >"$W1/exit_code.txt"
printf '%s\n' "$SECONDS" >"$W1/command_wall_s.txt"
test "$W1_STATUS" = 0
test -f "$W1/output/summary.json"
```

Before starting W8, require a valid W1 result:

```sh
W1_SUMMARY="$W1/output/summary.json" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

summary = json.loads(Path(os.environ["W1_SUMMARY"]).read_text())
assert summary["status"] == "ok"
assert summary["offset"] == 0 and summary["limit"] == 128
assert summary["workers"] == 1
assert summary["recognition_preprocess_threads"] == 16
assert summary["layout_execution"] == "torchair"
assert summary["layout_batch_size"] == 1
assert summary["validation"]["passed"] is True
assert summary["artifact"]["page_count"] == 128
assert summary["artifact"]["crop_count"] > 0
assert summary["worker_summary"]["prefix_diagnostics"]["new_first_call_count"] == 0
assert summary["worker_summary"]["worker_page_counts"] == [128]
assert len(summary["worker_setup_diagnostics"]) == 1
assert summary["worker_setup_diagnostics"][0]["prefix_graph_warmup"]["shape_count"] == 5
print("UNIREC_310P_W1_GATE: PASS")
PY
```

## 5. First-128 W8/T8

```sh
W8="$OUT/w8_t8"
mkdir -p "$W8/output"
w8_command=(
  "$PYTHON_BIN" "${common_args[@]}"
  --output-dir "$W8/output"
  --workers 8
  --warmup-pages 8
  --recognition-preprocess-threads 8
)
printf '%q ' "${w8_command[@]}" >"$W8/command.txt"
printf '\n' >>"$W8/command.txt"
SECONDS=0
set +e
"${w8_command[@]}" 2>&1 | tee "$W8/run.log"
W8_STATUS="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$W8_STATUS" >"$W8/exit_code.txt"
printf '%s\n' "$SECONDS" >"$W8/command_wall_s.txt"
if test "$W8_STATUS" != 0; then
  if grep -Eqi 'out of memory|OOM|memory allocation' "$W8/run.log"; then
    CLASSIFICATION=OOM_W8
  elif grep -q 'IndexByTensor' "$W8/run.log"; then
    CLASSIFICATION=FAIL_INDEX_BY_TENSOR_W8
  else
    CLASSIFICATION=FAIL_INTEGRATION_W8
  fi
  printf 'UNIREC_310P_COMPILED_LAYOUT_PREFILL_128: %s status=%s\n' \
    "$CLASSIFICATION" "$W8_STATUS"
  exit "$W8_STATUS"
fi
test -f "$W8/output/summary.json"
```

Graph-cache loading and eight per-worker graph warmups can take much longer
than the measured producer. Do not include them in pages/s. Do not classify
quiet cache loading as a hang while the owned process or selected NPU remains
active.

## 6. Validate, compare, and print the only required result

```sh
W1_SUMMARY="$W1/output/summary.json" \
W8_SUMMARY="$W8/output/summary.json" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

REFERENCE_910B_PG_S = 27.3511137111
REFERENCE_910B_TOK_S = 13145.6290274
PRIOR_310P_EAGER_W1_PG_S = 2.79
PRIOR_310P_EAGER_W8_PG_S = 5.11

w1 = json.loads(Path(os.environ["W1_SUMMARY"]).read_text())
w8 = json.loads(Path(os.environ["W8_SUMMARY"]).read_text())

def validate(summary, workers, threads):
    assert summary["status"] == "ok"
    assert summary["offset"] == 0 and summary["limit"] == 128
    assert summary["workers"] == workers
    assert summary["recognition_preprocess_threads"] == threads
    assert summary["artifact_storage"] == "discard"
    assert summary["cross_cache_length"] == 512
    assert summary["layout_execution"] == "torchair"
    assert summary["layout_batch_size"] == 1
    assert summary["vision_full_batches"] is True
    assert summary["recognition_input_contract"] == "compact_uint8_hwc"
    assert summary["validation"]["passed"] is True
    assert summary["artifact"]["page_count"] == 128
    assert summary["artifact"]["crop_count"] > 0
    worker = summary["worker_summary"]
    assert worker["worker_count"] == workers
    assert len(worker["worker_page_counts"]) == workers
    assert all(count > 0 for count in worker["worker_page_counts"])
    assert worker["prefix_diagnostics"]["new_first_call_count"] == 0
    setup = summary["worker_setup_diagnostics"]
    assert len(setup) == workers
    assert all(
        row["recognition_preprocess_threads"] == threads
        and row["layout_batch_size"] == 1
        and row["prefix_graph_warmup"]["shape_count"] == 5
        for row in setup
    )

def metrics(summary):
    worker = summary["worker_summary"]
    stages = worker["stage_s"]
    vision = worker["vision_batching"]
    artifact = summary["artifact"]
    return {
        "producer": float(summary["producer_wall_s"]),
        "pg_s": float(summary["throughput"]["pages_per_s"]),
        "tok_s": float(summary["throughput"]["real_source_tokens_per_s"]),
        "crops": int(artifact["crop_count"]),
        "rejected": int(artifact["rejected_crop_count"]),
        "real_tokens": int(artifact["real_source_tokens"]),
        "physical_tokens": int(artifact["physical_source_tokens"]),
        "real_rows": int(vision["compiled_real_rows"]),
        "physical_rows": int(vision["compiled_physical_rows"]),
        "slot_eff": float(vision["compiled_slot_efficiency"]),
        "fallback": int(vision["fallback_rows"]),
        "layout": float(stages["worker_detector_call_sum_s"]),
        "cpu_crop": float(stages["worker_recognition_input_prepare_sum_s"]),
        "npu_prefill": float(stages["worker_recognition_prefill_sum_s"]),
        "d2h": float(stages["worker_recognition_prefill_cache_d2h_sum_s"]),
        "pack": float(stages["worker_shared_pack_sum_s"]),
        "ipc": float(worker["ipc_delivery_sum_s"]),
        "file_io": float(stages["worker_file_read_sum_s"])
        + float(stages["worker_direct_rgb_decode_sum_s"]),
        "setup": float(summary["setup_s"]),
        "warmup": float(summary["warmup"]["wall_s"]),
        "shutdown": float(summary["shutdown_s"]),
        "total": float(summary["total_wall_s"]),
        "peak_worker_hbm_mib": int(vision["max_npu_peak_memory_bytes"]) / 2**20,
    }

validate(w1, 1, 16)
validate(w8, 8, 8)
a = metrics(w1)
b = metrics(w8)
assert a["crops"] == b["crops"]
assert a["rejected"] == b["rejected"]
assert a["real_tokens"] == b["real_tokens"]

print(
    "UNIREC_310P_COMPILED_LAYOUT_PREFILL_128: PASS — "
    f"W1T16 producer={a['producer']:.3f}s pg_s={a['pg_s']:.3f} "
    f"tok_s={a['tok_s']:.1f} layout={a['layout']:.3f}s "
    f"cpu_crop={a['cpu_crop']:.3f}s npu_prefill={a['npu_prefill']:.3f}s "
    f"d2h={a['d2h']:.3f}s pack={a['pack']:.3f}s ipc={a['ipc']:.3f}s "
    f"file_io={a['file_io']:.3f}s setup={a['setup']:.3f}s "
    f"warmup={a['warmup']:.3f}s hbm={a['peak_worker_hbm_mib']:.0f}MiB; "
    f"W8T8 producer={b['producer']:.3f}s pg_s={b['pg_s']:.3f} "
    f"tok_s={b['tok_s']:.1f} layout={b['layout']:.3f}s "
    f"cpu_crop={b['cpu_crop']:.3f}s npu_prefill={b['npu_prefill']:.3f}s "
    f"d2h={b['d2h']:.3f}s pack={b['pack']:.3f}s ipc={b['ipc']:.3f}s "
    f"file_io={b['file_io']:.3f}s setup={b['setup']:.3f}s "
    f"warmup={b['warmup']:.3f}s hbm={b['peak_worker_hbm_mib']:.0f}MiB; "
    f"crops={b['crops']} real_tokens={b['real_tokens']} "
    f"physical_tokens_w1={a['physical_tokens']} "
    f"physical_tokens_w8={b['physical_tokens']} "
    f"vision_rows_w1={a['real_rows']}/{a['physical_rows']} "
    f"vision_rows_w8={b['real_rows']}/{b['physical_rows']} "
    f"slot_eff_w1={a['slot_eff']:.3f} slot_eff_w8={b['slot_eff']:.3f} "
    f"fallback_w1={a['fallback']} fallback_w8={b['fallback']} "
    f"rejected={b['rejected']} "
    f"w8_over_w1={b['pg_s'] / a['pg_s']:.3f}x "
    f"w1_vs_prior_eager16={a['pg_s'] / PRIOR_310P_EAGER_W1_PG_S:.3f}x "
    f"w8_vs_prior_eager128={b['pg_s'] / PRIOR_310P_EAGER_W8_PG_S:.3f}x "
    f"w8_vs_910b_pg={b['pg_s'] / REFERENCE_910B_PG_S:.3f}x "
    f"w8_vs_910b_tok={b['tok_s'] / REFERENCE_910B_TOK_S:.3f}x"
)
PY
```

Send Luka only the final `UNIREC_310P_COMPILED_LAYOUT_PREFILL_128` line. Then
stop. Do not start decode or a larger run.
