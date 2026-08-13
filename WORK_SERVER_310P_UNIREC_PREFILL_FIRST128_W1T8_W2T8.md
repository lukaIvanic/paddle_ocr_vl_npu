# 310P UniRec optimized first-128 prefill: W1/T8 and W2/T8

Pull the commit containing this brief and run only this task. Measure the current
optimized page-to-cross-KV producer on the first 128 sorted OmniDocBench pages:

1. one process worker with eight recognition-preprocess threads (`W1/T8`);
2. two process workers, each with eight recognition-preprocess threads (`W2/T8`).

Run W1 first, then W2. Each lane processes all 128 pages once as an excluded
in-process warmup and then processes the same 128 pages as the measured pass.
Do not run decode, page assembly, evaluation, W4/W8, or the full benchmark.

## Exact production contract

- compiled FP16 PP-DocLayoutV2, B1 per process;
- current index-free layout source, `group16`, `torchair_internal` weights, and
  preformatted FrozenBN buffers;
- five compiled FP16 full-vision buckets;
- `constant_grouped_all` focal rewrite and `torchair_internal` vision weights;
- kornia-rs bicubic recognition resize;
- compact `uint8 HWC` crop transfer and NPU normalization;
- packed S1024 text/cross-KV prefill, cross-KV capacity 512;
- page lookahead 4, discarded artifacts, and no retained shared images.

Do **not** add `--profile-prefill-device-stages`. NPU timing events previously
failed in a forked 310P producer. Existing CPU/wall stage timers are sufficient.
Do not copy a 910B cache to 310P.

## Restrictions

- Pull only. Do not edit tracked files, branch, commit, or push.
- Use one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Do not terminate another user's process.
- Use the exact model, dataset, OpenOCR, and cache parents from the latest
  successful optimized UniRec prefill on this server. Do not search for ONNX.
- Launch the matrix in the background and immediately give Luka the absolute
  log path and a `tail -f` command.
- Setup/cache load and the complete first-128 warmup are excluded from
  `producer_wall_s`. Report them separately.

## 1. Pull and resolve the passed environment

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git status --short --branch
git pull --ff-only origin main
git status --short --branch

source npu-setup
test -n "${ASCEND_RT_VISIBLE_DEVICES:-}"
case ",${ASCEND_RT_VISIBLE_DEVICES}," in
  *,5,*|*,6,*) printf 'REJECTED_PHYSICAL_DEVICE_5_OR_6\n' >&2; exit 1 ;;
esac
case "$ASCEND_RT_VISIBLE_DEVICES" in
  *,*) printf 'REQUIRES_EXACTLY_ONE_VISIBLE_310P=%s\n' \
         "$ASCEND_RT_VISIBLE_DEVICES" >&2; exit 1 ;;
esac

export PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
export MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
export LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
export OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
export IMAGES_DIR="${IMAGES_DIR:?reuse the existing OmniDocBench images directory}"
export LAYOUT_CACHE="${LAYOUT_CACHE:?reuse the latest passed optimized-layout cache parent}"
export RECOGNITION_CACHE="${RECOGNITION_CACHE:?reuse the latest passed constant_grouped_all plus torchair_internal cache parent}"

PYTHON_BIN="$(readlink -f "$PYTHON_BIN")"
MODEL="$(readlink -f "$MODEL")"
LAYOUT_MODEL="$(readlink -f "$LAYOUT_MODEL")"
OPENOCR_ROOT="$(readlink -f "$OPENOCR_ROOT")"
IMAGES_DIR="$(readlink -f "$IMAGES_DIR")"
LAYOUT_CACHE="$(readlink -f "$LAYOUT_CACHE")"
RECOGNITION_CACHE="$(readlink -f "$RECOGNITION_CACHE")"
export PYTHON_BIN MODEL LAYOUT_MODEL OPENOCR_ROOT IMAGES_DIR
export LAYOUT_CACHE RECOGNITION_CACHE

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -d "$LAYOUT_MODEL"
test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
test -d "$IMAGES_DIR"
test -d "$LAYOUT_CACHE"
test -d "$RECOGNITION_CACHE"

COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
OUT="$REPO/tmp/12_unirec_0_1b_inference/310p_prefill_first128_w1t8_w2t8_${COMMIT_SHORT}_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$OUT"
printf 'commit=%s\nphysical_npu=%s\n' "$COMMIT" \
  "$ASCEND_RT_VISIBLE_DEVICES" >"$OUT/preflight.txt"
"$PYTHON_BIN" -c \
  'import torch, torch_npu, kornia_rs; print("torch="+torch.__version__); print("torch_npu="+torch_npu.__version__); print("kornia_rs="+getattr(kornia_rs,"__version__","unknown"))' \
  >>"$OUT/preflight.txt"
npu-smi info >>"$OUT/preflight.txt" 2>&1
```

If the cache variable names are not already known, recover them from the exact
latest successful optimized UniRec command/report on this server. Do not select
the older native-weight all-45 cache. If the current source creates a new cache
child under either passed parent, that is valid; report the setup time and child
path. Do not silently change the requested optimization lane.

## 2. Write and launch the background matrix

The following creates only an untracked run script inside `tmp/`:

```bash
MATRIX="$OUT/run_matrix.sh"
cat >"$MATRIX" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
RUNNER="$REPO/12_unirec_0_1b_inference/run_prefill_export.py"
: "${OUT:?}" "${PYTHON_BIN:?}" "${MODEL:?}" "${LAYOUT_MODEL:?}"
: "${OPENOCR_ROOT:?}" "${IMAGES_DIR:?}" "${LAYOUT_CACHE:?}"
: "${RECOGNITION_CACHE:?}" "${ASCEND_RT_VISIBLE_DEVICES:?}"

common=(
  "$PYTHON_BIN" "$RUNNER"
  --openocr-root "$OPENOCR_ROOT"
  --model-path "$MODEL"
  --layout-model "$LAYOUT_MODEL"
  --input "$IMAGES_DIR"
  --artifact-storage discard
  --offset 0
  --limit 128
  --warmup-pages 128
  --warmup-repeats 1
  --layout-execution torchair
  --layout-dtype float16
  --layout-batch-size 1
  --layout-depthwise-rewrite group16
  --layout-weight-format torchair_internal
  --layout-preformat-frozen-bn-buffers
  --layout-cache-dir "$LAYOUT_CACHE"
  --dtype float16
  --cross-cache-length 512
  --recognition-cache-dir "$RECOGNITION_CACHE"
  --vision-full-batches
  --vision-focal-depthwise-rewrite constant_grouped_all
  --vision-weight-format torchair_internal
  --recognition-input-contract compact_uint8_hwc
  --recognition-resize-backend kornia_rs
  --vision-page-lookahead 4
  --no-retain-shared-images
  --progress-every-pages 1
  --progress-heartbeat-s 15
)

run_lane() {
  local workers="$1" threads="$2" name="w${1}_t${2}"
  local lane="$OUT/$name"
  mkdir -p "$lane/output"
  local cmd=(
    "${common[@]}"
    --output-dir "$lane/output"
    --workers "$workers"
    --recognition-preprocess-threads "$threads"
  )
  printf 'UNIREC_310P_LANE_BEGIN lane=%s workers=%s threads=%s\n' \
    "$name" "$workers" "$threads"
  printf '%q ' "${cmd[@]}" >"$lane/command.txt"
  printf '\n' >>"$lane/command.txt"
  local started="$SECONDS"
  set +e
  "${cmd[@]}" 2>&1 | tee "$lane/run.log"
  local status="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "$status" >"$lane/exit_code.txt"
  printf '%s\n' "$((SECONDS - started))" >"$lane/process_wall_s.txt"
  printf 'UNIREC_310P_LANE_END lane=%s status=%s process_wall_s=%s\n' \
    "$name" "$status" "$(cat "$lane/process_wall_s.txt")"
  test "$status" = 0
  test -f "$lane/output/summary.json"
}

run_lane 1 8
run_lane 2 8

W1="$OUT/w1_t8/output/summary.json" \
W2="$OUT/w2_t8/output/summary.json" \
  "$PYTHON_BIN" - <<'PY' | tee "$OUT/report.log"
import json
import os
from pathlib import Path

w1 = json.loads(Path(os.environ["W1"]).read_text())
w2 = json.loads(Path(os.environ["W2"]).read_text())

def validate(p, workers):
    assert p["status"] == "ok"
    assert p["validation"]["passed"] is True
    assert (p["offset"], p["limit"], p["workers"]) == (0, 128, workers)
    assert p["artifact_storage"] == "discard"
    assert p["layout_execution"] == "torchair"
    assert p["layout_dtype"] == "float16"
    assert p["layout_batch_size"] == 1
    assert p["vision_full_batches"] is True
    assert p["vision_focal_depthwise_rewrite"] == "constant_grouped_all"
    assert p["vision_weight_format"] == "torchair_internal"
    assert p["recognition_input_contract"] == "compact_uint8_hwc"
    assert p["recognition_resize_backend"] == "kornia_rs"
    assert p["recognition_preprocess_threads"] == 8
    assert p["cross_cache_length"] == 512
    assert p["artifact"]["page_count"] == 128
    ws = p["worker_summary"]
    assert ws["worker_count"] == workers
    assert len(ws["worker_page_counts"]) == workers
    assert all(count > 0 for count in ws["worker_page_counts"])
    assert ws["prefix_diagnostics"]["new_first_call_count"] == 0
    assert len(p["worker_setup_diagnostics"]) == workers
    assert all(
        row["recognition_preprocess_threads"] == 8
        and row["layout_batch_size"] == 1
        and row["prefix_graph_warmup"]["shape_count"] == 5
        for row in p["worker_setup_diagnostics"]
    )

def metrics(p):
    ws = p["worker_summary"]
    s = ws["stage_s"]
    v = ws["vision_batching"]
    a = p["artifact"]
    return {
        "wall": float(p["producer_wall_s"]),
        "pg_s": float(p["throughput"]["pages_per_s"]),
        "crop_s": float(p["throughput"]["crops_per_s"]),
        "tok_s": float(p["throughput"]["real_source_tokens_per_s"]),
        "pages": int(a["page_count"]),
        "crops": int(a["crop_count"]),
        "rejected": int(a["rejected_crop_count"]),
        "tokens": int(a["real_source_tokens"]),
        "layout": float(s["worker_detector_call_sum_s"]),
        "input_prepare": float(s["worker_recognition_input_prepare_sum_s"]),
        "resize": float(s["worker_recognition_processor_resize_sum_s"]),
        "prefill": float(s["worker_recognition_prefill_sum_s"]),
        "d2h": float(s["worker_recognition_prefill_cache_d2h_sum_s"]),
        "pack": float(s["worker_shared_pack_sum_s"]),
        "ipc": float(ws["ipc_delivery_sum_s"]),
        "rgb_decode": float(s["worker_direct_rgb_decode_sum_s"]),
        "real_rows": int(v["compiled_real_rows"]),
        "physical_rows": int(v["compiled_physical_rows"]),
        "slot_eff": float(v["compiled_slot_efficiency"]),
        "fallback": int(v["fallback_rows"]),
        "hbm_mib": int(v["max_npu_peak_memory_bytes"]) / 2**20,
        "setup": float(p["setup_s"]),
        "warmup": float(p["warmup"]["wall_s"]),
        "shutdown": float(p["shutdown_s"]),
        "worker_pages": ws["worker_page_counts"],
        "worker_busy": ws["worker_busy_s"],
    }

validate(w1, 1)
validate(w2, 2)
a, b = metrics(w1), metrics(w2)
for key in ("pages", "crops", "rejected", "tokens"):
    assert a[key] == b[key], (key, a[key], b[key])

def line(name, x):
    return (
        f"UNIREC_310P_FIRST128_{name}: PASS "
        f"producer={x['wall']:.6f}s pages_s={x['pg_s']:.6f} "
        f"crops_s={x['crop_s']:.6f} tokens_s={x['tok_s']:.3f} "
        f"pages={x['pages']} crops={x['crops']} rejected={x['rejected']} "
        f"tokens={x['tokens']} layout_work={x['layout']:.6f}s "
        f"input_prepare_work={x['input_prepare']:.6f}s "
        f"resize_work={x['resize']:.6f}s prefill_work={x['prefill']:.6f}s "
        f"d2h_work={x['d2h']:.6f}s pack_work={x['pack']:.6f}s "
        f"ipc={x['ipc']:.6f}s rgb_decode_work={x['rgb_decode']:.6f}s "
        f"vision_rows={x['real_rows']}/{x['physical_rows']} "
        f"slot_eff={x['slot_eff']:.9f} fallback={x['fallback']} "
        f"peak_worker_hbm={x['hbm_mib']:.1f}MiB setup={x['setup']:.3f}s "
        f"warmup={x['warmup']:.3f}s shutdown={x['shutdown']:.3f}s "
        f"worker_pages={x['worker_pages']} worker_busy={x['worker_busy']}"
    )

print(line("W1T8", a))
print(line("W2T8", b))
print(
    "UNIREC_310P_FIRST128_W1T8_W2T8: PASS "
    f"w2_over_w1_pages_s={b['pg_s'] / a['pg_s']:.6f}x "
    f"w2_wall_over_w1_wall={b['wall'] / a['wall']:.6f} "
    f"workload_pages={a['pages']} crops={a['crops']} "
    f"rejected={a['rejected']} tokens={a['tokens']}"
)
PY
BASH
chmod +x "$MATRIX"

export OUT ASCEND_RT_VISIBLE_DEVICES
nohup bash "$MATRIX" >"$OUT/run.log" 2>&1 < /dev/null &
PID="$!"
printf '%s\n' "$PID" >"$OUT/pid.txt"
sleep 1
kill -0 "$PID"
printf '310P FIRST128 W1T8+W2T8 STARTED - pid=%s; run_log=%s; tail_command=tail -f %q\n' \
  "$PID" "$OUT/run.log" "$OUT/run.log"
```

Immediately send Luka the printed start line. Then inspect only this owned job:

```bash
tail -f "$OUT/run.log"
```

Every completed page is printed. During setup, cache discovery, or graph load,
the 15-second heartbeat and the owned PID distinguish progress from failure.
W2 uses two producer processes on the same one visible physical 310P; each sees
that device as logical `npu:0`.

## 3. Completion report

```bash
wait "$(cat "$OUT/pid.txt")" || true
cat "$OUT/preflight.txt"
cat "$OUT/report.log"
printf 'RUN_ROOT=%s\nRUN_LOG=%s\n' "$OUT" "$OUT/run.log"
```

Return:

1. commit, physical NPU, CANN, torch, torch_npu, and kornia-rs versions;
2. `UNIREC_310P_FIRST128_W1T8`, `...W2T8`, and the combined comparison line;
3. the absolute run root, matrix log, both `summary.json` paths, and the exact
   layout/recognition cache parents;
4. if a lane fails, its first causal error, last completed page, exit code, and
   whether the owned processes released NPU memory.

The `*_work` fields are sums across workers. For W2 they overlap and must not be
added to reconstruct wall time. The primary comparison is measured
`producer_wall_s` / `pages_per_s`; setup and warmup are reported separately.

Then stop.
