# 310P UniRec K10 height A/B

## Question

For the exact 310P mismatch crop `page_000032_crop_0002`, does changing only
the compiled canvas height from 448 to 512 remove the vision-output drift?

This is intentionally a two-crop, two-graph probe. Do not run decode, prefill,
profiling, or a bucket sweep.

The second crop, `page_000032_crop_0004`, is from the same page and has the
same processed `640x320` shape, but was not a K10 decode mismatch.

The 910B2 control already passed at commit `cc6a5ea`. Its exact JSON is:

`12_unirec_0_1b_inference/references/unirec_910b_k10_height_ab_cc6a5ea/report.json`

On 910B2, the suspect crop's `448_vs_512` comparison had:

- max abs: `0.0009765625`
- RMSE: `6.0023e-06`
- cosine: `0.99999994`

The control crop was essentially identical. The 448 graph was slightly faster
than 512: aggregate p50 `10.63 ms` versus `11.04 ms`.

## Constraints

- Pull commit `cc6a5ea` or newer from `main`.
- Use one free physical 310P device in `0,1,2,3`.
- There is no `npu-setup` on this server. Reuse the same CANN/torch-npu shell
  setup that passed the completed factorization run.
- Use the validated venv's real `python_nosym` executable. Do **not** pass the
  venv symlink through `readlink -f`; that previously escaped the venv and
  broke `kornia_rs`.
- Reuse the exact UniRec model, OpenOCR checkout, and `COMPILE_CACHE` from the
  completed factorization run.
- Do not edit tracked files, commit, push, create a branch, or create a
  worktree.
- A missing cache may compile one target graph. That is acceptable. Record the
  OM inventory before and after instead of aborting.

## Run

Resolve the checkout and export the already validated paths:

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"

: "${PYTHON_BIN:?real python_nosym from the validated venv}"
: "${MODEL:?validated unirec-0.1b directory}"
: "${OPENOCR_ROOT:?validated OpenOCR checkout}"
: "${COMPILE_CACHE:?same compile-cache parent as factorization}"
: "${FACTOR_ROOT:?completed factorization RUN_ROOT}"
: "${ASCEND_RT_VISIBLE_DEVICES:?one free physical 310P device 0-3}"

case "$ASCEND_RT_VISIBLE_DEVICES" in 0|1|2|3) ;; *) exit 2 ;; esac
test -x "$PYTHON_BIN"
test "$(basename "$PYTHON_BIN")" = python_nosym
test -f "$MODEL/model.pth"
test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"

ARTIFACT="$FACTOR_ROOT/prefill_production_buckets_optimized_weights"
test -s "$ARTIFACT/pages.jsonl"
test -s "$ARTIFACT/crops.jsonl"

RUN_ROOT="$WORK_SERVER_REPO/tmp/12_unirec_0_1b_inference/310p_k10_height_ab_cc6a5ea_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$RUN_ROOT"

find "$COMPILE_CACHE" -type f \( -name '*.om' -o -name compiled_module \) \
  | rg 'vision_full_bucket_960x(448|512)_b1_' \
  | sort >"$RUN_ROOT/cache_before.txt" || true

cat >"$RUN_ROOT/run_worker.sh" <<'EOF'
#!/usr/bin/env bash
set +e
started="$(date +%s)"
"$PYTHON_BIN" \
  12_unirec_0_1b_inference/probe_vision_k10_height_ab.py \
  --openocr-root "$OPENOCR_ROOT" \
  --model-path "$MODEL" \
  --page-manifest "$ARTIFACT/pages.jsonl" \
  --crop-manifest "$ARTIFACT/crops.jsonl" \
  --cache-dir "$COMPILE_CACHE" \
  --output "$RUN_ROOT/report.json" \
  --warmup-replays 1 \
  --timing-repeats 3
status=$?
printf '%s\n' "$status" >"$RUN_ROOT/exit_code.txt"
printf '%s\n' "$(($(date +%s) - started))" >"$RUN_ROOT/process_wall_s.txt"
exit "$status"
EOF
chmod +x "$RUN_ROOT/run_worker.sh"
export PYTHON_BIN MODEL OPENOCR_ROOT COMPILE_CACHE ARTIFACT RUN_ROOT
nohup "$RUN_ROOT/run_worker.sh" >"$RUN_ROOT/run.log" 2>&1 &
printf '%s\n' "$!" >"$RUN_ROOT/pid.txt"

echo "RUN_ROOT=$RUN_ROOT"
echo "tail -f '$RUN_ROOT/run.log'"
```

After the process exits:

```bash
find "$COMPILE_CACHE" -type f \( -name '*.om' -o -name compiled_module \) \
  | rg 'vision_full_bucket_960x(448|512)_b1_' \
  | sort >"$RUN_ROOT/cache_after.txt" || true
diff -u "$RUN_ROOT/cache_before.txt" "$RUN_ROOT/cache_after.txt" \
  >"$RUN_ROOT/cache.diff" || true

cat "$RUN_ROOT/exit_code.txt"
cat "$RUN_ROOT/process_wall_s.txt"
jq '{status,physical_devices,probe,rows,aggregate_timing_p50_ms,cache_inventory}' \
  "$RUN_ROOT/report.json"
cat "$RUN_ROOT/cache.diff"
```

## Report

Return only:

1. `RUN_ROOT`, commit, device, exit code, and process wall time.
2. Whether either target OM or `compiled_module` inventory changed.
3. For each crop, the max abs, mean abs, RMSE, and cosine for:
   - `448_vs_eager`
   - `512_vs_eager`
   - `448_vs_512`
4. The three p50 times: eager, 448, and 512.
5. A factual one-line result:
   - 512 materially fixes only the suspect;
   - both heights are equivalent;
   - or both compiled heights differ materially from eager.

Do not infer a production fix until these exact numbers are available.
