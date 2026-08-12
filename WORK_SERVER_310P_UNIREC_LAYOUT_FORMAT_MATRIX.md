# 310P UniRec FP16 layout weight-format matrix

Pull `main` at or after `e9fd39b`. Do not edit tracked files. Use one free full
310P and never physical device 5. Reuse the exact FP16 first-128 model, images,
OpenOCR checkout, and runtime from the successful 40 ms FP16 layout result.

The remaining target is the two depthwise-5x5 filter repacks:

- `FRACTAL_Z -> FRACTAL_Z:192`, 18 calls, about 3.5 ms;
- `FRACTAL_Z -> FRACTAL_Z:384`, 6 calls, about 2.28 ms.

## Restrictions

- Give every lane a new cache root. Never reuse a native compiled graph with
  reformatted weights. A 910B replay did that as a diagnostic and produced zero
  boxes on every page.
- Warm up twice and exclude warmup/compile time.
- First run eight pages. Continue to 128 only if boxes are nonzero and not the
  same result on every page.
- Profile only a valid 128-page winner.
- Preserve JSON and logs. Return only the four short lines requested below.

## Matrix

Set the already-known paths:

```sh
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main
PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
IMAGES_DIR="${IMAGES_DIR:?set the existing OmniDocBench image directory}"
OUT="$REPO/tmp/12_unirec_0_1b_inference/310p_layout_format_$(git rev-parse --short HEAD)"
test ! -e "$OUT"
mkdir -p "$OUT"
```

Run these lanes in order:

```text
native                 --weight-format native
depthwise_fz           --weight-format depthwise_fz
frozen                 --weight-format native --freeze-parameters
frozen_depthwise_fz    --weight-format depthwise_fz --freeze-parameters
```

For each lane, first run this command with `LIMIT=8`, then with `LIMIT=128` only
after the validity gate passes:

```sh
NAME=depthwise_fz
EXTRA=(--weight-format depthwise_fz)
LIMIT=8
CACHE="$REPO/.runtime_cache/12_unirec_0_1b_inference/310p_${NAME}_$(git rev-parse --short HEAD)_limit${LIMIT}"
test ! -e "$CACHE"
"$PYTHON_BIN" "$REPO/12_unirec_0_1b_inference/layout_detector_lab.py" \
  --openocr-root "$OPENOCR_ROOT" \
  --model-path "$LAYOUT_MODEL" \
  --input "$IMAGES_DIR" \
  --output "$OUT/${NAME}_${LIMIT}.json" \
  --device npu:0 --execution torchair --compile-cache-dir "$CACHE" \
  --dtype float16 --threshold 0.4 --offset 0 --limit "$LIMIT" \
  --warmup-pages 2 "${EXTRA[@]}" 2>&1 | tee "$OUT/${NAME}_${LIMIT}.log"
```

Change `NAME`, `EXTRA`, and `LIMIT` mechanically for the other lanes. For
`frozen`, use `EXTRA=(--weight-format native --freeze-parameters)`. For the
combined lane, use
`EXTRA=(--weight-format depthwise_fz --freeze-parameters)`.

The JSON must show the requested format, the before/after format histograms,
and the count of explicitly converted depthwise weights. Compare valid lanes by
`summary.stages.model_forward_s.mean_ms`, median, p90, and total.

For the fastest valid lane, repeat the existing one-replay pipe profiler and
report total TransData plus the two target signatures above. Do not run the
broad `torchair_internal` lane unless all four focused lanes fail; TorchAir's
helper changes every Linear and ordinary Conv2d while explicitly skipping the
target depthwise convolutions.

## Return exactly

```text
Validity: native=<pass/fail>; depthwise_fz=<pass/fail>; frozen=<pass/fail>; frozen_depthwise_fz=<pass/fail>.
Forward: native=<ms>; depthwise_fz=<ms>; frozen=<ms>; frozen_depthwise_fz=<ms>; winner=<name and speedup>.
Formats: <after histograms and depthwise converted count for the winner>.
Winner profile: TransData=<ms/count>; FZ192=<ms/count>; FZ384=<ms/count>.
```

910B context only: native FP16 was 21.979 ms mean. Explicit depthwise
FRACTAL_Z is valid for an isolated `groups=192`, 5x5 FP16 Conv2d, changing NPU
format 0 to 4. Fresh full-graph experimental compiles on the current 910B lane
were blocked by an unrelated `IndexByTensor/Index` AICPU failure. Replaying
depthwise-FZ weights through a copied native graph cache was invalid and
produced zero boxes, so it is not a performance result.
