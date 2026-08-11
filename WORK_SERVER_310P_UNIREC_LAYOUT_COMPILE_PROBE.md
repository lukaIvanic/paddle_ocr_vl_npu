# Work-server 310P focused compiled-layout probe

Run only PP-DocLayoutV2. Do not load UniRec, compile recognition graphs, run
prefill, decode, or process more than one page.

The previous full-pipeline attempt failed inside the compiled layout warmup
with the 310P AICPU `IndexByTensor` kernel. The exact Transformers 5.5.4 source
contained three remaining data-dependent table reads in the compiled forward:

1. per-class threshold selection;
2. class-order remapping;
3. reading-order relative-position bias selection.

The current commit rewrites all three as dtype-preserving embedding lookups.
It also gives `layout_detector_lab.py` a same-process TorchAir mode. A failure
now reports directly from the layout call instead of crossing a worker queue.

## Restrictions

- Pull only. Do not edit tracked files, branch, commit, or push.
- Use one free physical 310P. Never use physical device 5.
- Use eager FP32 as the parity reference and compiled FP32 as the test.
- Use a new empty compile-cache root for this source revision.
- Preserve the complete first traceback and CANN error on failure.
- Do not continue to the production prefill pipeline in this task.
- Print the final result to stdout. Do not create or push an agent report.

## 1. Resolve project-local inputs

Run with Bash from the project checkout:

```sh
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git status --short --branch
git pull --ff-only origin main
git status --short --branch

COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
IMAGES_DIR="${IMAGES_DIR:?set IMAGES_DIR to the existing 1651-page image directory}"
OUT="$REPO/tmp/12_unirec_0_1b_inference/310p_layout_probe_$COMMIT_SHORT"
CACHE="$REPO/.runtime_cache/12_unirec_0_1b_inference/310p_layout_probe_$COMMIT_SHORT"

test -x "$PYTHON_BIN"
test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
test "$(git -C "$OPENOCR_ROOT" rev-parse HEAD)" = \
  0d522801ec6dc1df852c6b6d4ed6a08f5127ed97
test -z "$(git -C "$OPENOCR_ROOT" status --short)"
test -f "$LAYOUT_MODEL/model.safetensors"
test -d "$IMAGES_DIR"
test -n "${ASCEND_RT_VISIBLE_DEVICES:-}"
case ",$ASCEND_RT_VISIBLE_DEVICES," in
  *,5,*) printf 'REJECTED_PHYSICAL_DEVICE_5\n'; exit 1 ;;
esac
test ! -e "$CACHE"
mkdir -p "$OUT" "$CACHE"
```

If the project-local directories have different names, set the corresponding
environment variables. Do not move or redownload passed artifacts merely to
match the defaults above.

Verify the exact layout weights:

```sh
test "$(stat -c %s "$LAYOUT_MODEL/model.safetensors")" = 214798436
printf '%s  %s\n' \
  e60f3725aeedc88fd319416ef166bda79171a41516a301c27cab9132dc2739d2 \
  "$LAYOUT_MODEL/model.safetensors" | sha256sum -c -
```

## 2. Run CPU source/parity gates

```sh
PYTHONPYCACHEPREFIX="$OUT/pycache" \
  "$PYTHON_BIN" -m unittest \
  "$REPO/12_unirec_0_1b_inference/test_layout_npu_compat.py" \
  2>&1 | tee "$OUT/cpu_tests.log"
```

All six tests must pass.

## 3. One-page eager reference

```sh
EAGER_JSON="$OUT/eager.json"
eager_command=(
  "$PYTHON_BIN"
  "$REPO/12_unirec_0_1b_inference/layout_detector_lab.py"
  --openocr-root "$OPENOCR_ROOT"
  --model-path "$LAYOUT_MODEL"
  --input "$IMAGES_DIR"
  --output "$EAGER_JSON"
  --device npu:0
  --dtype float32
  --execution eager
  --compile-cache-dir "$CACHE"
  --threshold 0.4
  --offset 0
  --limit 1
  --warmup-pages 1
)

printf '%q ' "${eager_command[@]}" >"$OUT/eager_command.txt"
printf '\n' >>"$OUT/eager_command.txt"
export PYTHONUNBUFFERED=1
set -o pipefail
"${eager_command[@]}" 2>&1 | tee "$OUT/eager.log"
EAGER_STATUS="${PIPESTATUS[0]}"
printf '%s\n' "$EAGER_STATUS" >"$OUT/eager_exit_code.txt"
test "$EAGER_STATUS" = 0
test -f "$EAGER_JSON"
```

Require `LAYOUT_LAB phase=warmup_call_end`, one measured page, and a nonzero
box count.

## 4. Same-page compiled test

```sh
COMPILED_JSON="$OUT/compiled.json"
compiled_command=(
  "$PYTHON_BIN"
  "$REPO/12_unirec_0_1b_inference/layout_detector_lab.py"
  --openocr-root "$OPENOCR_ROOT"
  --model-path "$LAYOUT_MODEL"
  --input "$IMAGES_DIR"
  --output "$COMPILED_JSON"
  --device npu:0
  --dtype float32
  --execution torchair
  --compile-cache-dir "$CACHE"
  --threshold 0.4
  --offset 0
  --limit 1
  --warmup-pages 2
)

printf '%q ' "${compiled_command[@]}" >"$OUT/compiled_command.txt"
printf '\n' >>"$OUT/compiled_command.txt"
set +e
"${compiled_command[@]}" 2>&1 | tee "$OUT/compiled.log"
COMPILED_STATUS="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$COMPILED_STATUS" >"$OUT/compiled_exit_code.txt"

if test "$COMPILED_STATUS" != 0; then
  LAST_PHASE="$(grep 'LAYOUT_LAB phase=' "$OUT/compiled.log" | tail -n 1)"
  if grep -q 'IndexByTensor' "$OUT/compiled.log"; then
    CLASSIFICATION=FAIL_INDEX_BY_TENSOR
  else
    CLASSIFICATION=FAIL_OTHER
  fi
  printf 'UNIREC_310P_LAYOUT_COMPILE: %s — status=%s last_phase=%q\n' \
    "$CLASSIFICATION" "$COMPILED_STATUS" "$LAST_PHASE"
  exit "$COMPILED_STATUS"
fi
test -f "$COMPILED_JSON"
```

Both compiled warmup calls and the measured call must finish.

## 5. Exact output comparison and stdout result

```sh
EAGER_JSON="$EAGER_JSON" COMPILED_JSON="$COMPILED_JSON" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

eager = json.loads(Path(os.environ["EAGER_JSON"]).read_text())
compiled = json.loads(Path(os.environ["COMPILED_JSON"]).read_text())
eager_page = eager["pages"][0]
compiled_page = compiled["pages"][0]
digest_match = eager_page["result_digest"] == compiled_page["result_digest"]
box_match = eager_page["box_count"] == compiled_page["box_count"]
classification = "PASS_EXACT" if digest_match and box_match else "FAIL_PARITY"
print(
    "UNIREC_310P_LAYOUT_COMPILE: "
    f"{classification} — "
    f"boxes={compiled_page['box_count']} "
    f"digest_match={str(digest_match).lower()} "
    f"eager_forward_ms={eager_page['stage_s']['model_forward_s'] * 1000:.3f} "
    f"compiled_forward_ms={compiled_page['stage_s']['model_forward_s'] * 1000:.3f} "
    f"compiled_pg_s={compiled['summary']['pages_per_s']:.3f}"
)
if classification != "PASS_EXACT":
    raise SystemExit(1)
PY
```

Then stop. Do not run the full prefill pipeline until Luka sends the result
back and explicitly asks for the integration run.
