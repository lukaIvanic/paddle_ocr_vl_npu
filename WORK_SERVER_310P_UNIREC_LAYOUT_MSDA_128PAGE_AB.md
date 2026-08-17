# 310P UniRec native-MSDA 128-page adoption gate

The one-page descriptor-bridge probe passed. Now run one faithful 128-page A/B
and two one-step NPU profiles:

- baseline: current optimized compiled layout with decomposed 18× GridSample;
- candidate: identical configuration with six native MSDA calls through the
  descriptor bridge.

Do not change the production default during this task.

## Restrictions

- Pull only. Do not edit tracked files, commit, branch, or push.
- Never use physical NPU 5 or 6.
- Use the existing real `python_nosym`; do not canonicalize it with
  `readlink -f`.
- Reuse the already-passing MSDA extension SO. Do not rebuild it.
- Use fresh, separate baseline and candidate graph caches created by the
  runner. Never reuse a failed or one-page cache.
- Keep `MSDA_HOST_OPP_MODE=none`.
- Use exactly one CPU thread as enforced by the runner.
- Run only this A/B. No recognition, decode, full prefill, or evaluation.
- Start it in the background and immediately give Luka the absolute log path
  and `tail -f` command.

## Launch

Source CANN before enabling shell nounset.

```bash
set -eo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main

source npu-setup
set -u
case ",${ASCEND_RT_VISIBLE_DEVICES:-}," in
  *,5,*|*,6,*) echo "REJECTED_PHYSICAL_DEVICE_5_OR_6" >&2; exit 1 ;;
esac

export PYTHON_BIN="${PYTHON_BIN:?set the existing venv python_nosym executable}"
test -x "$PYTHON_BIN"
test "$(basename "$PYTHON_BIN")" = python_nosym

export MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
export LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
export OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
export IMAGES_DIR="${IMAGES_DIR:?set the existing OmniDocBench images directory}"
export MSDA_EXTENSION_SO="${MSDA_EXTENSION_SO:?set the successful binding-probe extension SO}"
test -f "$MSDA_EXTENSION_SO"

export MSDA_REBUILD_EXTENSION=0
export MSDA_RUN_MODE=full_ab
export MSDA_FORWARD_LIMIT=128
export MSDA_WARMUP_PAGES=2
export MSDA_HOST_OPP_MODE=none

bash 12_unirec_0_1b_inference/run_310p_layout_msda_real_background.sh
```

Follow only the owned PID until `exit_code.txt` appears. Progress is visible as
`LAYOUT_LAB page=N/128` plus explicit phase begin/end markers.

## Execution order

The runner performs exactly:

1. baseline 128-page compiled forward;
2. candidate 128-page compiled forward;
3. baseline one-step NPU profile after two warmups and 20 clean repeats;
4. candidate profile with the same settings;
5. structural, geometric, timing, and profile comparison.

## Adoption gates

Report all results even if a gate fails. Do not make the candidate default.

Required structural evidence:

- candidate rewrites exactly six of six MSDA modules;
- 18 baseline GridSample calls become zero candidate GridSample calls;
- candidate profile contains exactly six
  `MultiScaleDeformableAttnFunction` calls;
- all 128 pages retain box count;
- mean paired IoU is at least 0.99;
- report every label or reading-order mismatch and the ten worst-IoU pages.

Required timing evidence:

- baseline → candidate forward mean, median, p90, min, and max;
- forward saved milliseconds and speedup;
- baseline → candidate page-wall mean, median, p90, pages/s, and layout-section
  speedup;
- clean profiled device-event milliseconds and speedup;
- total kernel count and compute milliseconds;
- count and total milliseconds for GridSample, native MSDA, Transpose, Cast,
  ReduceProdD, and Cumsum.

## Required report

Return:

- commit, physical NPU, Python, torch, torch-npu, and CANN;
- absolute run root, run log, both cache roots, both forward JSON files, both
  profile summaries, comparison JSON, and exit-code file;
- every `UNIREC_LAYOUT_MSDA_REAL_PHASE_END` line;
- the complete `UNIREC_LAYOUT_MSDA_REAL_AB` line;
- the complete `UNIREC_LAYOUT_MSDA_REAL_PROFILE` line;
- the `gates`, `quality_review_required`, full timing fields, selected-op
  profile fields, and worst-IoU pages from `comparison_summary.json`;
- confirmation that no host OPP was loaded and the owned NPU was released.

Then stop. Do not run production prefill or change defaults.
