# 310P UniRec native-MSDA static-metadata validation

Run only the optimized native-MSDA candidate. Reuse the completed decomposed
baseline and previous native candidate artifacts from the prior 128-page A/B.
Do not rerun the unchanged baseline.

## Change under test

PP-DocLayoutV2 always receives a compiled 800x800 layout tensor. Its three
feature levels are therefore fixed:

```text
spatial_shapes = [[100,100], [50,50], [25,25]]
level_products = [10000, 2500, 625]
level_cumsum = [10000, 12500, 13125]
level_start_index = [0, 10000, 12500]
```

The upstream model builds `spatial_shapes` through six scalar indexed writes.
Torch functionalization lowered those writes to 12 `ScatterElements` AICPU
calls after native MSDA made the metadata live. The candidate converter now
emits the exact constants above.

Fresh-cache 910B2 validation at `b66b4c7`:

```text
one-page boxes/digest: exact
ScatterElements: 12 -> 0
Cumsum: 2 -> 1
ReduceProdD: 1 -> 0
BroadcastTo: 37 -> 13
kernel count: 1289 -> 1241
accounted compute: 9.87114 -> 9.36162 ms
device event: 10.65656 -> 10.22187 ms
```

## Restrictions

- Pull only. Do not edit tracked files, commit, branch, or push.
- Never use physical NPU 5 or 6.
- Use the existing real `python_nosym`; do not canonicalize it with
  `readlink -f`.
- Reuse the successful MSDA extension SO. Do not rebuild it.
- Set `MSDA_REFERENCE_RUN_ROOT` to the completed prior 128-page A/B root that
  contains both forward JSON files and both profiles.
- Use a fresh candidate graph cache. Do not reuse the old native candidate
  cache because the cache key does not include this converter source.
- Keep `MSDA_HOST_OPP_MODE=none` and one CPU thread.
- Do not run the unchanged decomposed baseline, recognition, decode, prefill,
  or evaluation.
- Start in the background and immediately give Luka the log path and `tail -f`.

## Launch

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
export MSDA_REFERENCE_RUN_ROOT="${MSDA_REFERENCE_RUN_ROOT:?set the completed prior A/B root}"

export MSDA_REBUILD_EXTENSION=0
export MSDA_RUN_MODE=candidate_against_reference
export MSDA_FORWARD_LIMIT=128
export MSDA_WARMUP_PAGES=2
export MSDA_HOST_OPP_MODE=none

bash 12_unirec_0_1b_inference/run_310p_layout_msda_real_background.sh
```

## Pass gates

- Six static-spatial-shape markers, one products marker, and one cumsum marker.
- Candidate forward completes all 128 pages.
- Candidate profile contains:
  - zero `ScatterElements`;
  - one `Cumsum`;
  - zero `ReduceProdD`;
  - six native MSDA calls;
  - zero GridSample calls.
- `previous_native_output_parity.json` has all 128 page digests exact, no label
  mismatch, no order mismatch, and identical coordinates/scores. This is a
  strict gate because only integer metadata construction changed.
- The decomposed-baseline comparison retains the prior structural gates.
- Owned process exits and NPU is released.

## Required report

Return:

- commit, physical NPU, Python, torch, torch-npu, and CANN;
- absolute new run root/log/cache/profile/comparison/parity paths;
- all phase-end lines;
- static metadata marker counts;
- complete A/B, profile, operator-delta, regression, and savings lines;
- baseline -> new candidate forward mean/median/p90 and device event;
- previous native -> new native forward distribution;
- count/time for ScatterElements, Cumsum, ReduceProdD, BroadcastTo, native
  MSDA, GridSample, Transpose, Cast, and TransData;
- every structural/parity gate and confirmation that the NPU was released.

Then stop. Do not change production defaults.
