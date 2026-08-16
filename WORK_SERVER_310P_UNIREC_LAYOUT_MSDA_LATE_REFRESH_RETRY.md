# 310P UniRec native-MSDA late-refresh retry

Run one candidate compile probe only. The prior `af90456` run still used
CANN's stock broken shape inference and failed at `Reshape_86` with 64 input
elements versus 76,800 requested elements.

## What changed

The user OPP callback worked on 910B2 but did not win registration order on
310P. CANN 9.1 on 310P registers or refreshes the stock operator after loading
the persistent op-tiling library.

The new OPP keeps the callback implementation in the resident, `NODELETE`
op-tiling DSO. Every later proto-library load calls back into that resident DSO
and re-installs only the infer-shape and infer-dtype pointers. The callback
never points into the unloadable proto DSO. CANN's stock tiler, workspace code,
and device kernel remain unchanged.

A fresh-cache 910B2 control passed with:

```text
UNIREC_LAYOUT_MSDA_HOST_OPP_OVERRIDE_INSTALLED trigger=tiling_initial count=1
UNIREC_LAYOUT_MSDA_HOST_OPP_OVERRIDE_INSTALLED trigger=proto_refresh count=2
UNIREC_LAYOUT_MSDA_HOST_OPP_REFRESH_FROM_PROTO status=0
UNIREC_LAYOUT_MSDA_HOST_OPP_ACTIVE location_dim1=300 output=[1,300,256]
```

Both warmups completed, the one-page digest remained `e0e3510c8327`, and the
measured model forward was 12.55 ms on physical 910B2 NPU 7.

## Restrictions

- Pull only. Do not edit tracked files, commit, branch, or push.
- Never use physical NPU 5 or 6.
- Use the existing real `python_nosym` executable. Do not apply `readlink -f`
  to `PYTHON_BIN`.
- Use a fresh graph cache and rebuild the small torch binding.
- Run only one candidate page with two warmups.
- Do not rerun the decomposed baseline, profiling, recognition, or evaluation.
- If the same reshape error remains, stop after collecting the required
  markers. Do not launch another NPU run.

## First inspect the failed run without using an NPU

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
OLD_RUN="$REPO/tmp/12_unirec_0_1b_inference/310p_layout_msda_real_af90456_20260816T123929"
test -f "$OLD_RUN/run.log"
printf 'OLD_HOST_OPP_LINES\n'
grep -n 'UNIREC_LAYOUT_MSDA_HOST_OPP' "$OLD_RUN/run.log" || true
printf 'OLD_MARKER\n'
cat "$OLD_RUN/host_infer_marker.txt" 2>/dev/null || true
```

Keep this output for the final comparison.

## Launch the corrected candidate

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main

source npu-setup
case ",${ASCEND_RT_VISIBLE_DEVICES:-}," in
  *,5,*|*,6,*) echo "REJECTED_PHYSICAL_DEVICE_5_OR_6" >&2; exit 1 ;;
esac

# Use the same successful binding-probe environment as before.
export PYTHON_BIN="${PYTHON_BIN:?set the existing venv python_nosym executable}"
test -x "$PYTHON_BIN"
test "$(basename "$PYTHON_BIN")" = python_nosym

export MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
export LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
export OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
export IMAGES_DIR="${IMAGES_DIR:?set the existing OmniDocBench images directory}"
export MSDA_RUN_MODE=candidate_compile_probe
export MSDA_FORWARD_LIMIT=1
export MSDA_WARMUP_PAGES=2
export MSDA_REBUILD_EXTENSION=1

bash 12_unirec_0_1b_inference/run_310p_layout_msda_real_background.sh
```

Immediately give Luka the printed absolute `RUN_LOG` and `TAIL_COMMAND`.
Follow only the owned PID until `exit_code.txt` appears.

## Pass gate

All of the following are required:

1. `UNIREC_LAYOUT_MSDA_HOST_OPP_OVERRIDE_INSTALLED` appears first with
   `trigger=tiling_initial` and later with `trigger=proto_refresh`.
2. `UNIREC_LAYOUT_MSDA_HOST_OPP_REFRESH_FROM_PROTO status=0` appears before
   graph inference.
3. `UNIREC_LAYOUT_MSDA_HOST_OPP_ACTIVE` reports exactly
   `location_dim1=1 output=[1,256,300]` on 310P.
4. `host_infer_marker.txt` contains both refresh events and the same active
   output shape.
5. Both warmups and the measured page finish; exit code is zero.
6. The owned process exits and its physical NPU returns to clean idle state.

## Required report

Return:

- old `af90456` host-OPP lines and marker contents;
- new commit, physical NPU, exact Python path, torch, torch-npu, and CANN;
- new host-OPP lines and full marker contents;
- first real GE/TBE error if the candidate fails;
- final `LAYOUT_LAB done` line, forward time, boxes, and digest if it passes;
- absolute run root, run log, cache root, host OPP build log/vendor root,
  extension build log/SO, output JSON, and exit-code file;
- confirmation that the owned NPU was released.

Then stop. Do not change the production default.
