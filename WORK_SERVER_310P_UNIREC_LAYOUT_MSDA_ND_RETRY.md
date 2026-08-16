# 310P UniRec native-MSDA ND-layout retry

Retry only the failed native-MSDA TorchAir candidate. Do not rerun the
decomposed baseline yet.

## Why the prior conclusion is not established

The standalone eager binding passed on 310P. The failed compiled graph showed
`value_spatial_shapes` with logical shape `[3, 2]` but descriptor format and
original format `NCHW`. The installed CANN operator metadata requires `ND` for
all five inputs and the output of
`MultiScaleDeformableAttnFunction`. Therefore the prior run proves only that
the old converter emitted a bad graph descriptor. It does not prove that 310P
cannot compile native MSDA.

The current converter explicitly annotates all five inputs and the output as
`ND`. A fresh 910B2 TorchAir graph compiled and ran with this change. Both
warmups completed and the one-page measured forward was 12.13 ms.

The runner also no longer applies `readlink -f` to `PYTHON_BIN`. That operation
resolved the venv launcher to `/usr/local/.../python3.12` and could detach the
run from the intended venv. Use the existing real `python_nosym` executable
from the successful binding environment. The runner now preserves its path.

## Restrictions

- Pull only. Do not edit, commit, branch, or push.
- Never use physical NPU 5 or 6.
- Reuse the already-built, successful binding extension SO.
- Use a fresh candidate graph cache.
- Run only the one-page candidate compile probe below. Do not rerun the
  baseline, profiling, recognition, or evaluation.
- Do not call this a hard blocker if it still fails. Preserve the complete TBE
  compile log and every input descriptor first.

## Launch

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main

source npu-setup
case ",${ASCEND_RT_VISIBLE_DEVICES:-}," in
  *,5,*|*,6,*) echo "REJECTED_PHYSICAL_DEVICE_5_OR_6" >&2; exit 1 ;;
esac

# Use the exact venv from the successful binding probe. Prefer its real
# python_nosym file and do not canonicalize it with readlink -f.
export PYTHON_BIN="${PYTHON_BIN:?set the existing venv python_nosym executable}"
test -x "$PYTHON_BIN"
test "$(basename "$PYTHON_BIN")" = python_nosym

export MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
export LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
export OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
export IMAGES_DIR="${IMAGES_DIR:?set the existing OmniDocBench images directory}"
export MSDA_BINDING_RUN_ROOT="${MSDA_BINDING_RUN_ROOT:?set the successful binding run root}"
test "$(tr -d '[:space:]' <"$MSDA_BINDING_RUN_ROOT/exit_code.txt")" = 0
export MSDA_EXTENSION_SO
MSDA_EXTENSION_SO="$(tr -d '[:space:]' <"$MSDA_BINDING_RUN_ROOT/extension_so.txt")"
test -f "$MSDA_EXTENSION_SO"

export MSDA_RUN_MODE=candidate_compile_probe
export MSDA_FORWARD_LIMIT=1
export MSDA_WARMUP_PAGES=2
bash 12_unirec_0_1b_inference/run_310p_layout_msda_real_background.sh
```

Immediately give Luka the printed absolute `RUN_LOG` and `TAIL_COMMAND`. Follow
the owned PID until `exit_code.txt` appears.

## Required report

If it passes, return:

1. the final `LAYOUT_LAB done` line and measured forward time;
2. confirmation that both warmup calls completed;
3. physical NPU, commit, exact `python=` path, torch, and torch-npu;
4. absolute run root, cache root, log, output JSON, and exit-code file;
5. confirmation that the owned process exited and its NPU was released.

If it fails, return instead:

1. the first actual TBE/GE error, not only the downstream Python exception;
2. all five MSDA input descriptors and the output descriptor, including
   logical shape, dtype, format, and original format;
3. any `value_spatial_shapes` occurrence, which must now show `ND`, not
   `NCHW`;
4. the exact compile-log directory and failed kernel JSON/path;
5. the same environment and artifact paths listed above.

Then stop. Do not change the production default.
