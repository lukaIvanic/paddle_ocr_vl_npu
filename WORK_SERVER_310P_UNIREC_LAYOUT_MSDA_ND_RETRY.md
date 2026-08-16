# 310P UniRec native-MSDA internal-layout retry

Retry only the failed native-MSDA TorchAir candidate. Do not rerun the
decomposed baseline yet. This revision follows the second failed probe, which
correctly exposed `numQueries=3` inside the 310P tiler.

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

The second 310P failure is not evidence that the model has three queries. The
model has 300. Huawei's ACLNN wrapper always transforms the logical API tensors
before calling the lower-level graph operator on 310P. Our direct TorchAir
custom node had bypassed those hidden transforms, so the tiler read the three
feature levels as the query dimension.

The converter now reproduces Huawei's 310P ACLNN path exactly:

```text
logical value       [1,13125,8,32]  -> internal [1,8,13125,32]
logical locations   [1,300,8,3,4,2] -> internal [3,1,8,300,4,2]
logical weights     [1,300,8,3,4]   -> internal [3,1,8,300,4]
floating dtype      FP16            -> internal FP32
internal output     [1,256,300]      -> logical [1,300,256] FP16
```

This makes the 310P tiler read `numQueries=300` from internal weight dimension
3, as intended. Do not pad the feature-level dimension from 3 to 32.

The following retry targets the next graph-construction failure. Huawei's
shared GE infer-shape implementation reads the transposed location's dimension
5 (`2`, the coordinate pair) as the query count and produces the bogus internal
shape `[1,2,32]`. The tiler itself uses the correct attention-weight dimensions.
TorchAir's `_inference_rule` attribute did not override the already-registered
CANN host function. The binding extension now registers a corrected host infer
callback for the same operator. It returns internal `[B,H*D,Q]` for the 310P
layout and preserves logical `[B,Q,H*D]` inference for the public layout. This
changes graph construction only and adds no runtime operation.

This is derived from the public Ascend operator sources, not a guessed layout:

- `op_api/aclnn_multi_scale_deformable_attn_function.cpp` applies the 310P
  input transposes, FP32 casts, output transpose, and output cast;
- `op_host/multi_scale_deformable_attn_function_tiling.cpp` reads 310P
  `numQueries` from attention-weight dimension 3 and enforces `>= 32`.

Source tree:
`https://github.com/hicann/ops-nn/tree/master/vfusion/multi_scale_deformable_attn_function`

The runner also no longer applies `readlink -f` to `PYTHON_BIN`. That operation
resolved the venv launcher to `/usr/local/.../python3.12` and could detach the
run from the intended venv. Use the existing real `python_nosym` executable
from the successful binding environment. The runner now preserves its path.

## Restrictions

- Pull only. Do not edit, commit, branch, or push.
- Never use physical NPU 5 or 6.
- Rebuild the small host binding once. This revision adds the corrected CANN
  host infer callback to that library; the old SO does not contain it.
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
export MSDA_RUN_MODE=candidate_compile_probe
export MSDA_FORWARD_LIMIT=1
export MSDA_WARMUP_PAGES=2
export MSDA_REBUILD_EXTENSION=1
bash 12_unirec_0_1b_inference/run_310p_layout_msda_real_background.sh
```

Immediately give Luka the printed absolute `RUN_LOG` and `TAIL_COMMAND`. Follow
the owned PID until `exit_code.txt` appears.

## Required report

If it passes, return:

1. the final `LAYOUT_LAB done` line and measured forward time;
2. confirmation that both warmup calls completed;
3. the `UNIREC_LAYOUT_MSDA_HOST_INFER_OVERRIDE_ACTIVE` line, which must report
   internal output `[1,256,300]`;
4. physical NPU, commit, exact `python=` path, torch, and torch-npu;
5. absolute run root, cache root, extension build log, extension SO, output
   JSON, and exit-code file;
6. confirmation that the owned process exited and its NPU was released.

If it fails, return instead:

1. the first actual TBE/GE error, not only the downstream Python exception;
2. all five MSDA input descriptors and the output descriptor, including
   logical shape, dtype, format, and original format;
3. any `value_spatial_shapes` occurrence, which must now show `ND`, not
   `NCHW`;
4. the exact compile-log directory and failed kernel JSON/path;
5. the same environment and artifact paths listed above.

Then stop. Do not change the production default.
