# 310P UniRec layout native-MSDA binding probe

Run only this minimal binding and production-shape runtime comparison. The
preceding preflight reported `READY_FOR_MINIMAL_BINDING_PROBE`, including four
`ascend310p`-specific operator metadata files. Do not integrate the operator
into the layout model yet.

## Purpose

The probe builds a tiny torch-npu C++ extension around the already-installed
`aclnnMultiScaleDeformableAttnFunction`. It does not build an Ascend kernel.
It then tests both FP16 (the production layout-body dtype) and FP32 on the real
PP-DocLayoutV2 MSDA shape:

```text
value              [1, 13125, 8, 32]
spatial_shapes     [[100,100], [50,50], [25,25]]
sampling_locations [1, 300, 8, 3, 4, 2]
attention_weights  [1, 300, 8, 3, 4]
output             [1, 300, 256]
```

Each native result is compared with the exact three-GridSample PyTorch
decomposition. Warm event and wall latency are measured for both paths.

The 910B2 control passed at commit `afeeae2`:

| Dtype | Max abs | Mean abs | Native event | Reference event | Speedup |
|---|---:|---:|---:|---:|---:|
| FP16 | 0.0009766 | 0.0000581 | 0.1173 ms | 0.5285 ms | 4.51x |
| FP32 | 0.00000346 | 0.0000000394 | 0.1187 ms | 0.5330 ms | 4.49x |

The clean 910B2 process took 86 seconds, including the one-time host C++
extension build. Treat roughly two minutes as the expected wall-time budget on
310P. If it takes materially longer, inspect the current phase in `run.log`.

## Restrictions

- Pull only. Do not edit tracked files, create a branch, commit, or push.
- Never use physical NPU 5 or 6.
- Use one free 310P and the existing UniRec Python environment.
- Do not install or update CANN, torch-npu, DrivingSDK, or `mx_driving`.
- Do not run the layout model, prefill, decode, or OmniDocBench.
- This builds one small host C++ extension only. It must use the operator
  binary already present in the installed CANN package.
- Follow the owned background PID. Inspect the live log if a phase stalls.

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

export PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
test -x "$PYTHON_BIN"

bash 12_unirec_0_1b_inference/run_310p_layout_msda_binding_background.sh
```

Immediately give Luka the absolute `RUN_LOG` and printed `TAIL_COMMAND`.

## Required report

Return:

1. `UNIREC_LAYOUT_MSDA_BINDING ...` and both
   `UNIREC_LAYOUT_MSDA_BINDING_DTYPE ...` lines;
2. for FP16 and FP32: status, output shape/dtype, finite check, `max_abs`,
   `mean_abs`, cosine, allclose result, native event/wall ms, reference
   event/wall ms, and speedup;
3. the exact runtime error and traceback if either dtype fails;
4. physical NPU, CANN paths, torch and torch-npu versions;
5. process wall time and absolute paths to `run.log`, `build.log`,
   `binding_probe.json`, `extension_so.txt`, and `npu_after.txt`;
6. confirmation that the owned process exited and its NPU was released.

Then stop. Do not integrate or compile the full layout model. The next decision
depends on FP16 correctness and clean native-versus-reference latency.
