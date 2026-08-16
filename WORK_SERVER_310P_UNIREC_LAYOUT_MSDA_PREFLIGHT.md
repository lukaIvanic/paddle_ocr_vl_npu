# 310P UniRec layout native-MSDA preflight

Run only this fast support check. Do not install or compile anything yet.

Use commit `05e8709` or newer. The 910B2 control at `05e8709` completed with
exit code 0 and reported:

```text
UNIREC_LAYOUT_MSDA_RUNTIME status=no_python_wrapper mx_driving=false
UNIREC_LAYOUT_MSDA_PREFLIGHT verdict=NEEDS_BINDING_RUNTIME_PROBE headers=3 symbols=8 metadata=6 metadata_310p=0 runtime=no_python_wrapper
```

That control proves the runner and empty-result handling work. It does not
predict the 310P verdict; the point of this task is to measure that verdict.

The current PP-DocLayoutV2 path decomposes each multi-scale deformable-attention
call into three GridSample calls plus repeated Cast and Transpose operations.
CANN 9.0 on the 910B machine contains
`aclnnMultiScaleDeformableAttnFunction`, but the installed `torch_npu 2.10`
does not expose it through Python. Older public reports show that some CANN and
DrivingSDK releases did not ship a working 310P kernel. This preflight checks
the work server's actual installed stack rather than assuming either outcome.

## Restrictions

- Pull only. Do not edit tracked files, create a branch, commit, or push.
- Do not install or build `mx_driving`, DrivingSDK, torch-npu, or any custom op.
- Never use physical NPU 5 or 6.
- Use one free 310P and the existing UniRec Python environment.
- Do not run the layout model, prefill, decode, or OmniDocBench.
- The optional NPU call runs only if `mx_driving` is already installed and
  exposes `multi_scale_deformable_attn`.
- This should finish in well under one minute. If it does not, inspect the live
  log rather than waiting blindly.

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

bash 12_unirec_0_1b_inference/run_310p_layout_msda_preflight_background.sh
```

Immediately give Luka the printed absolute `RUN_LOG` and `TAIL_COMMAND`.
Follow the owned PID until `exit_code.txt` appears.

## Interpretation

The final line has one of these verdicts:

- `VERIFIED_RUNTIME_SUPPORTED`: an already-installed Python wrapper completed
  the production-shape NPU call.
- `READY_FOR_MINIMAL_BINDING_PROBE`: header, symbols, and explicit 310P package
  metadata exist; the next task is a minimal binding and real call.
- `NEEDS_BINDING_RUNTIME_PROBE`: the API exists, but static files do not prove
  the 310P kernel is registered. A minimal binding is required to decide.
- `VERIFIED_RUNTIME_UNSUPPORTED`: the real call produced an explicit missing or
  unsupported 310P operator error.
- `BLOCKED_MISSING_NATIVE_API`: the installed toolkit lacks the required API.

Absence of a Python wrapper is not failure. `torch_npu` does not expose this op
on the 910B reference system either.

## Completion report

Return:

1. the complete `UNIREC_LAYOUT_MSDA_PREFLIGHT ...` line;
2. the `UNIREC_LAYOUT_MSDA_RUNTIME ...` line and any runtime error line;
3. physical NPU, CANN/toolkit paths, torch, and torch_npu versions;
4. the contents of `header_hits.txt`, `symbol_hits.txt`,
   `operator_metadata_310p_hits.txt`, and `summary.env`;
5. absolute paths to `run.log`, `runtime_probe.json`, `environment.txt`, and
   `npu_after.txt`;
6. total process wall time.

Then stop. Do not proceed to installation, binding, compilation, or model work.
