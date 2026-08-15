# 310P UniRec layout FrozenBN-buffer confirmation

Pull the commit containing this brief. Run one new layout lane only. It adds
preformatted FrozenBN buffers to the already-passed layout configuration:

- TorchAir compiled;
- FP16 body and FP32 reading-order head;
- `torchair_internal` broad weight formatting;
- `constant_grouped` for all 27 depthwise Conv2D modules;
- original FrozenBN math with all four buffers stored as NC1HWC0;
- batch size 1 and one CPU thread.

Do not rerun the completed no-FrozenBN `constant_grouped` baseline. The new
runner reads its forward JSON and raw profile CSV directly.

## What this tests

The model contains 80 `PPDocLayoutV2FrozenBatchNorm2d` modules and four constant
buffers per module. The candidate must therefore remove exactly 320
NCHW-to-NC1HWC0 TransData calls, with the per-channel reductions matching the
model inventory. It must preserve every layout box, coordinate, score, class,
label, reading-order value, and result digest exactly.

This task does not fuse BN into Conv2D and does not change the FrozenBN formula.
The 42 ordinary evaluation BatchNorm modules are deliberately left unchanged.

## Restrictions

- Pull only. Do not edit tracked files, commit, push, or create a branch.
- Use one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Use the same Python, model files, OpenOCR checkout, image directory, and real
  page as the completed `constant_grouped` run.
- Use a fresh compile-cache root for the new graph.
- Run in the background and immediately give Luka the absolute log path and
  printed `tail -f` command.
- Stop after the one candidate forward, profile, and automated comparison. Do
  not run prefill, recognition, decode, evaluation, or another A/B.

## Exact 910B2 comparison

The same one-setting A/B passed on physical NPU 7 at source commit `2e71034`:

| Metric | No FrozenBN preformat | Preformatted FrozenBN |
|---|---:|---:|
| Clean 20-repeat device mean | 12.700917 ms | 12.241929 ms |
| Speedup | | 1.03749x |
| Total TransData | 561 / 1.163940 ms | 241 / 0.726660 ms |
| NCHW to NC1HWC0 | 493 | 173 |
| Removed FrozenBN buffer calls | | exactly 320 |

All 76 boxes and all numerical/order fields were bit-exact. The digest matched.
The 320 removed calls saved 0.437280 ms of total TransData time in the profiled
forward. The remaining 168 `1x256x1x1` calls belong to 42 ordinary evaluation
BatchNorm modules, not the 80 FrozenBN modules handled by this task.

910B2 evidence:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/12_unirec_0_1b_inference/layout_constant_grouped_frozenbn_910b_2e71034_20260815T110157/
```

The completed 310P baseline reported approximately 28.06 ms, 792 TransData
calls, and 2.35 ms total TransData. Treat the completed artifact as the exact
source of truth rather than these rounded values.

## Launch

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git status --short --branch
git pull --ff-only origin main
git status --short --branch

source npu-setup
case ",${ASCEND_RT_VISIBLE_DEVICES:-}," in
  *,5,*|*,6,*) echo "REJECTED_PHYSICAL_DEVICE_5_OR_6" >&2; exit 1 ;;
esac

export PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
export MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
export LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
export OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
export IMAGES_DIR="${IMAGES_DIR:?set the existing OmniDocBench images directory}"
export CONSTANT_GROUPED_RUN_ROOT="${CONSTANT_GROUPED_RUN_ROOT:?set the completed 310P constant-grouped run root}"

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -f "$LAYOUT_MODEL/model.safetensors"
test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
test -d "$IMAGES_DIR"
test -d "$CONSTANT_GROUPED_RUN_ROOT"

bash 12_unirec_0_1b_inference/run_310p_layout_frozenbn_profile_background.sh
```

The launcher prints `RUN_ROOT`, `RUN_LOG`, `TAIL_COMMAND`, and
`EXIT_CODE_FILE`. Send `RUN_LOG` and `TAIL_COMMAND` to Luka immediately.

## Automated pass gate

The analyzer exits nonzero unless all of these are true:

- the baseline has no preformatted FrozenBN buffers;
- the candidate inventories exactly 80 modules and 320 NC1HWC0 buffers;
- the NCHW-to-NC1HWC0 reductions match the candidate module/channel inventory;
- total TransData count falls by exactly 320 and TransData time improves;
- clean 20-repeat forward time does not regress;
- box count, class/label sequence, coordinates, scores, reading order, and digest
  are exact.

## Completion report

Wait for the owned PID and require `exit_code.txt` to contain `0`. Return:

1. the complete `UNIREC_LAYOUT_FROZENBN: PASS ...` line;
2. all 15 `UNIREC_LAYOUT_FROZENBN_TD ...` lines;
3. the `UNIREC_LAYOUT_FROZENBN_OUTPUT ...` path;
4. physical NPU, CANN, torch, and torch_npu;
5. every warning about internal formats, compilation/cache loading, JIT, or
   profiler parsing;
6. absolute paths to `run.log`, `comparison_summary.json`, candidate forward
   JSON, profile summary, parsed profile, raw candidate kernel CSV, and fresh
   compile cache.

Then stop.
