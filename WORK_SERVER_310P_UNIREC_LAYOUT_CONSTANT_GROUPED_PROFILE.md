# 310P UniRec layout constant-grouped confirmation

Pull the commit containing this brief. Run one new layout lane only:

- TorchAir compiled;
- FP16 body and FP32 reading-order head;
- `torchair_internal` broad weight formatting;
- `constant_grouped` for all 27 native depthwise Conv2D modules;
- no FrozenBN fusion, precompute, or buffer preformatting;
- batch size 1 and one CPU thread.

This is the 310P adoption gate for the new depthwise implementation. It must
eliminate both repeated TransData stages for every targeted depthwise weight:

1. 27 logical NCHW weights to `FRACTAL_Z`;
2. 27 `FRACTAL_Z` weights to the grouped physical `FRACTAL_Z` descriptor.

Do not change the repository default in this task. The local author will do
that only after this 310P run passes.

## Reuse the completed internal-weight baseline

Do not rerun native weights or the previous `torchair_internal` lane. Set
`INTERNAL_RUN_ROOT` to the completed 310P run produced by
`run_310p_layout_internal_weight_profile_background.sh`. It must contain:

```text
exit_code.txt = 0
forward_internal.json
profile_internal/profile_suite_summary.json
```

The new runner reads the raw kernel CSV referenced by that profile and compares
it with the new raw kernel CSV. It does not depend on truncated top-N tables.

## Restrictions

- Pull only. Do not edit tracked files, commit, push, or create a branch.
- Use one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Use the same Python, models, OpenOCR checkout, OmniDocBench images, and real
  page as the completed internal-weight run.
- Use a fresh compile-cache root for the `constant_grouped` graph.
- Run in the background. Immediately give Luka the absolute `RUN_LOG` and the
  printed `tail -f` command.
- Do not rerun eager, FP32, prefill, recognition, decode, or evaluation.
- Do not substitute another depthwise rewrite or enable FrozenBN formatting.
- Do not loosen or reinterpret the structural gate. The analyzer itself checks
  the 27-module rewrite inventory, both target TransData stages, and layout
  boxes.
- Stop after reporting this one result. Do not make `constant_grouped` the
  default on the work server.

## Exact 910B2 result for comparison

Same page and settings, physical NPU 7, commit `2e71034`:

| Metric | Internal + native depthwise | Internal + constant-grouped |
|---|---:|---:|
| Clean 20-repeat device mean | 13.967634 ms | 12.700917 ms |
| Profiled device time | 14.001254 ms | 12.738695 ms |
| Total TransData | 615 / 2.032900 ms | 561 / 1.163940 ms |
| Logical depthwise weight targets | 27 / 0.107400 ms | 0 |
| Grouped-FZ depthwise targets | 27 / 0.766460 ms | 0 |

The candidate was 1.0997x faster. All 76 boxes, coordinates, scores, class
labels, and reading-order values were exact. The result digest matched. All 54
targeted TransData calls disappeared.

910B2 evidence root:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/12_unirec_0_1b_inference/layout_constant_grouped_910b_2e71034_20260815T101955/
```

The previously reported 310P internal-weight anchor was approximately 34.48 ms,
846 TransData calls, and 8.84 ms total TransData. Use the existing baseline
artifact as the source of truth; do not type those approximate values into the
comparison.

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
export INTERNAL_RUN_ROOT="${INTERNAL_RUN_ROOT:?set the completed internal-weight run root}"

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -f "$LAYOUT_MODEL/model.safetensors"
test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
test -d "$IMAGES_DIR"
test -d "$INTERNAL_RUN_ROOT"

bash 12_unirec_0_1b_inference/run_310p_layout_constant_grouped_profile_background.sh
```

The launcher immediately prints `RUN_ROOT`, `RUN_LOG`, `TAIL_COMMAND`, and
`EXIT_CODE_FILE`. Send `RUN_LOG` and `TAIL_COMMAND` to Luka immediately.

## Automated pass gate

The analyzer exits nonzero unless all of these are true:

- the baseline raw CSV contains 27 logical and 27 grouped target calls;
- the candidate contains zero calls for both sets;
- the candidate reports exactly 27 rewritten modules, all backed by frozen
  prepacked grouped-FZ tensors;
- the box count and class/label sequence match;
- paired mean IoU is at least 0.999;
- no more than one reading-order value changes.

The gate intentionally reports coordinate, score, order, and digest differences
even when they are not required to be bit-exact.

## Completion

Wait for the owned PID and require `exit_code.txt` to contain `0`. Return:

1. the complete `UNIREC_LAYOUT_CONSTANT_GROUPED: PASS ...` line;
2. all 15 `UNIREC_LAYOUT_CONSTANT_GROUPED_TD ...` lines;
3. the `UNIREC_LAYOUT_CONSTANT_GROUPED_OUTPUT ...` path;
4. physical NPU, CANN, torch, and torch_npu;
5. every warning about internal formats, graph loading/compilation, JIT, or
   profiler parsing;
6. absolute paths to `run.log`, `comparison_summary.json`, candidate forward
   JSON, candidate profile summary, parsed profile, raw candidate kernel CSV,
   and fresh compile cache.

Then stop. Do not continue to another optimization or change defaults.
