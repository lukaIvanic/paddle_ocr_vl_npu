# 310P UniRec compiled layout internal-weight profile

Pull the commit containing this brief. Run one new layout lane only:

- TorchAir compiled;
- FP16 body and FP32 reading-order head;
- native depthwise convolutions;
- `torchair_internal` weight formatting;
- no FrozenBN fusion, precompute, or buffer preformatting;
- one CPU thread.

This tests the existing broad TorchAir internal-weight implementation. It
preformats ordinary Conv2D and Linear weights. Do not modify the implementation
or add a Conv-only variant in this task.

## Reuse the completed native baseline

Do not rerun the four native precision lanes. Set `NATIVE_RUN_ROOT` to the
completed 310P run produced by
`run_layout_precision_profile_matrix_background.sh`. It must contain:

```text
exit_code.txt = 0
forward_compiled_fp16_body_fp32_ro.json
profile_compiled_fp16_body_fp32_ro/profile_suite_summary.json
```

The new runner uses those exact JSONs for baseline timing, TransData, and
layout-output comparison.

## Restrictions

- Pull only. Do not edit tracked files, commit, push, or create a branch.
- Use one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Use the same Python, model, OpenOCR checkout, OmniDocBench images, and real
  page as the completed four-lane run.
- Use a fresh compile-cache root for the internal-weight graph.
- Run in the background and immediately give Luka the absolute `RUN_LOG` and
  printed `tail -f` command.
- Do not rerun native, eager, FP32, prefill, recognition, decode, or evaluation.
- Do not enable `group16` or FrozenBN buffer formatting. This task changes only
  `weight_format=native` to `weight_format=torchair_internal`.
- Preserve numerical differences. Do not apply an arbitrary tolerance and do
  not call a digest difference a failure.

## Exact 910B2 result for comparison

Same page, compiled FP16 body, FP32 reading-order head, native depthwise:

| Metric | Native weights | `torchair_internal` |
|---|---:|---:|
| Clean 20-repeat device mean | 14.469415 ms | 13.967634 ms |
| Total TransData | 678 / 2.797180 ms | 615 / 2.032900 ms |
| All NCHW→FRACTAL_Z | 123 / 1.177140 ms | 28 / 0.109320 ms |
| Regular FP16 Conv2D NCHW→FZ | 95 calls | 0 calls |
| Residual depthwise NCHW→FZ | 27 calls | 27 calls |

The one-page output kept all 76 boxes and the exact class/label sequence. It
changed one reading-order `custom_value` and had small score/coordinate drift,
so the digest did not match. Report the analogous 310P result exactly.

910B2 evidence root:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/12_unirec_0_1b_inference/layout_internal_weights_910b_bea0f1f_20260815T0944/
```

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
export NATIVE_RUN_ROOT="${NATIVE_RUN_ROOT:?set the completed four-lane run root}"

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -f "$LAYOUT_MODEL/model.safetensors"
test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
test -d "$IMAGES_DIR"
test -d "$NATIVE_RUN_ROOT"

bash 12_unirec_0_1b_inference/run_310p_layout_internal_weight_profile_background.sh
```

The launcher immediately prints `RUN_ROOT`, `RUN_LOG`, `TAIL_COMMAND`, and
`EXIT_CODE_FILE`. Send `RUN_LOG` and `TAIL_COMMAND` to Luka immediately.

## Completion

Wait for the owned PID and require `exit_code.txt` to contain `0`. Return:

1. the complete `UNIREC_310P_LAYOUT_INTERNAL_WEIGHT: PASS ...` line;
2. the `UNIREC_310P_LAYOUT_INTERNAL_WEIGHT_OUTPUT ...` path;
3. physical NPU, CANN, torch, and torch_npu;
4. the top 15 TransData signatures from the new internal profile;
5. every warning about internal formats, graph cache loading/compilation, JIT,
   or profiler parsing;
6. absolute paths to `run.log`, `comparison_summary.json`, the internal forward
   JSON, profile summary, parsed profile, and fresh compile cache.

Then stop. Do not continue to another optimization.
