# 310P UniRec layout direct-Softmax confirmation

Pull the commit containing this brief. Run one new layout lane only. It changes
the six reading-order attention layers from the explicit CogView stabilization:

```python
scaled = scores / 32
maximum = scaled.max(dim=-1, keepdim=True).values
softmax((scaled - maximum) * 32)
```

to the algebraically equivalent direct expression:

```python
softmax(scores)
```

Softmax already performs stable maximum subtraction. The candidate must remove
the six expensive `[1,8,302,302]` ArgMaxWithValue kernels while preserving the
layout result.

## Reuse the completed FrozenBN baseline

Do not rerun a baseline. Set `FROZENBN_RUN_ROOT` to the completed successful run
from `run_310p_layout_frozenbn_profile_background.sh`. It must contain:

```text
exit_code.txt = 0
forward_constant_grouped_frozenbn.json
profile_frozenbn/profile_suite_summary.json
```

The new candidate keeps every other setting identical: TorchAir, FP16 body,
FP32 reading-order head, `torchair_internal`, `constant_grouped`, preformatted
FrozenBN buffers, B1, and one CPU thread.

## Restrictions

- Pull only. Do not edit tracked files, commit, push, or create a branch.
- Use one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Use the same Python, models, OpenOCR checkout, image directory, and real page
  as the completed FrozenBN run.
- Use a fresh compile-cache root for the direct-Softmax graph.
- Run in the background and immediately give Luka the absolute `RUN_LOG` and
  printed `tail -f` command.
- Run only the candidate forward, profile, and automated comparison. Do not run
  prefill, recognition, decode, evaluation, or another A/B.

## Exact 910B2 result

The same one-setting A/B passed on physical NPU 7 at commit `e7830b8`:

| Metric | Stabilized | Direct Softmax |
|---|---:|---:|
| Clean 20-repeat device mean | 12.241929 ms | 12.105062 ms |
| Speedup | | 1.01131x |
| ArgMaxWithValue | 8 / 0.155120 ms | 2 / 0.029020 ms |
| `[1,8,302,302]` ArgMax calls | 6 | 0 |

All 76 boxes, coordinates, scores, class labels, reading-order values, and the
result digest were bit-exact.

Evidence root:

```text
/workspace/repos/paddle_ocr_vl_npu/tmp/12_unirec_0_1b_inference/layout_cogview_direct_910b_e7830b8_20260815T112731/
```

The 310P FrozenBN baseline reported approximately 27.75 ms and
ArgMaxWithValue `8 / 3.37 ms`. Use the saved baseline profile as the exact
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
export FROZENBN_RUN_ROOT="${FROZENBN_RUN_ROOT:?set the completed 310P FrozenBN run root}"

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -f "$LAYOUT_MODEL/model.safetensors"
test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
test -d "$IMAGES_DIR"
test -d "$FROZENBN_RUN_ROOT"

bash 12_unirec_0_1b_inference/run_310p_layout_cogview_softmax_profile_background.sh
```

The launcher prints `RUN_ROOT`, `RUN_LOG`, `TAIL_COMMAND`, and
`EXIT_CODE_FILE`. Send `RUN_LOG` and `TAIL_COMMAND` to Luka immediately.

## Automated pass gate

The analyzer exits nonzero unless:

- the saved baseline is stabilized and the candidate is direct Softmax;
- baseline has six `[1,8,302,302]` ArgMax calls and candidate has zero;
- total ArgMax count falls by exactly six and ArgMax time improves;
- clean 20-repeat forward time does not regress;
- box count and class/label sequence match;
- coordinate drift is at most one pixel, score drift at most `0.005`, and no
  more than one reading-order value changes.

The report also states whether output is bit-exact; do not claim exactness unless
the digest and all zero-drift fields confirm it.

## Completion report

Wait for `exit_code.txt = 0`. Return:

1. the complete `UNIREC_LAYOUT_COGVIEW_SOFTMAX: PASS ...` line;
2. all 15 `UNIREC_LAYOUT_COGVIEW_SOFTMAX_KERNEL ...` lines;
3. the `UNIREC_LAYOUT_COGVIEW_SOFTMAX_OUTPUT ...` path;
4. physical NPU, CANN, torch, and torch_npu;
5. all internal-format, compilation/cache, JIT, and profiler warnings;
6. absolute paths to `run.log`, comparison JSON, candidate forward JSON,
   profile summary, parsed profile, raw candidate kernel CSV, and fresh cache.

Then stop.
