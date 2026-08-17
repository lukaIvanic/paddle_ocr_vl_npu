# 310P UniRec optimized vision shape/batch sweep

Run the committed background sweep once. It measures only the optimized compiled
full vision encoder. Numerical differences are warnings and never fail the
sweep. Execution, compilation, missing-output, and device errors remain fatal.

Do not edit tracked files, create a branch, commit, or push. Do not use physical
NPU 5 or 6. Preserve the validated `python_nosym` executable path; do not apply
`readlink -f` to the executable.

## Contract

- implementation: `constant_grouped_all + torchair_internal`;
- dtype: FP16;
- NPU JIT compile: off;
- timing: synchronized NPU events, 2 warmups, 20 repeats;
- scope: compiled vision encoder graph only, including stem, stages 0-3 and
  final vision projection;
- excluded: H2D, preprocessing, layout, text prefill and decode;
- one canvas per process to avoid cumulative-memory ambiguity;
- reuse the existing warmed production recognition cache.

The exact matrix is:

| Canvas | Physical batches |
|---|---|
| `960x64` | B1, B4, B16 |
| `512x256` | B1, B2, B4, B8, B16 |
| `960x256` | B1, B4 |
| `512x512` | B1, B4, B8 |
| `960x512` | B1, B4 |

## Launch

Pull the commit containing this brief, then:

```bash
set -eo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main
source npu-setup
case ",${ASCEND_RT_VISIBLE_DEVICES:?}," in
  *,5,*|*,6,*) echo "Do not use physical NPU 5 or 6" >&2; exit 1 ;;
esac

export PYTHON_BIN="${PYTHON_BIN:?validated python_nosym executable}"
export MODEL="${MODEL:?OpenDoc UniRec model directory containing model.pth}"
export COMPILE_CACHE="${COMPILE_CACHE:?warmed production recognition cache parent}"
export REFERENCE_JSON="$REPO/12_unirec_0_1b_inference/references/unirec_vision_shape_batch_sweep_910b_20260817.json"

bash 12_unirec_0_1b_inference/run_vision_shape_batch_sweep_background.sh
```

The launcher immediately prints an absolute `RUN_ROOT`, `RUN_LOG`, and
`TAIL_COMMAND`. Follow the log. Every graph prints a begin marker before its
first call and an end marker containing its synchronized median, crops/s,
MPixels/s, warning count and peak allocation increment.

## Completion

The run is complete only when:

- `exit_code.txt` is `0`;
- `combined.json` exists;
- the log ends with `UNIREC_VISION_SWEEP_WORKER_END status=0`;
- all 15 requested buckets appear in `combined.json`;
- every numerical discrepancy remains visible under `warnings` and none was
  used as a hard pass/fail criterion;
- `UNIREC_VISION_SWEEP_CROSS_CHIP` prints all 15 comparisons against the
  committed 910B control.

Return the absolute `RUN_ROOT`, `RUN_LOG`, `combined.json`, process wall time,
all `UNIREC_VISION_SWEEP_COMBINED`, `UNIREC_VISION_SWEEP_ASPECT`,
`UNIREC_VISION_SWEEP_CROSS_CHIP`, warning lines, and any failed command/log if
execution does not complete.
