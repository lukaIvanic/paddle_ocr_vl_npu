# 310P UniRec optimized vision shape/batch sweep

Run the committed background sweep once. It measures only the optimized compiled
full vision encoder. Numerical differences are warnings and never fail the
sweep. Execution, compilation, missing-output, and device errors remain fatal.

Do not edit tracked files, create a branch, commit, or push. Do not use physical
device IDs outside the server's real four-device inventory (`0` through `3`).
This server does not have `npu-setup`; do not call it. Reuse the same validated
CANN/torch-npu shell environment and `python_nosym` executable that passed the
earlier UniRec runs. Do not apply `readlink -f` to the executable.

Do not copy any 910B path from this brief. `PYTHON_BIN`, `MODEL`, and
`COMPILE_CACHE` must be the already-validated paths on the 310P server.

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

## 910B2 control

The committed comparison artifact is:

`12_unirec_0_1b_inference/references/unirec_vision_shape_batch_sweep_910b_20260817.json`

It was measured on one physical Ascend 910B2 with the same graph settings. All
15 points completed. Six numerical comparisons emitted warnings; warnings did
not stop the sweep. The 948-second process wall time was dominated by cold
compilation of current-source graph keys and is not an inference timing.

| Bucket | Median ms | Crops/s | MPixels/s |
|---|---:|---:|---:|
| `960x64_b1` | 5.413 | 184.7 | 11.4 |
| `960x64_b4` | 6.594 | 606.6 | 37.3 |
| `960x64_b16` | 10.737 | 1490.1 | 91.6 |
| `512x256_b1` | 5.279 | 189.4 | 24.8 |
| `512x256_b2` | 6.272 | 318.9 | 41.8 |
| `512x256_b4` | 8.160 | 490.2 | 64.3 |
| `512x256_b8` | 11.112 | 719.9 | 94.4 |
| `512x256_b16` | 16.729 | 956.4 | 125.4 |
| `960x256_b1` | 6.253 | 159.9 | 39.3 |
| `960x256_b4` | 10.131 | 394.8 | 97.0 |
| `512x512_b1` | 6.053 | 165.2 | 43.3 |
| `512x512_b4` | 10.350 | 386.5 | 101.3 |
| `512x512_b8` | 15.794 | 506.5 | 132.8 |
| `960x512_b1` | 7.978 | 125.4 | 61.6 |
| `960x512_b4` | 15.194 | 263.3 | 129.4 |

## Launch

Pull the commit containing this brief, then:

```bash
set -eo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main

# Keep the already-validated 310P CANN/torch-npu environment. There is no
# npu-setup helper on this server. Inspect npu-smi, choose one free physical
# device from the actual four-device inventory, then expose only that device.
: "${ASCEND_RT_VISIBLE_DEVICES:?set one free physical 310P device from 0 through 3}"
case "$ASCEND_RT_VISIBLE_DEVICES" in
  0|1|2|3) ;;
  *) echo "Expected exactly one physical 310P device ID from 0 through 3" >&2; exit 1 ;;
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

Some matrix points can require a new graph compilation even when the production
cache is warm. A long interval after `POINT_BEGIN` is not by itself a stall.
Confirm that the owned process is alive and the log is advancing before taking
action. Do not restart a healthy compile.

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
