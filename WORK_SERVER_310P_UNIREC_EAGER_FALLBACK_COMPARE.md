# 310P UniRec native-shape eager fallback timing

## Purpose

Measure whether the production UniRec vision crops that miss all five compiled
buckets are disproportionately slow on Atlas 310P.  This is a timing probe, not
an NPU profile and not a page-pipeline run.

The benchmark uses the exact production eager call:

1. compact uint8 HWC input;
2. production H2D conversion and normalization;
3. `OptimizedUniRecRunner.model.forward_encoder` at the native processed shape.

Synthetic zero pixels preserve the operator shapes and timing contract.  They
avoid layout, disk I/O, crop extraction, compilation, and page-pipeline noise.

## Constraints

- Pull the commit named by Luka.  Do not edit tracked files, commit, push, or
  create a branch on the work server.
- Use one free physical 310P.  Do not use physical NPU 5 or NPU 6.
- Run in one process.  There are no forks, workers, or cross-process NPU events.
- Do not run NPU profiling and do not prepare an OM/TorchAir cache.
- Keep NPU JIT compilation disabled as set by the benchmark.
- Use the committed 13 KB shape workload.  No prior run artifact is needed.

## Resolve the inputs

```bash
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
source npu-setup

export PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
export MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
```

Confirm that `npu-setup` did not select physical NPU 5 or 6.  If it did, stop
and select another free device with the server's normal device-selection
mechanism before continuing.

The committed workload was extracted from the completed 1,651-page 910B2
production manifest.  It contains all 496 eager fallback calls and all 90
unique processed shapes.  Verify it directly:

```bash
export SHAPE_WORKLOAD="$REPO/12_unirec_0_1b_inference/references/eager_fallback_full1651_shape_workload.json"
test -f "$SHAPE_WORKLOAD"
test -x "$PYTHON_BIN"
test -d "$MODEL"
test "$(sha256sum "$SHAPE_WORKLOAD" | awk '{print $1}')" = \
  16ace5f42bf2b6f733a076ed537e2b03690865f4ef4447c0118bd075888cfdcb
```

## Run the short timing probe

The top 12 shapes cover 44.0% of fallback calls and 47.0% of the fallback
weighted pixels.  One warmup per shape is excluded.  Five measured repeats
produce the median used for comparison.

```bash
RUN_ROOT="$REPO/tmp/12_unirec_0_1b_inference/eager_fallback_top12_310p_321c23d_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_ROOT"

nohup "$PYTHON_BIN" \
  "$REPO/12_unirec_0_1b_inference/benchmark_eager_fallback_shapes.py" \
  --model-path "$MODEL" \
  --shape-workload "$SHAPE_WORKLOAD" \
  --output "$RUN_ROOT/result.json" \
  --device npu:0 \
  --max-shapes 12 \
  --warmups 1 \
  --repeats 5 \
  >"$RUN_ROOT/run.log" 2>&1 < /dev/null &

RUN_PID=$!
echo "PID=$RUN_PID"
echo "LOG=$RUN_ROOT/run.log"
echo "RESULT=$RUN_ROOT/result.json"
tail --pid="$RUN_PID" -f "$RUN_ROOT/run.log"
```

Wait for the owned PID.  Do not stop another process.  On completion, verify:

```bash
wait "$RUN_PID"
RUN_RC=$?
echo "exit_code=$RUN_RC"
test "$RUN_RC" -eq 0
test -s "$RUN_ROOT/result.json"
grep -E '^UNIREC_EAGER_FALLBACK_(WORKLOAD|SHAPE|SUMMARY)' "$RUN_ROOT/run.log"
```

## 910B2 reference from the identical benchmark

Commit `321c23d`, physical 910B2 NPU 4, FP16, JIT disabled:

- workload: 496 fallback calls, 90 unique shapes;
- selected top 12: 218 calls, call coverage 0.440, weighted-pixel coverage 0.470;
- normal eager encoder median: approximately 19.7--20.4 ms per crop;
- selected weighted eager encoder: **4.376 s**;
- selected weighted production boundary: **4.537 s**;
- actual top-12 benchmark loop wall time: **1.890 s**.

An additional all-90-shape 910B2 run measured:

- all 496 calls weighted eager encoder: **9.445 s**;
- all 496 calls weighted production boundary: **9.865 s**;
- actual all-90 benchmark loop wall time: **8.207 s**.

Do not interpret those as 310P numbers.  Compare the 310P top-12 weighted
encoder and production-wall fields directly against 4.376 s and 4.537 s.

## Report and stop

Return only:

1. pulled commit, physical NPU, CANN, torch-npu, and Python;
2. exit code and absolute log/result paths;
3. every `UNIREC_EAGER_FALLBACK_SHAPE` line and the final summary line;
4. 310P selected weighted encoder divided by 4.376 s;
5. 310P selected weighted production wall divided by 4.537 s;
6. any shape outlier and the exact error if the run fails.

Stop after the top-12 run.  Do not run all 90 shapes or the full pipeline yet.
