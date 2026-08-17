# 310P canonical-native prefill plus current decode comparison

## Why this run exists

The current 957-crop decode diagnostic did not use the same upstream inference
contract as the canonical 310P run that scored about 90.13 Overall.

Canonical 310P accuracy contract:

- `production_v1` five vision buckets;
- native vision weights;
- native focal-depthwise operations;
- eager FP32 native layout;
- cross-KV 1320, self-KV/max length 2048, decode B128.

Current problematic artifact:

- `310p_k10_l4_all` ten-bucket plan;
- `torchair_internal` vision weights;
- `constant_grouped_all` focal-depthwise operations;
- otherwise the same accuracy-sized KV and B128 decoder.

The 256-crop result showed mostly exact tokens but several 310P-only
single-token loops.  Zero live-arena warmup did not remove them.  This run
restores the canonical upstream vision path and compares the resulting first
128 pages directly with the saved canonical 310P recognition trace.

## Inputs from the history audit

Pull the commit named in Luka's message.  Do not edit tracked files, create a
branch, commit, or push.

From the identified canonical 90.13 run, set `CANONICAL_TRACE` to its actual
`output/recognition_trace.jsonl`.  It must contain more than 30,000 rows.  Reuse
the canonical production cache parent if it still exists.

```bash
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"

export PYTHON_BIN=/absolute/path/to/the/validated/venv/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export LAYOUT_MODEL=/absolute/path/to/PP-DocLayoutV2
export OPENOCR_ROOT=/absolute/path/to/OpenOCR
export IMAGES_DIR=/absolute/path/to/OmniDocBench/images
export COMPILE_CACHE=/absolute/path/to/the/historical/production/cache/parent
export CANONICAL_TRACE=/absolute/path/to/the/90.13/run/output/recognition_trace.jsonl
export UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE=/absolute/path/to/the/passed/B128/decode/cache/parent
export ASCEND_RT_VISIBLE_DEVICES=0  # one free physical 310P, 0-3 only
export CPUSET=0-63
```

Keep `PYTHON_BIN` inside the validated venv.  Do not use `readlink -f` on a
venv symlink that resolves outside it.

## Run

```bash
bash 12_unirec_0_1b_inference/run_310p_canonical_native_prefill_decode_parity_background.sh
```

Immediately paste the absolute `RUN_ROOT`, `RUN_LOG`, and follow command:

```bash
tail -f /absolute/RUN_LOG/from/the/launcher
```

The runner performs one canonical-native first-128 prefill export and one
957-crop decode replay.  It does not run the traced timing replay or evaluation.
It verifies every artifact CRC and compares each output against the canonical
310P trace by request ID.  It intentionally performs zero calls on the live
arena before token 1.

The historical native vision graphs should already exist.  Report immediately
if a graph compiles instead of loading.  Do not delete caches or switch to the
optimized vision path to avoid a missing cache.

## Required report

Paste back verbatim:

1. `UNIREC_CANONICAL_NATIVE_PREFILL_BEGIN/END`
2. `CANONICAL_NATIVE_ARTIFACT_CROPS`
3. `UNIREC_PRODUCTION_DECODE_REPLAY_END`
4. From `replay.json`:
   - `workload.generated_length`
   - `decode.decode_iterations`
   - `decode.decode_s`
   - `decode.raw_decode_tokens_per_s`
   - `decode.effective_decode_tokens_per_s`
   - `slot_efficiency`
   - `decode.timing_detail`
   - `validation`
5. `UNIREC_DECODE_OUTPUT_PARITY: PASS`
6. `DECODE_OUTPUT_PARITY_REPORT`
7. `DECODE_CACHE_OM_INVENTORY_UNCHANGED`
8. `ALL_COMPILE_CACHE_OM_PATHS_UNCHANGED` or the exact added paths
9. `RUN_ROOT`, `RUN_LOG`, and `exit_code.txt`

## Interpretation

- If canonical-native prefill restores the canonical output distribution and
  removes the new repeated caps, the decoder was not the source.  The optimized
  K10/internal/grouped vision path is not accuracy-safe on 310P.
- If canonical-native cross-KV still produces the new loops, the difference is
  in artifact replay/current decode versus the historical live production path.
- CPU affinity cannot explain different token sequences.  For speed, compare
  `decode_s` with admission/ingress fields separately; do not attribute graph
  wait time to CPU loading without those fields.
