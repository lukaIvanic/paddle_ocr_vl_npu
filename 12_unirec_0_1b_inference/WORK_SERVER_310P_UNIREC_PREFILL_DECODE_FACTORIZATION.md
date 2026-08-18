# 310P UniRec prefill/decode factorization

## Goal

Determine which part of the new vision-prefill system caused the first-128
runaway decode outputs. Do not rerun the full benchmark. Do not rebuild graphs.

The canonical full-1,651 baseline is already established:

- prefill: about 535 s / 3.08 pages/s;
- decode graph/work: about 290 s / 8.5k raw and 7.7k effective tokens/s;
- sequential core: 1.908 pages/s;
- 32,111 crops completed and about 2.5M output tokens.

The reported sequential rate implies 865.3 s of sequential core. Therefore the
whole decode phase is about 330.3 s: about 290 s of decode graph/work plus 40.3
s of ingress, output, and scheduler overhead. Prefill is about 61.8% of the
sequential core and remains the larger wall bottleneck. The correctness question
is separate. The current decoder
reproduced the canonical output when it received canonical native cross-KV, but
the K10/internal/grouped artifact produced repeated-to-cap outputs.

This gate adds only the missing middle lane:

1. existing canonical: `production_v1 + native + native`;
2. new middle lane: `production_v1 + torchair_internal + constant_grouped_all`;
3. existing optimized: `310p_k10_l4_all + torchair_internal + constant_grouped_all`.

All output checks use the unchanged B128, cross-KV 1320, self-KV 2048 decoder.
The decode probe is one B128 cohort containing every previously reported
mismatch plus short controls. All runaway rows therefore overlap in one decode
tail instead of creating several separate 2,047-step tails.

## Constraints

- Pull only. Do not edit tracked files, commit, push, or create a branch.
- Use one physical 310P device, 0-3. There is no `npu-setup` on this host.
- Use `python_nosym` or another validated executable inside the owned venv. The
  launcher deliberately does not use `readlink -f` on `PYTHON_BIN`.
- Reuse the exact passed B128 decode cache and existing optimized five-bucket
  vision caches. The launcher stops before inference if one is absent.
- Do not delete or rebuild any cache in this task.

## Prepare

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git rev-parse --short HEAD
git status --short
```

Use the artifacts from the already completed diagnostics. Set these four paths
from their printed `RUN_ROOT` values:

```bash
export CANONICAL_ARTIFACT=/absolute/path/to/prefill_artifact_first128_canonical_native
export OPTIMIZED_ARTIFACT=/absolute/path/to/the/K10/first128/prefill_artifact
export OPTIMIZED_MISMATCH_REPORT=/absolute/path/to/the/passed/parity_report.json
export CANONICAL_TRACE=/absolute/path/to/the/canonical/90.13/run/output/recognition_trace.jsonl
```

Validate the identities before launch:

```bash
jq '{workers,layout_batch_size,vision_focal_depthwise_rewrite,vision_weight_format,producer_wall_s,throughput,artifact:{page_count:.artifact.page_count,crop_count:.artifact.crop_count}}' "$CANONICAL_ARTIFACT/summary.json"
jq '{workers,layout_batch_size,vision_focal_depthwise_rewrite,vision_weight_format,producer_wall_s,throughput,artifact:{page_count:.artifact.page_count,crop_count:.artifact.crop_count}}' "$OPTIMIZED_ARTIFACT/summary.json"
jq '{candidate_count,compared_count,token_exact_count,long_output_counts,first_mismatches}' "$OPTIMIZED_MISMATCH_REPORT"
```

Both artifacts must contain 128 pages and 957 crops. The mismatch report must
show the previously observed non-exact rows. If not, stop and report the three
JSON blocks instead of guessing another path.

Export the already validated host-specific paths:

```bash
export PYTHON_BIN=/absolute/path/to/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export LAYOUT_MODEL=/absolute/path/to/PP-DocLayoutV2_safetensors
export OPENOCR_ROOT=/absolute/path/to/OpenOCR
export IMAGES_DIR=/absolute/path/to/OmniDocBench/images
export COMPILE_CACHE=/absolute/path/to/existing/production/cache/parent
export UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE=/absolute/path/to/passed/decode/cache/parent
export ASCEND_RT_VISIBLE_DEVICES=0
export CPUSET=0-63
```

Use a free device from 0-3. The example device is not a reservation.

## Launch

```bash
bash 12_unirec_0_1b_inference/run_310p_prefill_decode_factorization_background.sh
```

The launcher immediately prints `RUN_ROOT`, `RUN_LOG`, and `PID`. Show Luka the
absolute log path so he can follow it directly:

```bash
tail -f "$RUN_LOG"
```

Expected hot-cache wall time is roughly 1-3 minutes. The launcher prints phase
boundaries. If it is quiet for more than 30 seconds, inspect the live log and
process/NPU state immediately. Do not wait silently.

## Completion and report

Wait for `RUN_ROOT/exit_code.txt`. Exit zero and both of these lines are the
completion gate:

```text
UNIREC_PREFILL_DECODE_FACTORIZATION: PASS
UNIREC_FACTORIZATION_OM_INVENTORY_UNCHANGED
```

Paste back:

```bash
cat "$RUN_ROOT/final_report.txt"
cat "$RUN_ROOT/probe_selection.json"
cat "$RUN_ROOT/intermediate_cross_kv.json"
cat "$RUN_ROOT/optimized_cross_kv.json"
cat "$RUN_ROOT/intermediate_parity.json"
cat "$RUN_ROOT/process_wall_s.txt"
```

Interpretation is mechanical:

- `VISION_WEIGHT_OR_DEPTHWISE_PATH_IMPLICATED`: the middle lane mismatched;
  next split `torchair_internal` from `constant_grouped_all` on the existing
  five production buckets.
- `K10_BUCKET_PADDING_OR_MASK_PATH_IMPLICATED`: the middle lane was token-exact
  while K10 was not; retain optimized weights and inspect K10 padding/masks.
- Any OM inventory change invalidates the hot-cache timing and means a cache
  identity changed. Report the diff; do not rerun or delete the cache.
