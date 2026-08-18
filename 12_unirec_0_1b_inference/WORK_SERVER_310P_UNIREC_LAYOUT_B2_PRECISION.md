# 310P UniRec layout B2 precision timing

## Goal

Measure the faithful PP-DocLayoutV2 **model forward only** at physical batch
size two. Compare eager FP32, compiled FP32, and the current optimized compiled
FP16 body with an FP32 reading-order head.

This is not a page-pipeline benchmark. Processor, H2D, postprocess, and total
call wall time are recorded separately from synchronized `model_forward_s`.

## Constraints

- Pull only. Do not edit tracked files, commit, push, or create a branch.
- Use one free physical 310P device from 0-3. This server has no `npu-setup`.
- Use `python_nosym`; do not resolve the venv symlink with `readlink -f`.
- Reuse the warmed optimized FP16 layout cache from the K20 run.
- Use a persistent new FP32 cache root. Compiled FP32 may create exactly one B2
  graph on its first run. Steady timing starts only after two warmup calls.
- Do not delete either cache after a failure.

## Launch

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

export PYTHON_BIN=/absolute/path/to/venv/bin/python_nosym
export LAYOUT_MODEL=/absolute/path/to/PP-DocLayoutV2_safetensors
export IMAGES_DIR=/absolute/path/to/OmniDocBench-v1.6/images
export FP32_LAYOUT_CACHE=/persistent/path/layout_b2_fp32_native_cache
export FP16_LAYOUT_CACHE=/absolute/path/to/the/warmed/optimized/layout/cache
export ASCEND_RT_VISIBLE_DEVICES=0

bash 12_unirec_0_1b_inference/run_310p_layout_b2_precision_background.sh
```

Select an actually free device. Immediately give Luka the printed absolute
`RUN_LOG` path so it can be followed with `tail -f`.

## Monitor

Inspect every 15-30 seconds. The runner prints setup, first and second warmup,
and first/last measured calls for every lane:

```bash
grep -E 'UNIREC_LAYOUT_B2_(LANE_SETUP|WARMUP|MEASURE|LANE_RESULT)|compile|ERROR|Traceback' "$RUN_LOG" | tail -20
```

If FP32 compilation is cold, report when it starts and finishes. There are only
three lanes and only one possible new graph: compiled FP32 B2. The optimized
FP16 lane should load its existing B2 graph.

## Required report

Wait for `UNIREC_310P_LAYOUT_B2_PRECISION: PASS`, then paste:

```bash
cat "$RUN_ROOT/final_report.txt"
cat "$RUN_ROOT/process_wall_s.txt"
```

Also provide the absolute `RUN_ROOT`, `run.log`, and `summary.json` paths. State
whether compiled FP32 created a new OM and whether compiled FP16 reused its
existing cache. Do not report setup or warmup latency as forward latency.
