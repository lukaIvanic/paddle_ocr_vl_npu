# 310P K20 representative-128 compiled-FP32 accuracy gate

## Goal

Run one production-faithful representative-128 candidate with W4/T8, K20
vision, layout B2, and compiled FP32 layout. Measure full prefill+decode
throughput and compare accuracy against the canonical eager-FP32 full-run
outputs restricted to the same 128 pages.

This is candidate-only NPU inference. Do not rerun the FP16 or eager-FP32 NPU
baselines. The prior K20 compiled-FP16 summary supplies the performance
reference. The canonical full1651 eager-FP32 output supplies the accuracy
reference.

Expected process wall time is 2-3 minutes, with 4 minutes as a conservative
bound. There must be zero graph compilations. If the process exceeds four
minutes, identify and report the active phase, process state, latest log marker,
and any compile/recompile text before waiting longer.

## Constraints

- Pull only. Do not edit tracked files, commit, push, or create a branch.
- Use one free physical 310P device from 0-3. This server has no `npu-setup`.
- Use the real venv `python_nosym`; do not pass it through `readlink -f`.
- Use the completed K20 recognition/decode cache.
- Use the compiled-FP32 B2 cache that passed the three-lane B2 timing probe.
- Use the frozen OmniDocBench TeX Live 2025 runtime selected by the repository
  environment script. Do not use the ambient TeX 2022 installation.
- `K20_FP16_REFERENCE_SUMMARY` is the completed 8.32-pages/s K20 W4/T8
  prefill summary. It is a performance reference only.
- `EAGER_FP32_FULL_OUTPUT` is the canonical full1651 eager-FP32 output that
  scored about 90.23 Overall. It must still contain all page Markdown files and
  `run_summary.json`.
- Do not delete or repair caches automatically after a failure.

## Launch

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

export PYTHON_BIN=/absolute/path/to/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export LAYOUT_MODEL=/absolute/path/to/PP-DocLayoutV2_safetensors
export OPENOCR_ROOT=/absolute/path/to/OpenOCR
export IMAGES_DIR=/absolute/path/to/OmniDocBench-v1.6/images
export DATASET_JSON=/absolute/path/to/OmniDocBench.json
export COMPILE_CACHE=/absolute/path/to/completed/K20/cache/parent
export LAYOUT_FP32_CACHE=/absolute/path/to/warmed/compiled-FP32/B2/cache
export K20_FP16_REFERENCE_SUMMARY=/absolute/path/to/8.32-pages-s/K20/run_summary.json
export EAGER_FP32_FULL_OUTPUT=/absolute/path/to/canonical/eager-FP32/full1651/output
export EVALUATOR_ROOT=/absolute/path/to/clean/OmniDocBench/evaluator
export EVAL_PYTHON=/absolute/path/to/evaluator/python
export ASCEND_RT_VISIBLE_DEVICES=0
export TASKSET_CPUS=0-63
export MATCH_WORKERS=12
export TEDS_WORKERS=12
export CDM_WORKERS=64

bash 12_unirec_0_1b_inference/run_310p_k20_rep128_compiled_fp32_accuracy_background.sh
```

The device is an example, not a reservation. Select an actually free device.
Immediately give Luka the printed absolute `RUN_LOG` path for `tail -f`.

## Monitor and time-straggler contract

Inspect every 15 seconds:

```bash
while [[ ! -s "$RUN_ROOT/exit_code.txt" ]]; do
  date -Ins
  ps -p "$(cat "$RUN_ROOT/pid.txt")" \
    -o pid,etime,stat,%cpu,%mem --no-headers || true
  grep -E \
    'UNIREC_310P_REP128_FP32_PHASE|UNIREC_TWO_PHASE_(PREFILL|DECODE|END)|UNIREC_LAYOUT_PROCESS_(HEARTBEAT|PAGE)|compile|recompil|Traceback|ERROR' \
    "$RUN_LOG" | tail -20
  sleep 15
done
```

Expected phase bounds:

- startup plus cache/model load: 40-60 seconds;
- candidate inference after workers are ready: 40-60 seconds;
- both 128-page evaluator replays combined: 30-60 seconds;
- reporting: under 10 seconds.

The runner records the exact candidate command. A new OM or visible graph
compilation makes the run cold and fails the hot-cache gate.

## Required report

Wait for:

```text
UNIREC_310P_REP128_FP32_HOT_OM_INVENTORY_UNCHANGED
UNIREC_310P_K20_REP128_COMPILED_FP32_ACCURACY: PASS
```

Then paste:

```bash
cat "$RUN_ROOT/final_report.txt"
cat "$RUN_ROOT/process_wall_s.txt"
cat "$RUN_ROOT/hot_om.diff"
```

Also give the absolute paths to:

- `run.log`
- `candidate/output/run_summary.json`
- `markdown_comparison.json`
- both `full_eval_summary.json` files
- `comparison.json`

Report externally observed peak HBM. State separately:

1. candidate prefill pages/s versus the prior K20 compiled-FP16 reference;
2. candidate full sequential-core pages/s;
3. exact and image-tag-stripped Markdown page parity;
4. candidate-minus-baseline deltas for text edit, Page CDM, Page TEDS,
   reading edit, and Overall points.

Do not call small-subset structural parity proof of full accuracy. This gate
only decides whether the full1651 compiled-FP32 experiment is justified.
