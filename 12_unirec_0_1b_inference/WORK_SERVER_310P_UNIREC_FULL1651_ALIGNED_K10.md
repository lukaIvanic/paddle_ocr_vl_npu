# 310P full-1651 corrected aligned-K10 run

## Goal

Run one optimized, accuracy-scored OmniDocBench v1.6 lane on the 310P. Do not
run a native A/B lane. This is the full follow-up to the passed first-128 gate.

The production contract is:

- W4, recognition preprocessing T8, explicit CPU affinity `0-63`;
- eager FP32 native layout, B2, threshold 0.5;
- four-page vision lookahead and `310p_k10_l4_aligned`;
- `constant_grouped_all` plus `torchair_internal` vision weights;
- corrected direct-2D flat global context only in `1024x704_b1` and
  `1024x1408_b1`;
- zero rejected crops and zero eager vision fallbacks;
- B128 compiled IncreFA decode, cross-KV 1320, self-KV/max length 2048;
- full frozen-runtime OmniDocBench evaluation with HTML image tags removed
  only from evaluator copies.

The latest 310P first-128 run passed prefill and produced 956/957 token-exact
rows versus the older canonical trace. The sole changed row was
`page_000117_crop_0000`: the candidate ended normally at 597 tokens, while the
old canonical path generated a bad 2,047-token repetition. Do not treat that
single difference as an automatic regression. The full evaluator is the gate.

## Constraints

- Pull only. Do not edit tracked files, create a branch, commit, or push.
- Use one free physical device 0-3. This host has four 310P devices and no
  `npu-setup`.
- Preserve the venv's real `python_nosym`; never apply `readlink -f` to it.
- Reuse the exact cache directories from the passed first-128 run.
- Do not delete, rename, or repair caches. A cache miss is evidence and the
  launcher must stop before full inference.
- `/dev/shm` may expose only 64 GiB. The launcher records it but does not reject
  the run solely for that reason.

## Prepare and launch

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git rev-parse --short HEAD

export PYTHON_BIN=/absolute/path/to/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export LAYOUT_MODEL=/absolute/path/to/PP-DocLayoutV2_safetensors
export OPENOCR_ROOT=/absolute/path/to/OpenOCR
export IMAGES_DIR=/absolute/path/to/OmniDocBench/images
export DATASET_JSON=/absolute/path/to/OmniDocBench.json
export COMPILE_CACHE=/absolute/path/to/the/cache/used/by/the/passed/first128
export UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE=/absolute/path/to/the/passed/B128/decode/cache/parent
export EVALUATOR_ROOT=/absolute/path/to/clean/OmniDocBench/evaluator
export EVAL_PYTHON=/absolute/path/to/frozen/evaluator/python
export ASCEND_RT_VISIBLE_DEVICES=0
export CPUSET=0-63

bash 12_unirec_0_1b_inference/run_310p_full1651_aligned_k10_accuracy_background.sh
```

The device value is an example, not a reservation. Select a free device 0-3.
The launcher immediately prints absolute `RUN_ROOT`, `RUN_LOG`, and `PID`.
Give Luka the `RUN_LOG` path so he can use `tail -f`.

## Time-straggler monitoring

Inspect the run every 15-30 seconds. Do not wait silently:

```bash
while [[ ! -s "$RUN_ROOT/exit_code.txt" ]]; do
  date -Ins
  ps -p "$(cat "$RUN_ROOT/pid.txt")" -o pid,etime,stat,%cpu,%mem --no-headers || true
  grep -E 'UNIREC_310P_FULL1651_PHASE|UNIREC_310P_FULL1651_VISION_CACHE|UNIREC_310P_DECODE_CACHE_GATE|HEARTBEAT|page=.*1651|Traceback|ERROR' "$RUN_LOG" | tail -20
  sleep 15
done
```

Expected cache behavior before page inference:

- the vision cache reports `missing=0` for all ten graphs;
- the exact `decode_selfkv2048_cross1320_increfa_all_b128` directory contains
  one `compiled_module` and one OM;
- the decode gate passes on its first attempt;
- the decode gate is allowed only one attempt and must not change the OM
  inventory;
- no OM appears during inference;
- cache loading and first-call warmup can take time, but that is not a compile.

Stop and report immediately if a cache is missing, an OM inventory changes, no
page progress appears for 30 seconds after setup, or the NPU becomes unhealthy.
Do not automatically rerun.

Expected useful inference wall is roughly 9-10 minutes. Allow approximately
10-12 minutes including model/cache startup. Evaluation should use 64 workers
inside the explicit CPU affinity and normally adds a few minutes.

## Completion and report

Required completion markers:

```text
UNIREC_310P_FULL1651_OM_INVENTORY_UNCHANGED
UNIREC_310P_FULL1651_W4T8_EVAL: PASS
```

Paste these artifacts or their complete summaries:

```bash
cat "$RUN_ROOT/preflight.log"
cat "$RUN_ROOT/vision_cache_before.json"
cat "$RUN_ROOT/decode_cache_gate/passed.json"
cat "$RUN_ROOT/inference_process_wall_s.txt"
cat "$RUN_ROOT/process_wall_s.txt"
cat "$RUN_ROOT/final_report.txt"
cat "$RUN_ROOT/exit_code.txt"
```

Also report:

1. commit, physical NPU, CANN/Torch/Torch-NPU, CPU affinity, `/dev/shm`, and
   bare-metal available RAM;
2. graph warmup time per worker and confirmation that no OM was created;
3. 1,651 pages, crop/rejection counts, every vision bucket count, vision slot
   efficiency, and fallback count;
4. inference process wall, lifecycle, prefill, decode including ingress,
   decode graph, sequential-core pages/s, raw/effective tokens/s, and decode
   slot efficiency;
5. text edit, Page CDM, Page TEDS, reading-order edit, Overall, removed image
   tags, and evaluator timeout/exception counts;
6. absolute run root and log paths.

Do not impose token parity or a guessed accuracy threshold. Preserve and report
the measured output. The expected reference range is approximately 90.1-90.2
Overall, but the evaluator result is authoritative.
