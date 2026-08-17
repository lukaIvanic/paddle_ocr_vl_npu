# 310P full-1651 optimized UniRec accuracy run

Run only the optimized full benchmark. Do not run a native A/B lane.

## Purpose

Measure the full 1,651-page OmniDocBench v1.6 pipeline on one 310P with the
accuracy-safe layout and the optimized recognition vision encoder:

- W4, eight recognition preprocessing threads per worker;
- eager FP32 native layout, batch size 2, threshold 0.5;
- four-page vision lookahead and `310p_k10_l4_all` buckets;
- `constant_grouped_all` focal depthwise weights and `torchair_internal`
  vision weights;
- no eager vision fallback;
- B128 continuous IncreFA decode, cross-KV 1320, self-KV/max length 2048;
- embedded HTML image tags removed only from evaluator copies;
- clean OmniDocBench evaluator and frozen CDM runtime.

The compiled FP16 layout lane is deliberately excluded. Its full 910B2 run
changed the layout crop set from 32,109 to 32,038 and reduced Overall from the
known-good 90.18% to 89.80%. The lane in this brief restored exactly 32,109
crops and passed full evaluation.

## Pull and launch

Use the existing validated 310P paths. Preserve the final venv executable or
`python_nosym`; do not apply `readlink -f` to `PYTHON_BIN`.

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

# Export the same validated values used by the previous full-1651 run:
export PYTHON_BIN="${PYTHON_BIN:?validated 310P UniRec venv executable}"
export MODEL="${MODEL:?unirec-0.1b model directory}"
export LAYOUT_MODEL="${LAYOUT_MODEL:?PP-DocLayoutV2 model directory}"
export OPENOCR_ROOT="${OPENOCR_ROOT:?OpenOCR checkout}"
export IMAGES_DIR="${IMAGES_DIR:?OmniDocBench v1.6 images}"
export DATASET_JSON="${DATASET_JSON:?OmniDocBench.json}"
export COMPILE_CACHE="${COMPILE_CACHE:?existing production cache parent}"
export EVALUATOR_ROOT="${EVALUATOR_ROOT:?clean OmniDocBench evaluator source}"
export EVAL_PYTHON="${EVAL_PYTHON:?frozen-runtime evaluator Python}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:?one free 310P device, 0-3}"

# Explicit CPU and evaluator settings. Do not derive these from nproc.
export CPUSET="${CPUSET:-0-63}"
export LAYOUT_CPU_THREADS=16
export MATCH_WORKERS=64
export TEDS_WORKERS=64
export CDM_WORKERS=64

bash 12_unirec_0_1b_inference/run_310p_full1651_k10_l4_accuracy_background.sh
```

The launcher prints absolute `RUN_ROOT`, `RUN_LOG`, and `PID` values. Paste the
absolute `RUN_LOG` path so Luka can use `tail -f`. The job runs in the
background. Follow it until `RUN_ROOT/exit_code.txt` exists.

The launcher records `/dev/shm` and bare-metal `MemAvailable`, but intentionally
does not reject a host with less than 64 GiB free `/dev/shm`. If the real run
fails from memory pressure, preserve the actual failure and NPU logs.

## Live checks

Report immediately if any of these occur:

- a compiler process starts instead of loading the existing ten vision graphs;
- no page-progress line appears for 30 seconds after worker setup;
- any vision fallback row appears;
- the process exits or the NPU becomes unhealthy.

Do not edit tracked files, create a branch, commit, or push. If a code issue
appears, report the exact command, log path, first causal error, and minimal
proposed change.

## 910B2 reference from the rehearsed command

Physical device: one Ascend 910B2, physical NPU 3. Project source contained the
same UniRec implementation as this handoff.

Inference:

- process wall: 354 s;
- lifecycle: 344.791865 s;
- prefill: 109.719028 s, 15.047527 pages/s;
- decode including ingress: 153.560115 s, 10.751490 pages/s;
- decode graph: 121.010029 s;
- sequential prefill plus decode: 263.435782 s, 6.267182 pages/s;
- raw decode: 20,502.631 token slots/s;
- effective decode: 18,595.583 tokens/s;
- crops: 32,109; rejected: 0;
- compiled vision real/physical rows: 32,109 / 33,728;
- vision slot efficiency: 95.1998%; fallback rows: 0.

Evaluation used 64 match workers, 64 TEDS workers, and 64 CDM workers:

- match plus TEDS wall: 126 s;
- CDM wall: 106 s;
- total evaluation wall: 236 s;
- removed evaluator-only image tags: 1,545;
- text edit: 0.0538006054;
- Page CDM: 0.9212068348;
- Page TEDS: 0.8396283142;
- reading-order edit: 0.1454105419;
- Overall: 90.2344848%;
- zero page, TEDS, or CDM timeouts/exceptions.

The older accuracy anchor was 90.1843% Overall and 6.2156 warmed sequential
pages/s. This lane is accuracy-safe, but its 910B2 sequential speedup is small:
about 0.83%.

The compact source artifacts are committed under
`12_unirec_0_1b_inference/references/unirec_910b_full1651_accuracy_safe_k10_l4_20260817/`.
Use its `run_summary.json`, `full_eval_summary.json`,
`predictions_quick_match_stage_execution.json`, and transform/timing files for
direct field-by-field comparison. Do not reconstruct the 910B2 reference from
rounded values in this brief.

## Required final report

Paste:

1. commit, physical 310P, CANN/Torch/Torch-NPU, CPU affinity, `/dev/shm`, and
   bare-metal available RAM;
2. worker setup/warmup time and whether any graph compiled;
3. crop count, rejected crops, every vision bucket count, slot efficiency, and
   fallback count;
4. process wall, lifecycle, prefill, decode including ingress, decode graph,
   sequential-core time/pages/s, raw/effective tokens/s, and slot efficiency;
5. text edit, Page CDM, Page TEDS, reading-order edit, Overall, removed image
   tags, and all evaluator timeout/exception counts;
6. absolute run root and log paths.

Completion requires `exit_code.txt=0`, `run_summary.status=ok`, 1,651 output
pages, zero rejected crops, zero vision fallbacks, and a complete evaluator
report. Do not impose an arbitrary accuracy hard gate; report the measured
cross-chip difference against the 910B2 values above.
