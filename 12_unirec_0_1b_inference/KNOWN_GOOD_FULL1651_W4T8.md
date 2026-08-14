# Known-good UniRec OmniDocBench v1.6 run

This is the reproducibility anchor for the optimized full-page UniRec run and
its accuracy evaluation. Use it before changing performance code. Do not infer
the accuracy contract from an older temporary artifact or an ambient evaluator
installation.

## Version anchors

- Inference implementation and complete 910B2 reference output: project commit
  `470d8a6d01d4682fa7e15aad915e5ddf697e2fe0`.
- Canonical evaluator/runtime repair and same-host CDM replay: project commit
  `e267c3484a3653a7a378d01fe3e3595ec0c1b5a0`.
- OmniDocBench evaluator source: clean detached commit
  `2b161d010d2e3aff77a0edef359ea3a6411d23cd`.
- Complete 910B2 textual evidence:
  `references/unirec_full1651_910b_470d8a6_text_outputs.tar.gz`.
- The full launcher records the actual checked-out project commit, physical NPU,
  CANN, Torch, and Torch-NPU versions in `RUN_ROOT/preflight.log`. Preserve that
  file with every result.

Later project commits may contain the same inference implementation plus safer
evaluation launchers. For an exact historical comparison, use the anchors
above. For a new run, use current `main` and record its commit.

## Fixed inference contract

- OmniDocBench v1.6: all 1,651 sorted pages, offset 0.
- OpenDoc `topdu/unirec-0.1b` `model.pth`, not the 1217 safetensors checkpoint.
- PP-DocLayoutV2.
- Four page workers and eight recognition preprocessing threads per worker:
  W4/T8.
- Layout: eager FP32, batch size 2, threshold 0.5, native weights and native
  depthwise operations.
- Recognition: FP16, compact uint8 HWC input, four-page vision lookahead,
  native vision weights and native depthwise operations.
- Cross-KV length 1320. Crops above the supported capacity are rejected rather
  than resized.
- Self-KV and maximum length 2048.
- Continuous compiled IncreFA decode, batch size 128, two warmup passes.
- One physical NPU. Never use physical NPU 5 or 6.
- At least 64 GiB `/dev/shm` and 96 GiB available host RAM.

The canonical command is assembled by
`run_310p_full1651_w4t8_accuracy_background.sh` and saved verbatim as
`RUN_ROOT/command.sh`. The script name is historical; the same launcher was
rehearsed on 910B2 and is the cross-chip W4/T8 accuracy contract.

## Full inference plus evaluation

Resolve the environment-specific paths first. On 910B2, start with
`source npu-setup`; on 310P, use the validated equivalents supplied on that
host.

```bash
set -eo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main
test -z "$(git diff --name-only)"
test -z "$(git diff --cached --name-only)"

source npu-setup
case ",${ASCEND_RT_VISIBLE_DEVICES:?}," in
  *,5,*|*,6,*) echo "Do not use physical NPU 5 or 6" >&2; exit 1 ;;
esac

export PYTHON_BIN="${PYTHON_BIN:?validated UniRec inference Python}"
export MODEL="${MODEL:?OpenDoc unirec-0.1b model.pth directory}"
export LAYOUT_MODEL="${LAYOUT_MODEL:?PP-DocLayoutV2_safetensors directory}"
export OPENOCR_ROOT="${OPENOCR_ROOT:?OpenOCR checkout}"
export IMAGES_DIR="${IMAGES_DIR:?OmniDocBench v1.6 images directory}"
export DATASET_JSON="${DATASET_JSON:?OmniDocBench v1.6 JSON}"
export COMPILE_CACHE="${COMPILE_CACHE:?existing production compile-cache parent}"
export EVALUATOR_ROOT="${EVALUATOR_ROOT:?OmniDocBench evaluator checkout}"
export EVAL_PYTHON="${EVAL_PYTHON:?CDM-capable evaluator Python}"
export CDM_WORKERS="${CDM_WORKERS:-64}"

bash 12_unirec_0_1b_inference/run_310p_full1651_w4t8_accuracy_background.sh
```

The launcher immediately prints `RUN_ROOT`, `RUN_LOG`, and `PID`. Follow
`RUN_LOG` with `tail -f`. Wait for `RUN_ROOT/exit_code.txt`; zero plus the line
`UNIREC_310P_FULL1651_W4T8_EVAL: PASS` is the completion gate.

The launcher performs both inference and evaluation. Do not launch a separate
ad-hoc evaluator unless the recorded evaluation must be repaired.

## Evaluation contract

- Keep the original generated Markdown unchanged.
- Strip embedded HTML `<img>` tags only from evaluator copies. This is required
  for the published text metric; otherwise image markup is incorrectly scored
  as recognized text.
- Page matching and TEDS use 12 workers.
- CDM uses 64 workers unless the host has an explicit process limit.
- Use a clean evaluator checkout at commit `2b161d0...`; do not score with a
  dirty `latex2bbox_color.py`.
- Use the frozen runtime selected by
  `09_persistent_page_engine/scripts/omnidocbench_eval_env.sh`:
  TeX Live 2025/pdfTeX 1.40.28, ImageMagick 7.1.1-47, and Ghostscript 9.55.0,
  including the validated CJK and xcolor resources.
- Report page-weighted CDM and page-weighted TEDS. Do not substitute the
  formula-sample CDM mean.
- Official Overall is
  `((1 - page_text_edit) + page_cdm + page_teds) / 3`.

## Canonical CDM repair or cross-host audit

Use this only when inference already completed and CDM was scored with an
unknown or invalid runtime. It does not use an NPU and does not rerun inference,
page matching, or TEDS. It replays both the 310P and supplied 910B2 matched
formula inputs on the same host and runtime.

```bash
set -eo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main

export RUN_ROOT="${RUN_ROOT:?completed full-1651 run root}"
export EVAL_PYTHON="${EVAL_PYTHON:?evaluator Python}"
export EVALUATOR_ROOT="${EVALUATOR_ROOT:?evaluator checkout}"
export CDM_WORKERS="${CDM_WORKERS:-64}"
export REPAIR_RUNTIME=1

bash 12_unirec_0_1b_inference/run_310p_unirec_cdm_same_host_audit.sh
```

The audit prints `AUDIT_ROOT`, `RUN_LOG`, and `PID`. Preserve
`same_host_audit.json`, both replay outputs, runtime fingerprints, and the final
`UNIREC_CDM_SAME_HOST_AUDIT PASS` line.

## Recorded accuracy

Official OpenDoc OmniDocBench v1.6 target:

| Metric | Official |
|---|---:|
| Page text edit (lower is better) | 0.0490 |
| Page text accuracy | 95.10% |
| Page CDM | 93.02% |
| Page TEDS | 83.88% |
| Overall | 90.67% |

Known-good 910B2 full run, with canonical CDM replay rounded to the latest
reported result:

| Metric | 910B2 |
|---|---:|
| Page text edit | 0.054328 |
| Page text accuracy | 94.5672% |
| Page CDM | 92.171% |
| Page TEDS | 83.8066% |
| Overall | 90.18% |

The saved original 910B2 CDM was `0.921792`; the canonical same-host replay was
reported as approximately `0.92171`. This difference is immaterial to Overall.

Canonical 310P full result reported after runtime repair:

| Metric | 310P |
|---|---:|
| Page text edit | approximately 0.0551 |
| Page text accuracy | 94.49% |
| Page CDM | 92.179% |
| Page TEDS | 83.75% |
| Overall | 90.13% |

The canonical same-host CDM values were approximately `0.92179` for 310P and
`0.92171` for the supplied 910B2 output. This establishes cross-chip formula
accuracy parity. The 310P and 910B2 Overall scores differ by about 0.05 points.

## Required artifacts

Never report a run from only the terminal sentence. Preserve at minimum:

- `preflight.log`
- `command.sh`
- `inference_process_wall_s.txt`
- `output/run_summary.json`
- `output/recognition_trace.jsonl`
- `evaluation_image_tags_stripped/transform_summary.json`
- `evaluation_image_tags_stripped/full_eval_summary.json`
- `evaluation_image_tags_stripped/work/result/`
- `evaluation_image_tags_stripped/cdm/`
- `final_report.txt`
- `exit_code.txt`

For a repaired or cross-host CDM score, also preserve the complete audit root.
