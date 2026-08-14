# 310P UniRec full-1651 W4/T8 throughput and accuracy

Run the full OmniDocBench v1.6 benchmark once. The first-512 comparison already
passed; do not rerun it and do not run an A/B matrix. Do not edit tracked files,
commit, push, or create a branch.

## Fixed experiment

- all 1,651 sorted OmniDocBench pages, offset 0
- W4/T8, layout B2 eager FP32, threshold 0.5
- native layout and vision weights/depthwise operations
- compact uint8 HWC recognition input
- cross-KV 1320, self-KV/max length 2048
- continuous compiled IncreFA decode, B128
- strip embedded HTML image tags only from evaluator copies
- page matching/TEDS use 12 workers; CDM uses explicit `CDM_WORKERS=64`

Reject physical NPU 5 and 6. This run requires at least 64 GiB free `/dev/shm`
and 96 GiB available host RAM. The launcher prints every completed page so the
log can be followed directly.

## 910B2 reference

Source commit `78a65bc`, physical Ascend 910B2 NPU 7:

```text
pages=1651 crops=32109 rejected=0
lifecycle_s=391.501348
inference_process_wall_s=402
prefill_s=106.877412
decode_including_ingress_s=153.953668
sequential_core_s=260.989115
sequential_core_pg_s=6.325934
decode_raw_tok_s=20717.469
decode_effective_tok_s=18812.480
decode_slot_efficiency=0.908049
removed_image_tags=1545
text_edit=0.054328
page_cdm=0.921792
page_teds=0.838066
reading_edit=0.145485
official_overall=90.1843
match_fallbacks=0
teds_timeouts=0
cdm_timeouts=0
```

## Run

```bash
set -eo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main
test -z "$(git diff --name-only)"
test -z "$(git diff --cached --name-only)"

source npu-setup
case ",${ASCEND_RT_VISIBLE_DEVICES:?}," in
  *,5,*|*,6,*) echo "Select a free physical NPU other than 5 or 6" >&2; exit 1 ;;
esac

export PYTHON_BIN="${PYTHON_BIN:?set the passed 310P inference Python}"
export MODEL="${MODEL:?set the existing unirec-0.1b directory}"
export LAYOUT_MODEL="${LAYOUT_MODEL:?set PP-DocLayoutV2_safetensors}"
export OPENOCR_ROOT="${OPENOCR_ROOT:?set the passed OpenOCR checkout}"
export IMAGES_DIR="${IMAGES_DIR:?set OmniDocBench images}"
export DATASET_JSON="${DATASET_JSON:?set the full OmniDocBench.json}"
export COMPILE_CACHE="${COMPILE_CACHE:?set the existing production cache parent}"
export EVALUATOR_ROOT="${EVALUATOR_ROOT:?set OmniDocBench evaluator checkout}"
export EVAL_PYTHON="${EVAL_PYTHON:?set the existing CDM-capable evaluator Python}"
export CDM_WORKERS="${CDM_WORKERS:-64}"

bash 12_unirec_0_1b_inference/run_310p_full1651_w4t8_accuracy_background.sh
```

The launcher prints `RUN_ROOT`, `RUN_LOG`, and `PID`. Send Luka the absolute
`RUN_LOG` immediately and follow it with `tail -f`. Do not restart a quiet
worker while its heartbeat and owned process are alive.

## Required report

Wait for `exit_code.txt`. A pass requires exit code zero and
`UNIREC_310P_FULL1651_W4T8_EVAL: PASS`. Return that complete line plus:

1. absolute `RUN_ROOT` and `RUN_LOG`;
2. project/evaluator commits, physical NPU, CANN and torch_npu versions;
3. `output/run_summary.json`, `evaluation_image_tags_stripped/transform_summary.json`,
   and `evaluation_image_tags_stripped/full_eval_summary.json`;
4. process wall, lifecycle, prefill, decode, sequential-core pages/s, token/s,
   crop count, and accuracy components;
5. ratios/deltas against the 910B2 reference above.

If inference OOMs or evaluation fails, preserve the run root and exact traceback.
Do not lower B128, shrink either cache, change the threshold, skip pages, or
substitute sample CDM for page CDM.
