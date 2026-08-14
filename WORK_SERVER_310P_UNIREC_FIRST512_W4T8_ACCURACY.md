# 310P UniRec first-512 W4/T8 throughput and accuracy

Run one exact cross-chip comparison against the completed 910B2 reference below.
Do not edit tracked files, create a branch, commit, or push. Pull the commit Luka
names and use the committed background launcher. Do not run an A/B matrix.

## Fixed experiment

- OmniDocBench sorted-image offset `0`, limit `512`
- four process workers, eight recognition-preprocess threads per worker
- layout B2, eager FP32, threshold `0.5`
- native layout and vision weights/depthwise operations
- compact uint8 HWC recognition input
- cross-KV `1320`, self-KV/max length `2048`
- continuous compiled IncreFA decode, B128
- embedded HTML image tags stripped only from evaluator copies
- official Overall uses page text accuracy, page CDM, and page TEDS

The launcher rejects physical NPU 5 and 6. Use one other free 310P. It also
requires at least 16 GiB free `/dev/shm` and 32 GiB available host RAM.

## 910B2 reference

Project source commit for inference: `78a65bc` plus the evaluation launcher
commit named by Luka. Physical device: Ascend 910B2 NPU 7.

```text
pages=512 crops=7872
prefill_s=27.778520
decode_including_ingress_s=44.040900
sequential_core_s=72.118517
sequential_core_pg_s=7.099425
decode_raw_tok_s=20090.660
decode_effective_tok_s=15685.156
decode_slot_efficiency=0.780712
removed_image_tags=301
text_edit=0.040956
page_cdm=0.947388
page_teds=0.920905
reading_edit=0.125319
official_overall=94.2446
match_fallbacks=0
teds_timeouts=0
cdm_timeouts=0
```

The 94.2446 score is for this first-512 subset, not the full 1,651-page
benchmark. Compare 310P only against this same subset. The official notebook
uses page-level CDM in Overall:
<https://github.com/opendatalab/OmniDocBench/blob/main/tools/generate_result_tables.ipynb>.

## Run

Use the exact paths from the previously passed 310P UniRec and evaluator lanes.
The assignments below are deliberately environment-specific; resolve them on
the work server instead of copying any 910B path.

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
# Do not derive this from nproc: this container reports nproc=1 despite
# supporting the passed process-isolated evaluator lane.
export CDM_WORKERS="${CDM_WORKERS:-64}"

bash 12_unirec_0_1b_inference/run_310p_first512_w4t8_accuracy_background.sh
```

The launcher selects and validates the frozen OmniDocBench v1.6 CDM runtime
before it starts inference. It rejects ambient TeX, missing CJK/xcolor
resources, or wrong ImageMagick/Ghostscript versions. It evaluates with a clean
detached clone of the required evaluator commit, without modifying the existing
checkout.

The launcher prints `RUN_ROOT`, `RUN_LOG`, and `PID`. Immediately send Luka the
absolute `RUN_LOG`, then follow it with:

```bash
tail -f /absolute/path/printed/as/RUN_LOG
```

The log prints every completed decode page. Do not diagnose a stall from quiet
compiler/cache loading alone; use the 15-second heartbeat and process/NPU state.
Do not restart if the owned process is still active.

## Required report

Wait for `exit_code.txt`. A pass requires exit code zero and the line
`UNIREC_310P_FIRST512_W4T8_EVAL: PASS`. Return:

1. that complete line;
2. absolute `RUN_ROOT` and `RUN_LOG`;
3. project commit, evaluator commit, physical NPU, CANN and torch_npu versions;
4. `output/run_summary.json`, `evaluation_image_tags_stripped/transform_summary.json`,
   and `evaluation_image_tags_stripped/full_eval_summary.json`;
5. process wall time and the five comparison values below.

```text
310P pg_s / 7.099425
310P text_edit - 0.040956
310P page_cdm - 0.947388
310P page_teds - 0.920905
310P overall - 94.2446
```

If inference OOMs or CDM tooling is missing, preserve the run root and exact
traceback and stop. Do not lower B128, shrink either cache, change threshold,
skip pages, install packages, or substitute sample CDM for page CDM.
