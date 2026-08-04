#!/usr/bin/env bash
cd /workspace/repos/paddle_ocr_vl_npu
source npu-setup
set -euo pipefail
export ASCEND_RT_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1
printf "start_utc=%s\n" "$(date -u +%FT%TZ)"
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python 09_persistent_page_engine/scripts/run_omnidocbench.py --dataset-json /workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine/910b_table_cap_sweep_d88d6ed/OmniDocBench_table_pages_first.json --images-dir /workspace/datasets/OmniDocBench/images --layout-model /workspace/models/PP-DocLayoutV3_safetensors --recognizer-model /workspace/models/PaddleOCR-VL-1.6 --offset 0 --limit 458 --batch-size 32 --cache-length 4096 --max-new-tokens 4096 --preprocessor-min-pixels 28224 --preprocessor-max-pixels 1003520 --decode-backend torchair --decode-optimization combined_apply_static_actual --vision-backend torchair --vision-attention prompt_flash_attention --vision-buckets 128,256,384,512,640,768,1408,1920,2944,4992 --vision-promptfa-align-128 --vision-padding bucket --vision-packing greedy --vision-pack-target 1920 --vision-router-lookahead 32 --text-buckets 32,64,96,128,160,176,192,208,224,256,320,384,448,576,640,768,896,1024,1152,1280,1312 --text-packing production_group --text-pack-buckets 128,256,512,1024 --text-pack-max-members 32 --layout-device npu --no-layout-graph-capture --layout-workers 8 --preprocess-all-pages-first --no-timeline --table-preprocessor-max-pixels 602112 --output-dir /workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine/910b_table_cap_sweep_d88d6ed/cap3072/output
printf "exit_code=0\nfinished_utc=%s\n" "$(date -u +%FT%TZ)"
