#!/usr/bin/env bash
set -euo pipefail
cd /workspace/repos/paddle_ocr_vl_npu/tmp/09_persistent_page_engine/910b_table_cap_sweep_d88d6ed/cap2048/evaluation/work
/workspace/venvs/omnidocbench_py310/bin/python /workspace/repos/paddle_ocr_vl_npu/09_persistent_page_engine/scripts/run_omnidocbench_eval.py --config config.yaml --evaluator-root /workspace/repos/OmniDocBench_eval --match-workers 12 --teds-workers 8 --page-timeout-sec 120 --fallback-timeout-sec 180 --fallback-latex-timeout-sec 30
