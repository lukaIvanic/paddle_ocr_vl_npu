#!/usr/bin/env bash

set -euo pipefail
cd /workspace/repos/paddle_ocr_vl_npu
source npu-setup

/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  12_unirec_0_1b_inference/run_opendoc_batched_unirec.py \
  --openocr-root /workspace/repos/OpenOCR \
  --model-path /workspace/models/unirec-0.1b \
  --layout-model /root/.cache/openocr/PP_DoclayoutV2_onnx/PP-DoclayoutV2.onnx \
  --layout-backend transformers_npu \
  --layout-transformers-model /workspace/models/PP-DocLayoutV2_safetensors \
  --layout-dtype float32 \
  --stock-encoder /root/.cache/openocr/unirec_0_1b_onnx/unirec_encoder.onnx \
  --stock-decoder /root/.cache/openocr/unirec_0_1b_onnx/unirec_decoder.onnx \
  --stock-tokenizer-mapping /root/.cache/openocr/unirec_0_1b_onnx/unirec_tokenizer_mapping.json \
  --input /workspace/datasets/OmniDocBench/images \
  --output-dir tmp/12_unirec_0_1b_inference/opendoc_fixed_b4_32p_warm_control_7477463/output \
  --device npu:0 --dtype float16 --max-length 256 \
  --decode-mode compiled --compile-backend torchair \
  --compile-cache-dir .runtime_cache/12_unirec_0_1b_inference/opendoc_batched_decode_a372dbf \
  --decode-batch-size 4 --decode-scheduling fixed --limit 32
