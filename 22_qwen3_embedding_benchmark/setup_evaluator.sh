#!/usr/bin/env bash
set -euo pipefail
# CPU tokenizer/retrieval evaluator only; NPU inference stays in base vLLM env.
BASE_PYTHON=${BASE_PYTHON:-/usr/local/python3.12.13/bin/python3}
EVAL_ENV=${EVAL_ENV:-/workspace/venvs/qwen3_embedding_eval_py312}
if [[ ! -e "$EVAL_ENV" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$EVAL_ENV"
fi
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/workspace/.cache/pip}
"$EVAL_ENV/bin/python" -m pip install \
  'mteb==1.38.9' 'datasets==2.21.0' 'transformers==4.51.3' \
  'sentence-transformers==4.1.0' 'huggingface-hub==0.36.2' \
  'tokenizers==0.21.4' 'pyarrow==21.0.0'
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$EVAL_ENV/bin/python" -c \
  'import mteb, transformers, datasets; print("EVALUATOR_READY", mteb.__version__, transformers.__version__, datasets.__version__)'

