#!/usr/bin/env bash
set -euo pipefail

# Inherit the container's torch/torch_npu binary stack; override only HF/data
# packages locally. This environment is intentionally NOT for vLLM.
BASE_PYTHON="${BASE_PYTHON:-/usr/local/python3.12.13/bin/python3}"
COLQWEN_ENV="${COLQWEN_ENV:-/workspace/venvs/colqwen3_hf_py312}"
if [[ ! -e "$COLQWEN_ENV" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$COLQWEN_ENV"
fi
test -x "$COLQWEN_ENV/bin/python"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
mkdir -p "$TMPDIR"
"$COLQWEN_ENV/bin/python" -m pip install \
  'transformers==4.57.1' 'huggingface-hub==0.36.2' \
  'tokenizers==0.22.2' 'pyarrow==21.0.0'
"$COLQWEN_ENV/bin/python" -c \
  'import transformers, pyarrow; print("HF_BASELINE_ENV", transformers.__version__, pyarrow.__version__)'
