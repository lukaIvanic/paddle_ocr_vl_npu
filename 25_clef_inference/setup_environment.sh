#!/usr/bin/env bash
# Isolated Transformers overrides; never replace the matching torch/torch-npu.
set -euo pipefail
base=/usr/local/python3.12.13/bin/python3
target=/workspace/venvs/clef_transformers_py312
if [[ ! -x "$target/bin/python" ]]; then
    "$base" -m venv --system-site-packages "$target"
fi
"$target/bin/python" -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    'transformers==5.17.0' 'accelerate==1.15.0'
"$target/bin/python" -m pip show torch torch-npu transformers accelerate safetensors
