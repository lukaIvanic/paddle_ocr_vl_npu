#!/usr/bin/env bash
set -euo pipefail
# Isolated package overrides; inherit the existing matching torch/torch_npu.
base=/usr/local/python3.12.13/bin/python3
target=/workspace/venvs/decision2_eval_py312
if [[ ! -x "$target/bin/python" ]]; then
    "$base" -m venv --system-site-packages "$target"
fi
"$target/bin/python" -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    'transformers==5.17.0'
"$target/bin/python" -m pip show torch torch-npu transformers
