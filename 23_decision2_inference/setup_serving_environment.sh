#!/usr/bin/env bash
set -euo pipefail
target=/workspace/venvs/decision2_vllm_py312
if [[ ! -x "$target/bin/python" ]]; then
    /usr/local/python3.12.13/bin/python3 -m venv --system-site-packages "$target"
fi
"$target/bin/python" -m pip install --no-deps --no-build-isolation -e "$(dirname "$0")"
"$target/bin/python" -m pip show vllm vllm-ascend transformers decision2-ascend-experiment
