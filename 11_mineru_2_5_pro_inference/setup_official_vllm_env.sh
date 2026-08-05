#!/usr/bin/env bash
set -euo pipefail

BASE_PYTHON="${BASE_PYTHON:-/usr/local/python3.12.13/bin/python3}"
VENV="${VENV:-/workspace/venvs/mineru_pro_vllm_py312}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/3] Creating vLLM-compatible environment at $VENV"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV"
fi

echo "[2/3] Installing MinerU utilities without changing the base vLLM stack"
"$VENV/bin/python" -m pip install --no-deps \
  -r "$SCRIPT_DIR/requirements_official_vllm.txt"

echo "[3/3] Verifying the complete vLLM-Ascend 0.21 contract"
"$VENV/bin/python" - <<'PY'
from importlib.metadata import version

expected = {
    "vllm": "0.21.0+empty",
    "vllm-ascend": "0.21.0rc1",
    "torch": "2.10.0+cpu",
    "torch-npu": "2.10.0",
    "transformers": "5.5.4",
    "mineru-vl-utils": "1.0.5",
    "httpx-retries": "0.6.0",
}
actual = {name: version(name) for name in expected}
assert actual == expected, (actual, expected)
print("ENVIRONMENT_READY", actual)
PY
