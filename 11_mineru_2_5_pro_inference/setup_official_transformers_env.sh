#!/usr/bin/env bash
set -euo pipefail

BASE_PYTHON="${BASE_PYTHON:-/usr/local/python3.12.13/bin/python3}"
VENV="${VENV:-/workspace/venvs/mineru_pro_transformers_py312}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/3] Creating isolated environment at $VENV"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV"
fi

echo "[2/3] Installing pinned official MinerU Transformers dependencies"
"$VENV/bin/python" -m pip install --upgrade \
  -r "$SCRIPT_DIR/requirements_official_transformers.txt"

echo "[3/3] Verifying imports and exact versions"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$VENV/bin/python" - <<'PY'
import accelerate
import huggingface_hub
import mineru_vl_utils
import transformers

assert mineru_vl_utils.__version__ == "1.0.5"
assert transformers.__version__ == "4.57.6"
assert accelerate.__version__ == "1.14.0"
assert huggingface_hub.__version__ == "0.36.2"
print(
    "ENVIRONMENT_READY "
    f"mineru_vl_utils={mineru_vl_utils.__version__} "
    f"transformers={transformers.__version__} "
    f"accelerate={accelerate.__version__} "
    f"huggingface_hub={huggingface_hub.__version__}"
)
PY
