#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ENV="${SOURCE_ENV:-/workspace/venvs/mineru_pro_vllm_py312}"
TARGET_ENV="${TARGET_ENV:-/workspace/venvs/mineru_vllm_ascend_exp17_py312}"

if [[ ! -x "$SOURCE_ENV/bin/python" ]]; then
  echo "source environment is missing: $SOURCE_ENV" >&2
  exit 1
fi
if [[ -e "$TARGET_ENV" ]]; then
  echo "target environment already exists: $TARGET_ENV" >&2
  exit 1
fi

echo "[clone] source=$SOURCE_ENV"
echo "[clone] target=$TARGET_ENV"
cp -a "$SOURCE_ENV" "$TARGET_ENV"

set +u
source npu-setup
set -u
"$TARGET_ENV/bin/python" \
  "$SCRIPT_DIR/verify_environment.py" \
  --json-output "$TARGET_ENV/experiment17_environment.json"

echo "FRESH_ENVIRONMENT_READY $TARGET_ENV"
