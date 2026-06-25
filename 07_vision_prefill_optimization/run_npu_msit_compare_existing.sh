#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python}"
export OUT_ROOT="${OUT_ROOT:?set OUT_ROOT to an existing outputs/msit_ge_fx_* directory}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

GE_PATH="${GE_PATH:-${OUT_ROOT}/ge/msit_ge_dump}"
FX_PATH="${FX_PATH:-${OUT_ROOT}/fx/msit_fx_dump}"
FX_COMPARE_PATH="${FX_COMPARE_PATH:-${FX_PATH}}"
if [[ -d "${FX_PATH}/data_dump" && -z "${FX_COMPARE_PATH_OVERRIDE:-}" ]]; then
  FX_COMPARE_PATH="${FX_PATH}/data_dump"
elif [[ -d "${OUT_ROOT}/fx/data_dump" && -z "${FX_COMPARE_PATH_OVERRIDE:-}" ]]; then
  FX_COMPARE_PATH="${OUT_ROOT}/fx/data_dump"
fi

if [[ ! -d "${GE_PATH}" ]]; then
  echo "EXP07_MSIT_COMPARE_EXISTING ERROR missing GE dump directory: ${GE_PATH}" >&2
  exit 2
fi
if [[ ! -d "${FX_COMPARE_PATH}" ]]; then
  echo "EXP07_MSIT_COMPARE_EXISTING ERROR missing FX dump directory: ${FX_COMPARE_PATH}" >&2
  exit 2
fi

MSIT_CMD=()
if [[ -n "${MSIT_BIN:-}" ]]; then
  MSIT_CMD=("${MSIT_BIN}")
elif command -v msit >/dev/null 2>&1; then
  MSIT_CMD=("$(command -v msit)")
else
  PYTHON_DIR="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"
  if [[ -x "${PYTHON_DIR}/msit" ]]; then
    MSIT_CMD=("${PYTHON_DIR}/msit")
  elif "${PYTHON_BIN}" -c "import components.__main__" >/dev/null 2>&1; then
    MSIT_CMD=("${PYTHON_BIN}" -m components.__main__)
  fi
fi

echo "EXP07_MSIT_COMPARE_EXISTING PYTHON_BIN=${PYTHON_BIN}"
echo "EXP07_MSIT_COMPARE_EXISTING OUT_ROOT=${OUT_ROOT}"
echo "EXP07_MSIT_COMPARE_EXISTING GE_PATH=${GE_PATH}"
echo "EXP07_MSIT_COMPARE_EXISTING FX_COMPARE_PATH=${FX_COMPARE_PATH}"
echo "EXP07_MSIT_COMPARE_EXISTING PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION}"

if [[ "${#MSIT_CMD[@]}" -eq 0 ]]; then
  echo "EXP07_MSIT_COMPARE_EXISTING ERROR could not find msit CLI" >&2
  echo "EXP07_MSIT_COMPARE_EXISTING INSTALL_HINT=${PYTHON_BIN} -m pip install msit && ${PYTHON_BIN%/python}/msit install llm && ${PYTHON_BIN%/python}/msit check llm" >&2
  exit 2
fi

echo "EXP07_MSIT_COMPARE_EXISTING MSIT_CMD=${MSIT_CMD[*]}"
"${PYTHON_BIN}" - <<'PY' || true
try:
    import google.protobuf
    print("EXP07_MSIT_COMPARE_EXISTING PROTOBUF_VERSION=" + str(google.protobuf.__version__))
except Exception as exc:
    print("EXP07_MSIT_COMPARE_EXISTING PROTOBUF_VERSION_ERROR=" + exc.__class__.__name__ + ": " + str(exc))
PY

if ! "${MSIT_CMD[@]}" llm compare \
  --my-path "${GE_PATH}" \
  --golden-path "${FX_COMPARE_PATH}" \
  --output "${OUT_ROOT}/msit_compare"; then
  echo "EXP07_MSIT_COMPARE_EXISTING ERROR msit llm compare failed" >&2
  echo "EXP07_MSIT_COMPARE_EXISTING HINT protobuf pb2 errors often need one of:" >&2
  echo "EXP07_MSIT_COMPARE_EXISTING HINT   export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python" >&2
  echo "EXP07_MSIT_COMPARE_EXISTING HINT   ${PYTHON_BIN} -m pip install 'protobuf==3.20.2'" >&2
  exit 3
fi

echo "EXP07_MSIT_COMPARE_EXISTING OUTPUT_TREE"
find "${OUT_ROOT}/msit_compare" -maxdepth 3 -type f | sort | head -n 80
