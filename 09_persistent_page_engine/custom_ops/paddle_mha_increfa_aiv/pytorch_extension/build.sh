#!/usr/bin/env bash

# Non-interactive SSH does not source the CANN and torch_npu environment.
source npu-setup
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/python3.12.13/bin/python3}"

cd "$SCRIPT_ROOT"
"$PYTHON_BIN" setup.py build_ext --inplace

mapfile -t EXTENSION_SOS < <(
    find "$SCRIPT_ROOT/paddle_mha_increfa_aiv_eager" -maxdepth 1 -type f \
        -name '_C*.so' -print | sort
)
if [[ "${#EXTENSION_SOS[@]}" != "1" ]]; then
    echo "ERROR: expected one eager extension shared object" >&2
    printf '%s\n' "${EXTENSION_SOS[@]}" >&2
    exit 2
fi

echo "PADDLE_MHA_INCREFA_AIV_EAGER_SO=${EXTENSION_SOS[0]}"
