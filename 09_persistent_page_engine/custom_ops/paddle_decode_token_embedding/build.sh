#!/bin/bash
set -euo pipefail

: "${ASCEND_HOME_PATH:?source the CANN environment first}"

script_path=$(realpath "$(dirname "$0")")
build_dir="$script_path/build_out"
mkdir -p "$build_dir"
# CANN code generation is not reliably incremental across operator-definition
# changes.  Remove only this operator's generated build tree before rebuilding.
find "$build_dir" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +

opts=$(python3 "$ASCEND_HOME_PATH/tools/tikcpp/ascendc_kernel_cmake/fwk_modules/util/preset_parse.py" "$script_path/CMakePresets.json")
cmake_version=$(cmake --version | awk '/cmake version/ {print $3}')
if [ "$cmake_version" \< "3.19.0" ]; then
    cmake -S "$script_path" -B "$build_dir" $opts
else
    cmake -S "$script_path" -B "$build_dir" --preset=default
fi
cmake --build "$build_dir" --target binary package -j"$(nproc)"
