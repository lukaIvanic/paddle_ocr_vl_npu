#!/usr/bin/env bash
set -euo pipefail
: "${ASCEND_HOME_PATH:?source the CANN build environment first}"
root=$(realpath "$(dirname "$0")")
build=$(realpath -m "${1:?pass a new external build directory}")
[[ ! -e "$build" ]] || { echo "Refusing existing build directory: $build" >&2; exit 1; }
mkdir -p "$build"
cmake -S "$root" -B "$build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$ASCEND_HOME_PATH/tools/tikcpp/ascendc_kernel_cmake" \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DASCEND_COMPUTE_UNIT=ascend910b \
  -DASCEND_PYTHON_EXECUTABLE=/usr/local/python3.12.13/bin/python3 \
  -Dvendor_name=rwkv_reference \
  -DENABLE_BINARY_PACKAGE=ON -DENABLE_SOURCE_PACKAGE=ON -DENABLE_TEST=OFF
cmake --build "$build" --target binary package -j4
packages=("$build"/*.run)
[[ ${#packages[@]} -eq 1 && -s "${packages[0]}" ]]
bash "${packages[0]}" --quiet --install-path="$build/install"
api=$(find "$build/install" -path '*/op_api/lib/libcust_opapi.so' -type f)
[[ -n "$api" && $(echo "$api" | wc -l) -eq 1 ]]
nm -D "$api" | grep -E 'aclnnRwkvReferenceWkv7(GetWorkspaceSize)?$'
printf 'WKV7_BUILD_ROOT=%s\nWKV7_INSTALL_ROOT=%s\n' "$build" "$build/install"
