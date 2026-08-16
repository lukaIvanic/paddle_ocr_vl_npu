#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SCRIPT_DIR/csrc/layout_msda_host_infer.cpp"
: "${ASCEND_HOME_PATH:?source the target CANN environment first}"
: "${1:?usage: build_host_opp.sh OUTPUT_ROOT}"

OUTPUT_ROOT="$(realpath -m "$1")"
ARCH="$(uname -m)"
case "$ARCH" in
  aarch64|x86_64) ;;
  *) printf 'ERROR unsupported architecture=%s\n' "$ARCH" >&2; exit 2 ;;
esac

CANN_INCLUDE="$ASCEND_HOME_PATH/include"
CANN_LIB="$ASCEND_HOME_PATH/lib64"
test -f "$CANN_INCLUDE/register/op_impl_registry.h"
test -f "$CANN_LIB/libexe_graph.so"
test -f "$CANN_LIB/libregister.so"

VENDOR_ROOT="$OUTPUT_ROOT/vendors/unirec_layout_msda"
TILING_ROOT="$VENDOR_ROOT/op_impl/ai_core/tbe/op_tiling"
TILING_LIB_DIR="$TILING_ROOT/lib/linux/$ARCH"
TILING_TARGET="$TILING_LIB_DIR/libcust_opmaster_rt2.0.so"
PROTO_LIB_DIR="$VENDOR_ROOT/op_proto/lib/linux/$ARCH"
PROTO_TARGET="$PROTO_LIB_DIR/libcust_opsproto_rt2.0.so"
mkdir -p "$TILING_LIB_DIR" "$PROTO_LIB_DIR"

build_host_library() {
  local target="$1" role="$2" kind_define="$3"
  g++ -std=c++17 -O2 -fPIC -shared \
    "$SOURCE" \
    -I"$CANN_INCLUDE" \
    -D"UNIREC_MSDA_LIBRARY_ROLE=\"$role\"" \
    -D"$kind_define" \
    -L"$CANN_LIB" \
    -Wl,--no-as-needed \
    -lexe_graph -lregister -lgraph -lgraph_base -lplatform \
    -lascendalog -lerror_manager -lmmpa -lc_sec \
    -Wl,--as-needed \
    -o "$target"
}

build_host_library "$TILING_TARGET" tiling OP_TILING_LIB
build_host_library "$PROTO_TARGET" proto OP_PROTO_LIB

ln -sfn "lib/linux/$ARCH/$(basename "$TILING_TARGET")" \
  "$TILING_ROOT/liboptiling.so"

printf 'UNIREC_LAYOUT_MSDA_HOST_OPP_BUILT vendor_root=%s tiling=%s proto=%s\n' \
  "$VENDOR_ROOT" "$TILING_TARGET" "$PROTO_TARGET"
printf '%s\n' "$VENDOR_ROOT"
