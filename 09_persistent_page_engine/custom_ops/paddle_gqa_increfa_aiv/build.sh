#!/usr/bin/env bash

# Build the independent Paddle GQA IncreFA AIV operator from pinned CANN source.
source npu-setup
set -euo pipefail

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
CUSTOM_ROOT="$PROJECT_ROOT/09_persistent_page_engine/custom_ops/paddle_gqa_increfa_aiv"
OVERLAY_ROOT="$CUSTOM_ROOT/source_overlay"
PATCH_PATHS=(
    "$CUSTOM_ROOT/patches/0001-gqa-aiv-core-control.patch"
    "$CUSTOM_ROOT/patches/0002-separate-tiling-data-registration.patch"
    "$CUSTOM_ROOT/patches/0003-separate-tiling-template-registration.patch"
    "$CUSTOM_ROOT/patches/0004-restore-composite-tiling-schema.patch"
    "$CUSTOM_ROOT/patches/0005-allvec-gqa-query-heads.patch"
    "$CUSTOM_ROOT/patches/0006-allvec-gqa-flashdecode.patch"
)
EXPERIMENT_VARIANT="${PADDLE_GQA_EXPERIMENT_VARIANT:-production}"
SOURCE_ROOT="${INCREFA_SOURCE_ROOT:-$PROJECT_ROOT/.runtime_cache/increfa_aiv_source}"
EXPECTED_SOURCE_COMMIT="afe72144f9f2ac8441929035795db88a111b30c5"
UPSTREAM_OP_REL="attention/incre_flash_attention"
CUSTOM_OP_SNAKE="paddle_gqa_incre_flash_attention_aiv"
CUSTOM_OP_GE="PaddleGqaIncreFlashAttentionAiv"
CUSTOM_OP_REL="attention/$CUSTOM_OP_SNAKE"
PREPARED_OP_REL="$CUSTOM_OP_REL"
OP_API_SYMBOL_PREFIX="PaddleGqaIncreFlashAttentionAiv"
FUSED_DECODE=false
ENTRY_REL="$UPSTREAM_OP_REL/op_kernel/incre_flash_attention_arch32.h"
EXPECTED_ENTRY_SHA256="20cb2397d84cf5d5386ebc09b6aa79eacfd3f32d956309c1b4e7f3e2690ef63b"
HOST_TILER_REL="$UPSTREAM_OP_REL/op_host/incre_flash_attention_tiling.cpp"
EXPECTED_HOST_TILER_SHA256="c84dbe37632ed428c080918ce4ca124d43640fbcfd05e69c212b9453f2b46a74"
GQA_AIV_TILING_KEYS="11000000000000000,11000000000100000"
VENDOR_NAME="paddle_gqa_increfa_aiv"
CACHE_NAMESPACE="paddle_gqa_increfa_aiv"
case "$EXPERIMENT_VARIANT" in
    production)
        ;;
    grouped_serial_control)
        PATCH_PATHS+=("$CUSTOM_ROOT/patches/0007-grouped-serial-control.patch")
        VENDOR_NAME="paddle_gqa_grouped_serial_increfa_aiv"
        CACHE_NAMESPACE="paddle_gqa_grouped_serial_increfa_aiv"
        ;;
    grouped_half_control)
        PATCH_PATHS+=(
            "$CUSTOM_ROOT/patches/0007-grouped-serial-control.patch"
            "$CUSTOM_ROOT/patches/0008-grouped-half-control.patch"
        )
        VENDOR_NAME="paddle_gqa_grouped_half_increfa_aiv"
        CACHE_NAMESPACE="paddle_gqa_grouped_half_increfa_aiv"
        ;;
    split_k32_control)
        PATCH_PATHS+=("$CUSTOM_ROOT/patches/0009-force-split-k32-control.patch")
        VENDOR_NAME="paddle_gqa_split_k32_increfa_aiv"
        CACHE_NAMESPACE="paddle_gqa_split_k32_increfa_aiv"
        ;;
    decode_split_k32_control)
        PATCH_PATHS+=("$CUSTOM_ROOT/patches/0009-force-split-k32-control.patch")
        VENDOR_NAME="paddle_decode_gqa_split_k32_increfa_aiv"
        CACHE_NAMESPACE="paddle_decode_gqa_split_k32_increfa_aiv"
        OVERLAY_ROOT="$CUSTOM_ROOT/source_overlay_decode"
        CUSTOM_OP_SNAKE="paddle_decode_gqa_incre_flash_attention_aiv"
        CUSTOM_OP_GE="PaddleDecodeGqaIncreFlashAttentionAiv"
        CUSTOM_OP_REL="attention/$CUSTOM_OP_SNAKE"
        PREPARED_OP_REL="attention/paddle_gqa_incre_flash_attention_aiv"
        OP_API_SYMBOL_PREFIX="PaddleDecodeGqaIncreFlashAttentionAiv"
        FUSED_DECODE=true
        ;;
    split_k32_pairwise_sync_control)
        PATCH_PATHS+=(
            "$CUSTOM_ROOT/patches/0009-force-split-k32-control.patch"
            "$CUSTOM_ROOT/patches/0011-split-k32-pairwise-sync-control.patch"
        )
        VENDOR_NAME="paddle_gqa_split_k32_pairwise_sync_increfa_aiv"
        CACHE_NAMESPACE="paddle_gqa_split_k32_pairwise_sync_increfa_aiv"
        ;;
    split_k32_two_way_reduce_control)
        PATCH_PATHS+=(
            "$CUSTOM_ROOT/patches/0009-force-split-k32-control.patch"
            "$CUSTOM_ROOT/patches/0012-split-k32-two-way-reduce-control.patch"
        )
        VENDOR_NAME="paddle_gqa_split_k32_two_way_reduce_increfa_aiv"
        CACHE_NAMESPACE="paddle_gqa_split_k32_two_way_reduce_increfa_aiv"
        ;;
    split_k32_local_partial_reduce_control)
        PATCH_PATHS+=(
            "$CUSTOM_ROOT/patches/0009-force-split-k32-control.patch"
            "$CUSTOM_ROOT/patches/0012-split-k32-two-way-reduce-control.patch"
            "$CUSTOM_ROOT/patches/0013-split-k32-local-partial-reduce-control.patch"
        )
        VENDOR_NAME="paddle_gqa_split_k32_local_partial_reduce_increfa_aiv"
        CACHE_NAMESPACE="paddle_gqa_split_k32_local_partial_reduce_increfa_aiv"
        ;;
    split_k48_control)
        PATCH_PATHS+=("$CUSTOM_ROOT/patches/0010-force-split-k48-control.patch")
        VENDOR_NAME="paddle_gqa_split_k48_increfa_aiv"
        CACHE_NAMESPACE="paddle_gqa_split_k48_increfa_aiv"
        ;;
    *)
        echo "ERROR: unsupported PADDLE_GQA_EXPERIMENT_VARIANT=$EXPERIMENT_VARIANT" >&2
        exit 2
        ;;
esac
INSTALLED_VENDOR_NAME="${VENDOR_NAME}_transformer"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$PROJECT_ROOT/.runtime_cache/$CACHE_NAMESPACE/builds/$RUN_ID"
PYTHON_BIN="/usr/local/python3.12.13/bin/python3"
PYTHON_SITE="/usr/local/python3.12.13/lib/python3.12/site-packages"

if [[ ! -d "$SOURCE_ROOT/.git" ]]; then
    echo "ERROR: pinned official ops-transformer source is missing at $SOURCE_ROOT" >&2
    exit 2
fi
if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$EXPECTED_SOURCE_COMMIT" ]]; then
    echo "ERROR: pinned official source commit drifted" >&2
    exit 2
fi
if [[ -n "$(git -C "$SOURCE_ROOT" status --short --untracked-files=no)" ]]; then
    echo "ERROR: pinned official source has tracked changes" >&2
    exit 2
fi
if [[ "$(sha256sum "$SOURCE_ROOT/$ENTRY_REL" | awk '{print $1}')" != "$EXPECTED_ENTRY_SHA256" ]]; then
    echo "ERROR: pinned kernel source hash drifted" >&2
    exit 2
fi
if [[ "$(sha256sum "$SOURCE_ROOT/$HOST_TILER_REL" | awk '{print $1}')" != "$EXPECTED_HOST_TILER_SHA256" ]]; then
    echo "ERROR: pinned host tiler hash drifted" >&2
    exit 2
fi

OVERLAY_SHA="$(
    { find "$OVERLAY_ROOT" -type f -print0; printf '%s\0' "${PATCH_PATHS[@]}"; } \
        | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'
)"
BUILD_SOURCE_PARENT="$PROJECT_ROOT/.runtime_cache/$CACHE_NAMESPACE/sources"
BUILD_SOURCE_ROOT="${PADDLE_GQA_BUILD_SOURCE_ROOT:-$BUILD_SOURCE_PARENT/source_${EXPECTED_SOURCE_COMMIT:0:16}}"
SOURCE_MANIFEST="$BUILD_SOURCE_ROOT/.paddle_gqa_increfa_aiv_source_manifest.sha256"
OVERLAY_MANIFEST="$BUILD_SOURCE_ROOT/.paddle_gqa_increfa_aiv_overlay_sha"
mkdir -p "$BUILD_SOURCE_PARENT" "$RUN_ROOT"

SOURCE_PREPARED=false
if [[ ! -e "$BUILD_SOURCE_ROOT/.git" ]]; then
    git -C "$SOURCE_ROOT" worktree add --detach "$BUILD_SOURCE_ROOT" "$EXPECTED_SOURCE_COMMIT"
    git -C "$BUILD_SOURCE_ROOT" mv "$UPSTREAM_OP_REL" "$PREPARED_OP_REL"
    OP_ROOT="$BUILD_SOURCE_ROOT/$PREPARED_OP_REL"
    mv "$OP_ROOT/op_host/incre_flash_attention_def.cpp" "$OP_ROOT/op_host/incre_flash_attention_def.cpp.upstream_disabled"
    mv "$OP_ROOT/op_host/incre_flash_attention_infershape.cpp" "$OP_ROOT/op_host/incre_flash_attention_infershape.cpp.upstream_disabled"
    mv "$OP_ROOT/op_host/incre_flash_attention_tiling_register.cpp" "$OP_ROOT/op_host/incre_flash_attention_tiling_register.cpp.upstream_disabled"
    mv "$OP_ROOT/op_host/fallback_incre_flash_attention.cpp" "$OP_ROOT/op_host/fallback_incre_flash_attention.cpp.upstream_disabled"
    mv "$OP_ROOT/op_host/op_api" "$OP_ROOT/op_host/op_api_upstream_disabled"
    mv "$OP_ROOT/op_kernel/incre_flash_attention.cpp" "$OP_ROOT/op_kernel/incre_flash_attention.cpp.upstream_disabled"
    mv "$OP_ROOT/op_kernel/incre_flash_attention_apt.cpp" "$OP_ROOT/op_kernel/incre_flash_attention_apt.cpp.upstream_disabled"
else
    if [[ ! -s "$SOURCE_MANIFEST" ]]; then
        echo "ERROR: cached separate-op source lacks its manifest" >&2
        exit 2
    fi
    sha256sum -c "$SOURCE_MANIFEST"
    if [[ ! -s "$OVERLAY_MANIFEST" ]] || [[ "$(<"$OVERLAY_MANIFEST")" != "$OVERLAY_SHA" ]]; then
        echo "ERROR: cached separate-op source was prepared from a different overlay; use a fresh PADDLE_GQA_BUILD_SOURCE_ROOT" >&2
        exit 2
    fi
    SOURCE_PREPARED=true
fi

if [[ "$SOURCE_PREPARED" == false ]]; then
    OP_ROOT="$BUILD_SOURCE_ROOT/$PREPARED_OP_REL"
    cp -p "$OVERLAY_ROOT/CMakeLists.txt" "$OP_ROOT/CMakeLists.txt"
    cp -p "$OVERLAY_ROOT/op_host/CMakeLists.txt" "$OP_ROOT/op_host/CMakeLists.txt"
    if [[ "$FUSED_DECODE" == false ]]; then
        mkdir -p "$OP_ROOT/op_host/op_api"
        cp -p "$OVERLAY_ROOT/op_host/op_api/"* "$OP_ROOT/op_host/op_api/"
    fi
    cp -p "$OVERLAY_ROOT/op_host/"*.cpp "$OP_ROOT/op_host/"
    cp -p "$OVERLAY_ROOT/op_kernel/"*.cpp "$OP_ROOT/op_kernel/"
    for patch_path in "${PATCH_PATHS[@]}"; do
        git -C "$BUILD_SOURCE_ROOT" apply --unidiff-zero --check "$patch_path"
        git -C "$BUILD_SOURCE_ROOT" apply --unidiff-zero "$patch_path"
    done
    if [[ "$FUSED_DECODE" == true ]]; then
        while IFS= read -r -d '' source_path; do
            sed -i \
                -e 's/PaddleGqaIncreFlashAttentionAiv/PaddleDecodeGqaIncreFlashAttentionAiv/g' \
                -e 's/paddle_gqa_incre_flash_attention_aiv/paddle_decode_gqa_incre_flash_attention_aiv/g' \
                "$source_path"
        done < <(find "$OP_ROOT" -type f \
            ! -path '*/op_api_upstream_disabled/*' ! -name '*.upstream_disabled' -print0)
        git -C "$BUILD_SOURCE_ROOT" mv "$PREPARED_OP_REL" "$CUSTOM_OP_REL"
        OP_ROOT="$BUILD_SOURCE_ROOT/$CUSTOM_OP_REL"
    fi
    find "$OP_ROOT/CMakeLists.txt" "$OP_ROOT/op_host" "$OP_ROOT/op_kernel" -type f \
        ! -path '*/op_api_upstream_disabled/*' ! -name '*.upstream_disabled' -print0 \
        | sort -z | xargs -0 sha256sum > "$SOURCE_MANIFEST"
    printf '%s\n' "$OVERLAY_SHA" > "$OVERLAY_MANIFEST"
else
    OP_ROOT="$BUILD_SOURCE_ROOT/$CUSTOM_OP_REL"
fi

cd "$BUILD_SOURCE_ROOT"
PYTHONPATH="$PYTHON_SITE" PATH="$PROJECT_ROOT/.runtime_cache/increfa_bin:$PATH" \
bash build.sh --pkg --soc=ascend910b --vendor_name="$VENDOR_NAME" \
    --ops="$CUSTOM_OP_SNAKE" \
    --ops-compile-options "--tiling_key=$GQA_AIV_TILING_KEYS" \
    -j8 -O3 2>&1 | tee "$RUN_ROOT/build.log"

KERNEL_DIR="$BUILD_SOURCE_ROOT/build/binary/ascend910b/bin/$CUSTOM_OP_SNAKE"
mapfile -t KERNEL_JSONS < <(find "$KERNEL_DIR" -maxdepth 1 -type f -name "${CUSTOM_OP_GE}_*.json" -print | sort)
if [[ "${#KERNEL_JSONS[@]}" != "1" ]]; then
    echo "ERROR: expected one two-key $CUSTOM_OP_GE kernel bundle, got ${#KERNEL_JSONS[@]}" >&2
    printf '%s\n' "${KERNEL_JSONS[@]}" >&2
    exit 3
fi
KERNEL_JSON="${KERNEL_JSONS[0]}"
KERNEL_OBJECT="${KERNEL_JSON%.json}.o"
PACKAGE_PATH="$BUILD_SOURCE_ROOT/build_out/cann-ops-transformer-${VENDOR_NAME}_linux-aarch64.run"
if [[ ! -s "$KERNEL_OBJECT" || ! -s "$PACKAGE_PATH" ]]; then
    echo "ERROR: two-key operator object or package is missing" >&2
    exit 3
fi

symbols="$(readelf -Ws "$KERNEL_OBJECT")"
if [[ "$(awk '$4 == "FUNC" && $8 ~ /_mix_aiv$/ { count += 1 } END { print count + 0 }' <<<"$symbols")" != "2" ]]; then
    echo "ERROR: $KERNEL_OBJECT does not contain both MIX_AIV functions" >&2
    exit 3
fi
if awk '$4 == "FUNC" && $8 ~ /_mix_aic$/ { found = 1 } END { exit !found }' <<<"$symbols"; then
    echo "ERROR: $KERNEL_OBJECT still contains a cube function" >&2
    exit 3
fi

"$PYTHON_BIN" - "$KERNEL_JSON" <<'PY'
import json
import sys

expected = {11000000000000000, 11000000000100000}
path = sys.argv[1]
metadata = json.load(open(path, encoding="utf-8"))
kernels = metadata.get("kernelList", [])
if len(kernels) != 2:
    raise SystemExit(f"{path}: expected two fixed-key kernels, got {len(kernels)}")
seen = set()
summaries = []
for kernel in kernels:
    key = kernel.get("tilingKey")
    if key is None:
        suffix = kernel.get("kernelName", "").rsplit("_", 1)[-1]
        key = int(suffix) if suffix.isdigit() else None
    task_ratio = kernel.get("taskRation", metadata.get("taskRation"))
    summaries.append({
        "coreType": metadata.get("coreType"),
        "intercoreSync": metadata.get("intercoreSync"),
        "tilingKey": key,
        "taskRation": task_ratio,
    })
    if metadata.get("coreType") not in ("MIX", "MIX_AIV") or task_ratio != "0:1":
        raise SystemExit(f"{path}: not a zero-cube MIX_AIV kernel")
    seen.add(key)
print("PADDLE_GQA_INCREFA_AIV_KERNEL_METADATA=" + json.dumps(summaries, sort_keys=True))
if metadata.get("magic") != "RT_DEV_BINARY_MAGIC_ELF":
    raise SystemExit(f"{path}: unexpected ELF metadata")
if metadata.get("intercoreSync") != 1:
    raise SystemExit(f"{path}: hard-sync runtime contract is missing")
if seen != expected:
    raise SystemExit(f"unexpected tiling keys: {seen}")
PY

cp -p "$KERNEL_JSON" "$KERNEL_OBJECT" "$PACKAGE_PATH" "$RUN_ROOT/"
sha256sum "$RUN_ROOT"/*.json "$RUN_ROOT"/*.o "$PACKAGE_PATH" | tee "$RUN_ROOT/sha256.txt"

INSTALL_ROOT="$RUN_ROOT/installed"
"$PACKAGE_PATH" --quiet --install-path="$INSTALL_ROOT"
SET_ENV_PATH="$INSTALL_ROOT/vendors/$INSTALLED_VENDOR_NAME/bin/set_env.bash"
OP_API_LIB="$INSTALL_ROOT/vendors/$INSTALLED_VENDOR_NAME/op_api/lib/libcust_opapi.so"
[[ -s "$SET_ENV_PATH" ]] || { echo "ERROR: installed runtime is incomplete" >&2; exit 4; }
if [[ "$FUSED_DECODE" == false ]]; then
    [[ -s "$OP_API_LIB" ]] || { echo "ERROR: installed op API is missing" >&2; exit 4; }
    OP_API_SYMBOLS="$(nm -D "$OP_API_LIB")"
    for symbol in "aclnn${OP_API_SYMBOL_PREFIX}GetWorkspaceSize" "aclnn${OP_API_SYMBOL_PREFIX}"; do
        grep -q " T $symbol$" <<<"$OP_API_SYMBOLS" || { echo "ERROR: missing $symbol" >&2; exit 4; }
    done
fi

echo "PADDLE_GQA_INCREFA_AIV_BUILD_ROOT=$RUN_ROOT"
echo "PADDLE_GQA_INCREFA_AIV_BUILD_SOURCE=$BUILD_SOURCE_ROOT"
echo "PADDLE_GQA_INCREFA_AIV_PACKAGE=$RUN_ROOT/$(basename "$PACKAGE_PATH")"
echo "PADDLE_GQA_INCREFA_AIV_SET_ENV=$SET_ENV_PATH"
echo "PADDLE_GQA_INCREFA_AIV_OP_API=$OP_API_LIB"
echo "PADDLE_GQA_INCREFA_AIV_EXPERIMENT_VARIANT=$EXPERIMENT_VARIANT"
echo "STOCK_INCRE_FLASH_ATTENTION_SOURCE_UNCHANGED=true"
