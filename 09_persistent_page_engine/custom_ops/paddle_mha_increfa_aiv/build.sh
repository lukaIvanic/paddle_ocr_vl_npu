#!/usr/bin/env bash

# npu-setup can run harmless nonzero commands while sourcing CANN and ATB.
source npu-setup
set -euo pipefail

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
CUSTOM_ROOT="$PROJECT_ROOT/09_persistent_page_engine/custom_ops/paddle_mha_increfa_aiv"
OVERLAY_ROOT="$CUSTOM_ROOT/source_overlay"
PATCH_PATHS=(
    "$CUSTOM_ROOT/patches/0001-mha-aiv-launch.patch"
    "$CUSTOM_ROOT/patches/0002-separate-tiling-data-registration.patch"
    "$CUSTOM_ROOT/patches/0003-separate-tiling-template-registration.patch"
    "$CUSTOM_ROOT/patches/0004-restore-composite-tiling-schema.patch"
)
SOURCE_ROOT="${INCREFA_SOURCE_ROOT:-$PROJECT_ROOT/.runtime_cache/increfa_aiv_source}"
EXPECTED_SOURCE_COMMIT="afe72144f9f2ac8441929035795db88a111b30c5"
UPSTREAM_OP_REL="attention/incre_flash_attention"
CUSTOM_OP_SNAKE="paddle_mha_incre_flash_attention_aiv"
CUSTOM_OP_GE="PaddleMhaIncreFlashAttentionAiv"
CUSTOM_OP_REL="attention/$CUSTOM_OP_SNAKE"
ENTRY_REL="$UPSTREAM_OP_REL/op_kernel/incre_flash_attention_arch32.h"
EXPECTED_ENTRY_SHA256="20cb2397d84cf5d5386ebc09b6aa79eacfd3f32d956309c1b4e7f3e2690ef63b"
HOST_TILER_REL="$UPSTREAM_OP_REL/op_host/incre_flash_attention_tiling.cpp"
EXPECTED_HOST_TILER_SHA256="c84dbe37632ed428c080918ce4ca124d43640fbcfd05e69c212b9453f2b46a74"
MHA_AIV_TILING_KEY="11000000000100000"
VENDOR_NAME="paddle_mha_increfa_aiv"
# ops-transformer appends this fixed suffix to the installed vendor directory.
INSTALLED_VENDOR_NAME="${VENDOR_NAME}_transformer"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$PROJECT_ROOT/.runtime_cache/paddle_mha_increfa_aiv/builds/$RUN_ID"
PYTHON_BIN="/usr/local/python3.12.13/bin/python3"
PYTHON_SITE="/usr/local/python3.12.13/lib/python3.12/site-packages"

if [[ ! -d "$SOURCE_ROOT/.git" ]]; then
    echo "ERROR: recovered official ops-transformer source is missing at $SOURCE_ROOT" >&2
    exit 2
fi
if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$EXPECTED_SOURCE_COMMIT" ]]; then
    echo "ERROR: recovered official source commit drifted" >&2
    exit 2
fi
if [[ -n "$(git -C "$SOURCE_ROOT" status --short --untracked-files=no)" ]]; then
    echo "ERROR: recovered official source has tracked changes" >&2
    git -C "$SOURCE_ROOT" status --short --untracked-files=no >&2
    exit 2
fi
if [[ "$(sha256sum "$SOURCE_ROOT/$ENTRY_REL" | awk '{print $1}')" != "$EXPECTED_ENTRY_SHA256" ]]; then
    echo "ERROR: recovered kernel source hash drifted" >&2
    exit 2
fi
if [[ "$(sha256sum "$SOURCE_ROOT/$HOST_TILER_REL" | awk '{print $1}')" != "$EXPECTED_HOST_TILER_SHA256" ]]; then
    echo "ERROR: recovered host tiler hash drifted" >&2
    exit 2
fi

OVERLAY_SHA="$(find "$OVERLAY_ROOT" "$CUSTOM_ROOT/patches" -type f -print0 \
    | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
BUILD_SOURCE_PARENT="$PROJECT_ROOT/.runtime_cache/paddle_mha_increfa_aiv/sources"
ACTIVE_SOURCE_FILE="$BUILD_SOURCE_PARENT/active_source_path"
if [[ -n "${PADDLE_MHA_BUILD_SOURCE_ROOT:-}" ]]; then
    BUILD_SOURCE_ROOT="$PADDLE_MHA_BUILD_SOURCE_ROOT"
elif [[ -s "$ACTIVE_SOURCE_FILE" ]]; then
    BUILD_SOURCE_ROOT="$(<"$ACTIVE_SOURCE_FILE")"
else
    BUILD_SOURCE_ROOT="$BUILD_SOURCE_PARENT/source_${EXPECTED_SOURCE_COMMIT:0:16}"
fi
SOURCE_MANIFEST="$BUILD_SOURCE_ROOT/.paddle_mha_increfa_aiv_source_manifest.sha256"
mkdir -p "$BUILD_SOURCE_PARENT" "$RUN_ROOT"
printf '%s\n' "$BUILD_SOURCE_ROOT" > "$ACTIVE_SOURCE_FILE"

if [[ ! -e "$BUILD_SOURCE_ROOT/.git" ]]; then
    git -C "$SOURCE_ROOT" worktree add --detach \
        "$BUILD_SOURCE_ROOT" "$EXPECTED_SOURCE_COMMIT"
    git -C "$BUILD_SOURCE_ROOT" mv "$UPSTREAM_OP_REL" "$CUSTOM_OP_REL"

    OP_ROOT="$BUILD_SOURCE_ROOT/$CUSTOM_OP_REL"
    mv "$OP_ROOT/op_host/incre_flash_attention_def.cpp" \
        "$OP_ROOT/op_host/incre_flash_attention_def.cpp.upstream_disabled"
    mv "$OP_ROOT/op_host/incre_flash_attention_infershape.cpp" \
        "$OP_ROOT/op_host/incre_flash_attention_infershape.cpp.upstream_disabled"
    mv "$OP_ROOT/op_host/incre_flash_attention_tiling_register.cpp" \
        "$OP_ROOT/op_host/incre_flash_attention_tiling_register.cpp.upstream_disabled"
    mv "$OP_ROOT/op_host/fallback_incre_flash_attention.cpp" \
        "$OP_ROOT/op_host/fallback_incre_flash_attention.cpp.upstream_disabled"
    mv "$OP_ROOT/op_host/op_api" "$OP_ROOT/op_host/op_api_upstream_disabled"
    mv "$OP_ROOT/op_kernel/incre_flash_attention.cpp" \
        "$OP_ROOT/op_kernel/incre_flash_attention.cpp.upstream_disabled"
    mv "$OP_ROOT/op_kernel/incre_flash_attention_apt.cpp" \
        "$OP_ROOT/op_kernel/incre_flash_attention_apt.cpp.upstream_disabled"

else
    if [[ ! -s "$SOURCE_MANIFEST" ]]; then
        echo "ERROR: cached separate-op source lacks its source manifest" >&2
        exit 2
    fi
    sha256sum -c "$SOURCE_MANIFEST"
fi

OP_ROOT="$BUILD_SOURCE_ROOT/$CUSTOM_OP_REL"
cp -p "$OVERLAY_ROOT/CMakeLists.txt" "$OP_ROOT/CMakeLists.txt"
cp -p "$OVERLAY_ROOT/op_host/CMakeLists.txt" "$OP_ROOT/op_host/CMakeLists.txt"
rm -f "$OP_ROOT/op_host/aclnn_paddle_mha_incre_flash_attention_aiv.cpp"
mkdir -p "$OP_ROOT/op_host/op_api"
cp -p "$OVERLAY_ROOT/op_host/op_api/"* "$OP_ROOT/op_host/op_api/"
cp -p "$OVERLAY_ROOT/op_host/"*.cpp "$OP_ROOT/op_host/"
cp -p "$OVERLAY_ROOT/op_kernel/"*.cpp "$OP_ROOT/op_kernel/"
for patch_path in "${PATCH_PATHS[@]}"; do
    if git -C "$BUILD_SOURCE_ROOT" apply --unidiff-zero --reverse --check "$patch_path" 2>/dev/null; then
        continue
    fi
    git -C "$BUILD_SOURCE_ROOT" apply --unidiff-zero --check "$patch_path"
    git -C "$BUILD_SOURCE_ROOT" apply --unidiff-zero "$patch_path"
done
{
    find \
        "$OP_ROOT/CMakeLists.txt" \
        "$OP_ROOT/op_host" \
        "$OP_ROOT/op_kernel" \
        -type f \
        ! -path '*/op_api_upstream_disabled/*' \
        ! -name '*.upstream_disabled' \
        -print0
} | sort -z | xargs -0 sha256sum > "$SOURCE_MANIFEST"
printf '%s\n' "$OVERLAY_SHA" > "$BUILD_SOURCE_ROOT/.paddle_mha_increfa_aiv_overlay_sha"

cd "$BUILD_SOURCE_ROOT"
PYTHONPATH="$PYTHON_SITE" \
PATH="$PROJECT_ROOT/.runtime_cache/increfa_bin:$PATH" \
bash build.sh \
    --pkg \
    --soc=ascend910b \
    --vendor_name="$VENDOR_NAME" \
    --ops="$CUSTOM_OP_SNAKE" \
    --ops-compile-options "--tiling_key=$MHA_AIV_TILING_KEY" \
    -j8 -O3 2>&1 | tee "$RUN_ROOT/build.log"

KERNEL_DIR="$BUILD_SOURCE_ROOT/build/binary/ascend910b/bin/$CUSTOM_OP_SNAKE"
mapfile -t KERNEL_JSONS < <(find "$KERNEL_DIR" -maxdepth 1 -type f \
    -name "${CUSTOM_OP_GE}_*.json" -print | sort)
if [[ "${#KERNEL_JSONS[@]}" != "1" ]]; then
    echo "ERROR: expected one $CUSTOM_OP_GE kernel JSON, got ${#KERNEL_JSONS[@]}" >&2
    printf '%s\n' "${KERNEL_JSONS[@]}" >&2
    exit 3
fi
KERNEL_JSON="${KERNEL_JSONS[0]}"
KERNEL_OBJECT="${KERNEL_JSON%.json}.o"
PACKAGE_PATH="$BUILD_SOURCE_ROOT/build_out/cann-ops-transformer-${VENDOR_NAME}_linux-aarch64.run"
if [[ ! -s "$KERNEL_OBJECT" || ! -s "$PACKAGE_PATH" ]]; then
    echo "ERROR: separate operator object or package is missing" >&2
    exit 3
fi

KERNEL_SYMBOLS="$(readelf -Ws "$KERNEL_OBJECT")"
if ! grep -q '_mix_aiv$' <<<"$KERNEL_SYMBOLS"; then
    echo "ERROR: separate operator ELF lacks the MIX_AIV vector function" >&2
    exit 3
fi
if grep -q '_mix_aic$' <<<"$KERNEL_SYMBOLS"; then
    echo "ERROR: separate operator ELF still contains a cube function" >&2
    exit 3
fi

"$PYTHON_BIN" - "$KERNEL_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
metadata = json.load(open(path, encoding="utf-8"))
kernels = metadata.get("kernelList", [])
if len(kernels) != 1:
    raise SystemExit(f"expected one fixed-key kernel, got {len(kernels)}")
kernel = kernels[0]
kernel_name = kernel.get("kernelName", "")
tiling_key = kernel.get("tilingKey")
if tiling_key is None and kernel_name.rsplit("_", 1)[-1].isdigit():
    tiling_key = int(kernel_name.rsplit("_", 1)[-1])
task_ratio = kernel.get("taskRation", metadata.get("taskRation"))
summary = {
    "coreType": metadata.get("coreType"),
    "intercoreSync": metadata.get("intercoreSync"),
    "magic": metadata.get("magic"),
    "tilingKey": tiling_key,
    "kernelType": kernel.get("kernelType"),
    "crossCoreSync": kernel.get("crossCoreSync"),
    "taskRation": task_ratio,
}
print("PADDLE_MHA_INCREFA_AIV_KERNEL_METADATA=" + json.dumps(summary, sort_keys=True))
if tiling_key != 11000000000100000:
    raise SystemExit("unexpected tiling key")
if metadata.get("coreType") not in ("MIX", "MIX_AIV"):
    raise SystemExit("kernel metadata does not describe a MIX_AIV launch")
if metadata.get("magic") != "RT_DEV_BINARY_MAGIC_ELF":
    raise SystemExit("kernel binary has unexpected ELF magic")
if metadata.get("intercoreSync") != 1:
    raise SystemExit("hard-sync runtime contract is missing")
if kernel.get("crossCoreSync") not in (None, 1):
    raise SystemExit("kernel has an unexpected cross-core sync value")
if task_ratio != "0:1":
    raise SystemExit("separate MHA AIV operator does not have a 0:1 task ratio")
PY

cp -p "$KERNEL_JSON" "$KERNEL_OBJECT" "$PACKAGE_PATH" "$RUN_ROOT/"
sha256sum "$KERNEL_JSON" "$KERNEL_OBJECT" "$PACKAGE_PATH" \
    | tee "$RUN_ROOT/sha256.txt"

INSTALL_ROOT="$RUN_ROOT/installed"
"$PACKAGE_PATH" --quiet --install-path="$INSTALL_ROOT"
SET_ENV_PATH="$INSTALL_ROOT/vendors/$INSTALLED_VENDOR_NAME/bin/set_env.bash"
OP_API_LIB="$INSTALL_ROOT/vendors/$INSTALLED_VENDOR_NAME/op_api/lib/libcust_opapi.so"
if [[ ! -s "$SET_ENV_PATH" || ! -s "$OP_API_LIB" ]]; then
    echo "ERROR: installed package is missing its environment or op-api library" >&2
    exit 4
fi
OP_API_SYMBOLS="$(nm -D "$OP_API_LIB")"
for symbol in \
    aclnnPaddleMhaIncreFlashAttentionAivGetWorkspaceSize \
    aclnnPaddleMhaIncreFlashAttentionAiv; do
    if ! grep -q " T $symbol$" <<<"$OP_API_SYMBOLS"; then
        echo "ERROR: installed op-api library does not export $symbol" >&2
        exit 4
    fi
done

mapfile -t TILING_LIBS < <(find \
    "$INSTALL_ROOT/vendors/$INSTALLED_VENDOR_NAME/op_impl/ai_core/tbe/op_tiling/lib" \
    -maxdepth 2 -type f -name 'libcust_opmaster_rt2.0.so' -print)
if [[ "${#TILING_LIBS[@]}" != "1" ]]; then
    echo "ERROR: expected one installed host tiling library" >&2
    exit 4
fi
TILING_SCHEMA_STRINGS="$(strings "${TILING_LIBS[0]}")"
for schema_name in \
    IncreFlashAttentionTilingDataOp \
    IncreFlashAttentionTilingDataPrefixOp \
    PaddleMhaIncreFlashAttentionAiv; do
    if ! grep -qx "$schema_name" <<<"$TILING_SCHEMA_STRINGS"; then
        echo "ERROR: installed host tiler lacks schema $schema_name" >&2
        exit 4
    fi
done

echo "PADDLE_MHA_INCREFA_AIV_BUILD_ROOT=$RUN_ROOT"
echo "PADDLE_MHA_INCREFA_AIV_BUILD_SOURCE=$BUILD_SOURCE_ROOT"
echo "PADDLE_MHA_INCREFA_AIV_PACKAGE=$RUN_ROOT/$(basename "$PACKAGE_PATH")"
echo "PADDLE_MHA_INCREFA_AIV_SET_ENV=$SET_ENV_PATH"
echo "PADDLE_MHA_INCREFA_AIV_OP_API=$OP_API_LIB"
echo "STOCK_INCRE_FLASH_ATTENTION_SOURCE_UNCHANGED=true"
