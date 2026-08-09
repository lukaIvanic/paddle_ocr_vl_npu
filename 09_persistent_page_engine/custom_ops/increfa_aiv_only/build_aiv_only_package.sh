#!/usr/bin/env bash

# The CANN and ATB set_env scripts sourced by npu-setup can contain harmless
# nonzero intermediate commands. Source them before enabling errexit so Bash
# does not terminate this runner in the middle of npu-setup.
source npu-setup
set -euo pipefail

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
SOURCE_ROOT="${INCREFA_SOURCE_ROOT:-$PROJECT_ROOT/.runtime_cache/increfa_aiv_source}"
PATCH_PATH="$PROJECT_ROOT/09_persistent_page_engine/custom_ops/increfa_aiv_only/patches/0001-fp16-mha-flashdecode-aiv-only.patch"
ONE_CORE_PATCH_PATH="$PROJECT_ROOT/09_persistent_page_engine/custom_ops/increfa_aiv_only/patches/0002-diagnostic-one-aiv-launch.patch"
EXPECTED_SOURCE_COMMIT="afe72144f9f2ac8441929035795db88a111b30c5"
ENTRY_REL="attention/incre_flash_attention/op_kernel/incre_flash_attention_arch32.h"
EXPECTED_ENTRY_SHA256="20cb2397d84cf5d5386ebc09b6aa79eacfd3f32d956309c1b4e7f3e2690ef63b"
HOST_TILER_REL="attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp"
EXPECTED_HOST_TILER_SHA256="c84dbe37632ed428c080918ce4ca124d43640fbcfd05e69c212b9453f2b46a74"
TILING_KEY="11000000000100000"
AIV_LAUNCH_BLOCKS="${AIV_LAUNCH_BLOCKS:-auto}"
if [[ "$AIV_LAUNCH_BLOCKS" == "auto" ]]; then
    VENDOR_NAME="paddle_increfa_aiv_only"
elif [[ "$AIV_LAUNCH_BLOCKS" == "1" ]]; then
    VENDOR_NAME="paddle_increfa_aiv_only_1core"
else
    echo "ERROR: AIV_LAUNCH_BLOCKS must be auto or 1" >&2
    exit 2
fi
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$PROJECT_ROOT/.runtime_cache/increfa_aiv_only_builds/$RUN_ID"
PRESERVE_ROOT="$PROJECT_ROOT/.runtime_cache/increfa_build_preserved/$RUN_ID"
PYTHON_BIN="/usr/local/python3.12.13/bin/python3"
PYTHON_SITE="/usr/local/python3.12.13/lib/python3.12/site-packages"
ENTRY_PATH="$SOURCE_ROOT/$ENTRY_REL"
HOST_TILER_PATH="$SOURCE_ROOT/$HOST_TILER_REL"
PATCH_APPLIED=0
ONE_CORE_PATCH_APPLIED=0

restore_source() {
    if [[ "$ONE_CORE_PATCH_APPLIED" == "1" ]]; then
        git -C "$SOURCE_ROOT" apply -R "$ONE_CORE_PATCH_PATH"
        ONE_CORE_PATCH_APPLIED=0
    fi
    if [[ "$PATCH_APPLIED" == "1" ]]; then
        git -C "$SOURCE_ROOT" apply -R "$PATCH_PATH"
        PATCH_APPLIED=0
    fi
}
trap restore_source EXIT

if [[ ! -d "$SOURCE_ROOT/.git" ]]; then
    echo "ERROR: recovered IncreFA source is missing at $SOURCE_ROOT" >&2
    exit 2
fi
if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$EXPECTED_SOURCE_COMMIT" ]]; then
    echo "ERROR: recovered source commit drifted; inspect before updating the patch" >&2
    exit 2
fi
if [[ -n "$(git -C "$SOURCE_ROOT" status --short --untracked-files=no)" ]]; then
    echo "ERROR: recovered source has tracked modifications; refusing to overlap them" >&2
    git -C "$SOURCE_ROOT" status --short --untracked-files=no >&2
    exit 2
fi
if [[ "$(sha256sum "$ENTRY_PATH" | awk '{print $1}')" != "$EXPECTED_ENTRY_SHA256" ]]; then
    echo "ERROR: pristine entry-file hash differs; inspect before patching" >&2
    exit 2
fi
if [[ "$(sha256sum "$HOST_TILER_PATH" | awk '{print $1}')" != "$EXPECTED_HOST_TILER_SHA256" ]]; then
    echo "ERROR: pristine host-tiler hash differs; inspect before patching" >&2
    exit 2
fi

git -C "$SOURCE_ROOT" apply --check "$PATCH_PATH"
mkdir -p "$RUN_ROOT" "$PRESERVE_ROOT"
for build_name in build build_out; do
    if [[ -e "$SOURCE_ROOT/$build_name" ]]; then
        mv "$SOURCE_ROOT/$build_name" "$PRESERVE_ROOT/$build_name"
    fi
done

git -C "$SOURCE_ROOT" apply "$PATCH_PATH"
PATCH_APPLIED=1
if [[ "$AIV_LAUNCH_BLOCKS" == "1" ]]; then
    git -C "$SOURCE_ROOT" apply --check "$ONE_CORE_PATCH_PATH"
    git -C "$SOURCE_ROOT" apply "$ONE_CORE_PATCH_PATH"
    ONE_CORE_PATCH_APPLIED=1
fi

cd "$SOURCE_ROOT"
PYTHONPATH="$PYTHON_SITE" \
PATH="$PROJECT_ROOT/.runtime_cache/increfa_bin:$PATH" \
bash build.sh \
    --pkg \
    --soc=ascend910b \
    --vendor_name="$VENDOR_NAME" \
    --ops=incre_flash_attention \
    --ops-compile-options "--tiling_key=$TILING_KEY" \
    -j8 -O3 2>&1 | tee "$RUN_ROOT/build.log"

KERNEL_JSON="$SOURCE_ROOT/build/binary/ascend910b/bin/incre_flash_attention/IncreFlashAttention_7b761bdde53e2d667f3cdc458400fc8e.json"
PACKAGE_PATH="$SOURCE_ROOT/build_out/cann-ops-transformer-${VENDOR_NAME}_linux-aarch64.run"
if [[ ! -s "$KERNEL_JSON" || ! -s "$PACKAGE_PATH" ]]; then
    echo "ERROR: expected fixed-key metadata or package was not produced" >&2
    exit 3
fi

cp -p "$KERNEL_JSON" "$RUN_ROOT/kernel_metadata.json"
cp -p "$PACKAGE_PATH" "$RUN_ROOT/"
sha256sum "$ENTRY_PATH" "$HOST_TILER_PATH" "$KERNEL_JSON" "$PACKAGE_PATH" | tee "$RUN_ROOT/sha256.txt"

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
summary = {
    "coreType": metadata.get("coreType"),
    "core_type": metadata.get("core_type"),
    "intercoreSync": metadata.get("intercoreSync"),
    "magic": metadata.get("magic"),
    "tilingKey": tiling_key,
    "kernelType": kernel.get("kernelType"),
    "crossCoreSync": kernel.get("crossCoreSync"),
    "taskRation": kernel.get("taskRation"),
}
print("AIV_ONLY_KERNEL_METADATA=" + json.dumps(summary, sort_keys=True))
if tiling_key != 11000000000100000:
    raise SystemExit("unexpected tiling key")
if metadata.get("coreType") != "VectorCore" or metadata.get("core_type") != "AIV":
    raise SystemExit("compiler metadata does not describe an AIV-only kernel")
if metadata.get("magic") != "RT_DEV_BINARY_MAGIC_ELF_AIVEC":
    raise SystemExit("kernel binary is not tagged as AIVEC")
if kernel.get("kernelType", "").startswith("MIX"):
    raise SystemExit("per-kernel metadata still describes a mixed-core kernel")
if metadata.get("intercoreSync") not in (0, None):
    raise SystemExit("AIV-only metadata unexpectedly enables inter-core-type synchronization")
if kernel.get("crossCoreSync") not in (0, None):
    raise SystemExit("AIV-only kernel unexpectedly enables cross-core-type synchronization")
if kernel.get("taskRation") not in (None, ""):
    raise SystemExit("AIV-only kernel unexpectedly has a mixed-core task ratio")
PY

restore_source
trap - EXIT
if [[ "$(sha256sum "$ENTRY_PATH" | awk '{print $1}')" != "$EXPECTED_ENTRY_SHA256" ]]; then
    echo "ERROR: source restoration hash check failed" >&2
    exit 4
fi
if [[ "$(sha256sum "$HOST_TILER_PATH" | awk '{print $1}')" != "$EXPECTED_HOST_TILER_SHA256" ]]; then
    echo "ERROR: host-tiler restoration hash check failed" >&2
    exit 4
fi

echo "AIV_ONLY_BUILD_ROOT=$RUN_ROOT"
echo "AIV_ONLY_PACKAGE=$RUN_ROOT/$(basename "$PACKAGE_PATH")"
echo "UPSTREAM_SOURCE_RESTORED=true"
