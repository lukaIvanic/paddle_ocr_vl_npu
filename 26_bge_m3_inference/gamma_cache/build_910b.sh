#!/usr/bin/env bash
# Generate an isolated build tree from pinned source; never alter the baseline.
set -euo pipefail
UPSTREAM=${1:?pinned upstream checkout}
DEST=${2:?new experiment directory}
REPO=$(git rev-parse --show-toplevel)
PIN=ceb4536a2bd6fc99b85aec9d0fdcc0f470376292
test "$(git -C "$UPSTREAM" rev-parse HEAD)" = "$PIN"
git -C "$UPSTREAM" diff --exit-code
mkdir "$DEST"
mkdir "$DEST/source"
git -C "$UPSTREAM" archive "$PIN" | tar -x -C "$DEST/source"
# Preserve already downloaded, pinned build dependencies; no baseline build edits.
cp -a "$UPSTREAM/third_party/." "$DEST/source/third_party/"
cd "$DEST/source"
patch -p1 < "$REPO/26_bge_m3_inference/gamma_cache/gamma_beta_cache.patch"
sha256sum norm/add_layer_norm_quant/op_kernel/add_layer_norm_static_quant_normal_kernel.h \
  norm/add_layer_norm_quant_v2/op_kernel/add_layer_norm_quant_v2.cpp > "$DEST/source.sha256"
bash build.sh --pkg --soc=ascend910b --ops=add_layer_norm_quant_v2 \
  --no_force --vendor_name=bge_v2_gc -j8
mapfile -t packages < <(find build_out -maxdepth 1 -name 'cann-ops-nn-bge_v2_gc_linux-*.run')
test "${#packages[@]}" -eq 1
sha256sum "${packages[0]}" > "$DEST/package.sha256"
bash "${packages[0]}" --quiet --install-path="$DEST/install"
cd "$REPO"
python3 26_bge_m3_inference/prepare_v2_graph_package.py \
  --vendor "$DEST/install/vendors/bge_v2_gc_nn" \
  --output "$DEST/graph/vendors/bge_v2_gc_nn" --cann "$ASCEND_HOME_PATH"
