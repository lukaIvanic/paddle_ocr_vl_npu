#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=omnidocbench_eval_env.sh
source "$SCRIPT_DIR/omnidocbench_eval_env.sh"

TOOLS_ROOT="$(dirname "$(dirname "$OMNIDOCBENCH_TEXLIVE_ROOT")")"
CACHE_ROOT="$TOOLS_ROOT/cache/downloads"
BUILD_ROOT="$TOOLS_ROOT/build"
BUILD_JOBS="${OMNIDOCBENCH_BUILD_JOBS:-16}"

TEXLIVE_INSTALLER_URL="https://ftp.tu-chemnitz.de/pub/tug/historic/systems/texlive/2025/tlnet-final/install-tl-unx.tar.gz"
TEXLIVE_REPOSITORY_URL="https://ftp.tu-chemnitz.de/pub/tug/historic/systems/texlive/2025/tlnet-final"
TEXLIVE_INSTALLER_SHA256="311df9f1477fd90c520159d1feddc2d6270f010d8349d1f6bdb9461a93b48a5c"
IMAGEMAGICK_TAG="7.1.1-47"
IMAGEMAGICK_COMMIT="82572afc879b439cbf8c9c6f3a9ac7626adf98fb"

mkdir -p "$CACHE_ROOT" "$BUILD_ROOT" "$HF_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

echo "[1/5] Installing Ubuntu build/runtime packages (no apt upgrade)"
apt-get update
apt-get install -y --no-install-recommends \
  build-essential ca-certificates fontconfig ghostscript git \
  libcairo2-dev libfontconfig1 libfontconfig-dev libfreetype6-dev \
  libjbig-dev libjpeg-dev liblzma-dev libpango1.0-dev libpng-dev \
  libtiff-dev libx11-dev perl pkg-config uuid-dev wget xz-utils zlib1g-dev

echo "[2/5] Installing ImageMagick ${IMAGEMAGICK_TAG} at ${OMNIDOCBENCH_IMAGEMAGICK_ROOT}"
if [[ ! -x "$OMNIDOCBENCH_IMAGEMAGICK_ROOT/bin/magick" ]]; then
  rm -rf "$BUILD_ROOT/ImageMagick"
  git clone --branch "$IMAGEMAGICK_TAG" --depth 1 \
    https://github.com/ImageMagick/ImageMagick.git "$BUILD_ROOT/ImageMagick"
  test "$(git -C "$BUILD_ROOT/ImageMagick" rev-parse HEAD)" = "$IMAGEMAGICK_COMMIT"
  (
    cd "$BUILD_ROOT/ImageMagick"
    ./configure --prefix="$OMNIDOCBENCH_IMAGEMAGICK_ROOT"
    make -j"$BUILD_JOBS"
    make install
  )
fi

echo "[3/5] Installing the frozen TeX Live 2025 CDM package set"
if [[ ! -x "$OMNIDOCBENCH_PDFLATEX" ]]; then
  INSTALLER="$CACHE_ROOT/install-tl-unx-2025.tar.gz"
  if [[ ! -f "$INSTALLER" ]]; then
    wget -O "$INSTALLER.part" "$TEXLIVE_INSTALLER_URL"
    mv "$INSTALLER.part" "$INSTALLER"
  fi
  echo "$TEXLIVE_INSTALLER_SHA256  $INSTALLER" | sha256sum -c -

  rm -rf "$BUILD_ROOT/install-tl"
  mkdir -p "$BUILD_ROOT/install-tl"
  tar -xzf "$INSTALLER" -C "$BUILD_ROOT/install-tl" --strip-components=1
  cat > "$BUILD_ROOT/texlive.profile" <<EOF
selected_scheme scheme-basic
TEXDIR $OMNIDOCBENCH_TEXLIVE_ROOT
TEXMFCONFIG $OMNIDOCBENCH_TEXLIVE_ROOT/texmf-config
TEXMFHOME $OMNIDOCBENCH_TEXLIVE_ROOT/texmf-home
TEXMFLOCAL $OMNIDOCBENCH_TEXLIVE_ROOT/texmf-local
TEXMFSYSCONFIG $OMNIDOCBENCH_TEXLIVE_ROOT/texmf-config
TEXMFSYSVAR $OMNIDOCBENCH_TEXLIVE_ROOT/texmf-var
option_doc 0
option_src 0
EOF
  "$BUILD_ROOT/install-tl/install-tl" \
    -profile "$BUILD_ROOT/texlive.profile" \
    -repository "$TEXLIVE_REPOSITORY_URL"
fi

"$OMNIDOCBENCH_TEXLIVE_BIN/tlmgr" option repository "$TEXLIVE_REPOSITORY_URL"
"$OMNIDOCBENCH_TEXLIVE_BIN/tlmgr" install \
  amsmath amsfonts arphic booktabs cjk cjkutils geometry multirow was xcolor

echo "[4/5] Preparing the isolated Python 3.10 evaluator environment"
if [[ ! -x "$OMNIDOCBENCH_EVAL_PYTHON" ]]; then
  /usr/bin/python3.10 -m venv "$(dirname "$(dirname "$OMNIDOCBENCH_EVAL_PYTHON")")"
  "$OMNIDOCBENCH_EVAL_PYTHON" -m pip install --upgrade \
    pip==26.0.1 setuptools==82.0.1 wheel==0.46.3
  "$OMNIDOCBENCH_EVAL_PYTHON" -m pip install \
    --no-build-isolation "$OMNIDOCBENCH_EVALUATOR_ROOT"
fi
"$OMNIDOCBENCH_EVAL_PYTHON" -c \
  'import apted, bs4, lxml, numpy, pandas, PIL, scipy, yaml; print("Python evaluator dependencies: PASS")'

echo "[5/5] Verifying exact critical versions and CDM resources"
"$OMNIDOCBENCH_EVAL_PYTHON" --version
"$OMNIDOCBENCH_EVAL_PYTHON" -m pip --version
pdflatex --version | head -n 2
kpsewhich --version | head -n 2
kpsewhich CJK.sty
kpsewhich c70gkai.fd
magick --version | head -n 2
gs --version
"$OMNIDOCBENCH_EVAL_PYTHON" \
  "$SCRIPT_DIR/verify_omnidocbench_eval_runtime.py" \
  --evaluator-root "$OMNIDOCBENCH_EVALUATOR_ROOT"

[[ "$(pdflatex --version | head -n 1)" == *"1.40.28 (TeX Live 2025)"* ]]
[[ "$(magick --version | head -n 1)" == *"ImageMagick 7.1.1-47"* ]]
[[ "$(gs --version)" == "9.55.0" ]]

echo "Runtime ready. Before evaluation, run:"
echo "  source $SCRIPT_DIR/omnidocbench_eval_env.sh"
