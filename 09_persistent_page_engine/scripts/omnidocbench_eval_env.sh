#!/usr/bin/env bash

# Runtime paths for the native ARM64 OmniDocBench evaluator environment.
# This file is safe to source from an NPU shell: it does not modify CANN,
# Torch-NPU, or device-selection variables.

_omnidocbench_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_omnidocbench_repo="$(cd "${_omnidocbench_script_dir}/../.." && pwd)"
_omnidocbench_workspace="${OMNIDOCBENCH_WORKSPACE_ROOT:-/workspace}"
_omnidocbench_repo_tools="${_omnidocbench_repo}/.runtime_cache/omnidocbench_eval/tools"
_omnidocbench_workspace_tools="${_omnidocbench_workspace}/.tools/omnidocbench_eval"

if [[ -n "${OMNIDOCBENCH_EVAL_TOOLS_ROOT:-}" ]]; then
  _omnidocbench_tools="$OMNIDOCBENCH_EVAL_TOOLS_ROOT"
elif compgen -G "${_omnidocbench_repo_tools}/texlive/2025/bin/*/pdflatex" >/dev/null; then
  # The 310P setup installs the frozen evaluator runtime beside this repo.
  _omnidocbench_tools="$_omnidocbench_repo_tools"
else
  # The 910B setup keeps the same frozen runtime at workspace scope.
  _omnidocbench_tools="$_omnidocbench_workspace_tools"
fi

export OMNIDOCBENCH_EVAL_PYTHON="${OMNIDOCBENCH_EVAL_PYTHON:-${_omnidocbench_workspace}/venvs/omnidocbench_py310/bin/python}"
export OMNIDOCBENCH_EVALUATOR_ROOT="${OMNIDOCBENCH_EVALUATOR_ROOT:-${_omnidocbench_workspace}/repos/OmniDocBench_eval}"

export OMNIDOCBENCH_TEXLIVE_ROOT="${OMNIDOCBENCH_TEXLIVE_ROOT:-${_omnidocbench_tools}/texlive/2025}"
export OMNIDOCBENCH_TEXLIVE_BIN="${OMNIDOCBENCH_TEXLIVE_BIN:-${OMNIDOCBENCH_TEXLIVE_ROOT}/bin/aarch64-linux}"
export OMNIDOCBENCH_PDFLATEX="${OMNIDOCBENCH_PDFLATEX:-${OMNIDOCBENCH_TEXLIVE_BIN}/pdflatex}"
export OMNIDOCBENCH_KPSEWHICH="${OMNIDOCBENCH_KPSEWHICH:-${OMNIDOCBENCH_TEXLIVE_BIN}/kpsewhich}"

export CDM_TEXLIVE_ROOT="$OMNIDOCBENCH_TEXLIVE_ROOT"
export CDM_TEXLIVE_BIN="$OMNIDOCBENCH_TEXLIVE_BIN"
export CDM_PDFLATEX="$OMNIDOCBENCH_PDFLATEX"
export CDM_KPSEWHICH="$OMNIDOCBENCH_KPSEWHICH"
export CDM_CJK_FONT="${CDM_CJK_FONT:-gkai}"

export OMNIDOCBENCH_IMAGEMAGICK_ROOT="${OMNIDOCBENCH_IMAGEMAGICK_ROOT:-${_omnidocbench_tools}/imagemagick-7.1.1-47}"
export OMNIDOCBENCH_EVAL_TOOLS_ROOT="$_omnidocbench_tools"
# Compatibility alias used by the direct CDM runner before the 310P runtime
# was installed under the repository-local cache.
export OMNIDOCBENCH_TOOL_ROOT="$_omnidocbench_tools"
export PATH="${OMNIDOCBENCH_TEXLIVE_BIN}:${OMNIDOCBENCH_IMAGEMAGICK_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${OMNIDOCBENCH_IMAGEMAGICK_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export MAGICK_HOME="$OMNIDOCBENCH_IMAGEMAGICK_ROOT"

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-${_omnidocbench_tools}/cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${_omnidocbench_tools}/cache/xdg}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${_omnidocbench_tools}/cache/matplotlib}"

unset _omnidocbench_script_dir _omnidocbench_repo \
  _omnidocbench_workspace _omnidocbench_repo_tools \
  _omnidocbench_workspace_tools _omnidocbench_tools
