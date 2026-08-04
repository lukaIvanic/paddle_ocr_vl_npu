#!/usr/bin/env bash

# Runtime paths for the native ARM64 OmniDocBench evaluator environment.
# This file is safe to source from an NPU shell: it does not modify CANN,
# Torch-NPU, or device-selection variables.

_omnidocbench_workspace="${OMNIDOCBENCH_WORKSPACE_ROOT:-/workspace}"
_omnidocbench_tools="${OMNIDOCBENCH_EVAL_TOOLS_ROOT:-${_omnidocbench_workspace}/.tools/omnidocbench_eval}"

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
export PATH="${OMNIDOCBENCH_TEXLIVE_BIN}:${OMNIDOCBENCH_IMAGEMAGICK_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${OMNIDOCBENCH_IMAGEMAGICK_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export MAGICK_HOME="$OMNIDOCBENCH_IMAGEMAGICK_ROOT"

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-${_omnidocbench_tools}/cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${_omnidocbench_tools}/cache/xdg}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${_omnidocbench_tools}/cache/matplotlib}"

unset _omnidocbench_workspace _omnidocbench_tools
