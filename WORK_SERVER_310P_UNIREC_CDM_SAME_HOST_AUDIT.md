# 310P UniRec canonical CDM re-score

This corrects the invalid 310P CDM score produced with ambient TeX Live
2022/dev and a dirty evaluator source file. It is the fastest decisive test for
the reported cross-chip Page-CDM gap.
It performs no inference, page matching, TEDS, compilation, or NPU work.

It runs CDM twice on the 310P host with one identical environment:

1. replay the completed 310P matched formulas;
2. replay the supplied 910B matched formulas.

The launcher automatically selects the previously installed frozen evaluator
runtime under `.runtime_cache/omnidocbench_eval/tools`. It rejects anything
other than TeX Live 2025/pdfTeX 1.40.28, ImageMagick 7.1.1-47, Ghostscript
9.55.0, or a TeX tree missing the official CJK/xcolor resources. It creates a clean
detached local clone of evaluator commit
`2b161d010d2e3aff77a0edef359ea3a6411d23cd` inside the audit root, so it does not
modify the existing dirty evaluator checkout.

It also fingerprints the clean evaluator source, Python packages,
architecture, TeX binaries and resources/fonts, ImageMagick configuration and
delegates, Ghostscript, linked libraries, locale, and CDM environment variables.

## Run

```bash
set -eo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main
test -z "$(git diff --name-only)"
test -z "$(git diff --cached --name-only)"

export RUN_ROOT="${RUN_ROOT:?set the completed full-1651 310P run root}"
export EVAL_PYTHON="${EVAL_PYTHON:?set the same evaluator Python used by that run}"
export EVALUATOR_ROOT="${EVALUATOR_ROOT:?set the same evaluator checkout used by that run}"
export CDM_WORKERS="${CDM_WORKERS:-64}"

bash 12_unirec_0_1b_inference/run_310p_unirec_cdm_same_host_audit.sh
```

The launcher prints `AUDIT_ROOT`, `RUN_LOG`, and `PID`. Send Luka the absolute
log path immediately and use `tail -f`. Expected CDM time is approximately
105 seconds per full pass at 64 workers on the 910B host; the 310P server time
depends only on its CPUs. There are two passes.

## Required report

Wait for `exit_code.txt`. Return:

1. the complete `UNIREC_CDM_SAME_HOST_AUDIT PASS` line;
2. `report.md`;
3. the `comparisons` and `scores` objects from `same_host_audit.json`;
4. clean evaluator commit and status from `candidate_runtime_fingerprint.json`;
5. platform machine, Python version, package versions, runtime tool versions,
   TeX-resource hashes, and `runtime_fingerprints.focused_comparison`;
6. absolute `AUDIT_ROOT` and `RUN_LOG`.

Do not rerun inference, modify the existing evaluator checkout, or install
packages. If the frozen runtime is unavailable, preserve the exact preflight
error and stop.
