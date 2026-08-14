# 310P UniRec canonical CDM re-score

This corrects the invalid 310P CDM score produced with ambient TeX Live
2022/dev and a dirty evaluator source file. It is the fastest decisive test for
the reported cross-chip Page-CDM gap.
It performs no inference, page matching, TEDS, compilation, or NPU work.

It runs CDM twice on the 310P host with one identical environment:

1. replay the completed 310P matched formulas;
2. replay the supplied 910B matched formulas.

The launcher selects or repairs the frozen evaluator runtime under
`.runtime_cache/omnidocbench_eval/tools`. It rejects anything other than TeX
Live 2025/pdfTeX 1.40.28, ImageMagick 7.1.1-47, Ghostscript 9.55.0, or a TeX tree
missing the official CJK/xcolor resources. It creates a clean detached local clone of evaluator commit
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
export REPAIR_RUNTIME=1

bash 12_unirec_0_1b_inference/run_310p_unirec_cdm_same_host_audit.sh
```

The launcher prints `AUDIT_ROOT`, `RUN_LOG`, and `PID`. Send Luka the absolute
log path immediately and use `tail -f`. Expected CDM time is approximately
105 seconds per full pass at 64 workers on the 910B host; the 310P server time
depends only on its CPUs. There are two passes.

`REPAIR_RUNTIME=1` makes the background worker repair the missing runtime before
those passes. It first reuses any cached TeX installer and exact existing
ImageMagick/Ghostscript binaries. In the expected case, only the frozen TeX
Live 2025 package set is installed. It does not use an NPU.

Mirror speed is measured on the 310P host, not assumed from another machine.
The worker races complete installer downloads from the TU Chemnitz and Utah
frozen-2025 mirrors, validates each with the pinned SHA-256, accepts the first
valid completion, terminates the slower transfer, and uses the winner as the
repository for the larger TeX package download.

Watch these live lines in `RUN_LOG`:

- `CDM_RUNTIME_REPAIR_BEGIN`: states whether apt and ImageMagick compilation
  were skipped;
- `CDM_RUNTIME_MIRROR_PROGRESS` and `CDM_RUNTIME_MIRROR_SELECTED`: actual
  310P-side download bytes/rates and the selected repository;
- `CDM_RUNTIME_REPAIR_PROGRESS`: every 15 seconds, with stage, elapsed time,
  log growth rate, idle time, and the last installer line;
- `CDM_RUNTIME_REPAIR_PASS`: exact repair wall time;
- `[same-host] ... begin/done`: the two CDM scoring passes.

The repair has a 1,200-second hard timeout. Do not wait silently beyond it. If
`idle_s` exceeds 180, inspect `runtime_repair.log` immediately. Do not start a
second installer while the recorded PID is alive.

## Required report

Wait for `exit_code.txt`. Return:

1. the complete `UNIREC_CDM_SAME_HOST_AUDIT PASS` line;
2. `report.md`;
3. the `comparisons` and `scores` objects from `same_host_audit.json`;
4. clean evaluator commit and status from `candidate_runtime_fingerprint.json`;
5. platform machine, Python version, package versions, runtime tool versions,
   TeX-resource hashes, and `runtime_fingerprints.focused_comparison`;
6. absolute `AUDIT_ROOT` and `RUN_LOG`.
7. `CDM_RUNTIME_REPAIR_BEGIN`, every distinct progress stage, and
   `CDM_RUNTIME_REPAIR_PASS` with its wall time.

Do not rerun inference or modify the existing evaluator checkout. The committed
repair path is authorized to install only the missing frozen evaluator runtime.
Do not improvise another TeX version or manually patch evaluator source.
