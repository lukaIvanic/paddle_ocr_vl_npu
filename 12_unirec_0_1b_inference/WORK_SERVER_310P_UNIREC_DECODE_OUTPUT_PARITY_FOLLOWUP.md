# 310P UniRec decode-output parity follow-up

## Purpose

The completed first-128 diagnostic reported 957 crops, 10,263 decode iterations,
about 210 seconds of decode-graph time, about 6.23k raw token slots/s, and about
5k effective tokens/s.  The matched 910B2 reference used the same 957 crop IDs
but only 2,047 iterations and generated 40,917 tokens in total (42.76/crop).

The existing `PASS` means that the diagnostic runner completed.  It did not pass
`--reference-trace`, so it is not an output-parity or OCR-accuracy result.

This follow-up must answer two questions:

1. Did the completed 310P run generate roughly one thousand tokens per crop and
   hit the 2,047-token cap repeatedly?
2. Across 256 crops on the fixed B128 graph, including at least 128 real slot
   refills, where do tokens first differ from the existing 910B2 output for the
   same request IDs, and are the long outputs repetitive?

Do not regenerate prefill.  Do not delete or rebuild the decode cache.  Reuse
the completed persistent artifact and the already-passed B128 cache.

## Pull and inputs

Pull the commit named in Luka's message.  Work in the repository root.  Do not
edit tracked files, create a branch, commit, or push.

Reuse the same validated environment variables from the completed diagnostic:

```bash
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"

export PYTHON_BIN=/absolute/path/to/the/validated/venv/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export COMPILE_CACHE=/absolute/path/to/the/existing/production/cache/parent
export UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE=/absolute/path/reported/by/the/passed/cache/gate
export ASCEND_RT_VISIBLE_DEVICES=0  # one free physical 310P, 0-3 only
export CPUSET=0-63
export PRIOR_RUN_ROOT=/absolute/path/to/the/completed/310p_production_decode_diagnostic_run
```

`PYTHON_BIN` must be the real executable inside the validated venv.  Do not use
`readlink -f` on a venv symlink that escapes to `/usr/local`.  The committed
runner preserves the supplied executable basename for this reason.

## Run

```bash
bash 12_unirec_0_1b_inference/run_310p_decode_output_parity_followup_background.sh
```

The launcher prints `RUN_ROOT` and the absolute `RUN_LOG`.  Immediately give
Luka this command:

```bash
tail -f /absolute/RUN_LOG/from/the/launcher
```

The first `PRIOR_CLEAN_SUMMARY` line is CPU-only and should appear immediately.
It is already enough to confirm or reject runaway generation from the saved
length distribution.

The NPU part replays only the first 256 crop rows through the existing B128
graph.  This deliberately forces slot reuse; a 128-crop-only test could miss a
refill/reset defect.  It writes every generated token after the decode timing
window and compares an irreversible SHA-256 sequence digest with the matched
910B2 reference.  No 910B2 OCR tokens are stored in Git.  The matched subset
generated 8,625 tokens total, averaged 33.69 tokens/crop, and had exactly one
2,047-token cap.  At the observed 310P step
time, the bounded replay should need roughly 42-50 seconds of graph time even
if refilled rows run to the cap.  Cache load and model setup add some wall time,
but there must be no prefill run and no compile-cache rebuild.

## Hard stops

Stop and report the exact log if any of these occurs:

- the runner attempts prefill or layout work;
- the B128 cache inventory is not exactly one `compiled_module` and one OM;
- TorchDynamo reports recompilation;
- an OM is added or replaced;
- a physical device outside 0-3 is selected;
- the existing artifact cannot be resolved from `PRIOR_RUN_ROOT/clean.json`.

Do not repair by deleting caches.  Do not switch to B64 or change KV lengths.

## Report

Paste back, verbatim:

1. `PRIOR_CLEAN_SUMMARY`
2. `UNIREC_PRODUCTION_DECODE_REPLAY_END`
3. the complete `validation` object from `replay.json`
4. `UNIREC_DECODE_OUTPUT_PARITY: PASS`
5. `DECODE_OUTPUT_PARITY_REPORT`
6. `DECODE_CACHE_OM_INVENTORY_UNCHANGED`
7. `RUN_ROOT`, `RUN_LOG`, and `exit_code.txt`

Also state whether the cache inventory and OM hashes were unchanged before and
after the replay.  Interpret the result conservatively:

- many `length_cap` outputs confirm missing EOS/runaway generation;
- large repeated-token or repeated-4-gram counts confirm repetition;
- a digest mismatch with equal length means token content changed;
- a digest and length mismatch plus a candidate cap confirms runaway output;
- candidate repetition indicators distinguish repeated loops from merely long
  but varied hallucinated output.
