# Static-kernel A/B evidence

This directory preserves compact evidence for the 128-page result documented
in `../../STATIC_KERNEL_AB_128.md`.

- `matrix.log.gz` contains the cold static-off cache warmup and both measured
  runs.
- `static_off_warmup/` records the one-page cache population run.
- `static_off/` and `static_on/` preserve the matched run manifests, summaries,
  input manifests, and model manifests.
- `static_off_eval/` and `static_on_eval/` preserve the pinned evaluator config,
  prediction hashes, metrics, runtime, stage execution, and exit status.
- `comparison.json` is the compact derived comparison.

Inference source commit: `a2397dff411999610e1327416a1575979266c41b`.
Evaluator commit: `2b161d010d2e3aff77a0edef359ea3a6411d23cd`.
