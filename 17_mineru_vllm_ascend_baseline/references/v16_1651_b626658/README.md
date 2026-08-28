# OmniDocBench v1.6 full evidence

This directory preserves the compact evidence for the 1,651-page stock
vLLM-Ascend MinerU2.5-Pro result documented in
`../../VERIFIED_V1_6_1651_BASELINE.md`.

- `primary_*` records the full inference process and its output-writer failure.
- `recovery_*` records the exact 160-page suffix recovery from offset 1,491.
- `combined_*` proves the complete accuracy-only prediction set.
- `evaluation_*` records evaluation preparation and exit status.
- `evaluator_*` preserves the pinned evaluator configuration, metrics, runtime,
  and stage execution.
- `prediction_manifest.json` records all 1,651 prediction hashes. The generated
  prediction bodies remain in the ignored run directory on the 910B container.

Primary inference source commit: `d7033177d0632aea2f658c7767b03f5fac1256d3`.
Recovery source commit: `d9b83b882cbeeea7f1693ee4f35eaabe442a9be2`.
Empty-prediction scoring fix: `b626658`.
Evaluator commit: `2b161d010d2e3aff77a0edef359ea3a6411d23cd`.
