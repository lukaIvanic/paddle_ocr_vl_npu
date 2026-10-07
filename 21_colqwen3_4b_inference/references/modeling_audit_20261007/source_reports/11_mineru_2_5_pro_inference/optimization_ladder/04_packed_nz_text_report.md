# MinerU optimization rung 4: packed NZ text-prefill projections

## Result

Rejected and reverted. Compiled packed text prefill reused the decode-only
packed QKV and gate/up projections. These weights were already converted to
`FRACTAL_NZ`. The portable activation path remained unchanged.

| Metric | CPU-MRoPE baseline | Packed NZ text | Change |
|---|---:|---:|---:|
| Pipeline wall | 184.545 s | 187.566 s | +3.021 s (+1.6%) |
| Pages/s | 0.6936 | 0.6824 | -1.6% |
| Generation wall | 159.812 s | 161.629 s | +1.818 s |
| Prefill wall | 78.585 s | 80.723 s | +2.138 s |
| Decode wall | 40.010 s | 39.859 s | -0.151 s |
| Text transformer prefill | 22.338 s | 22.207 s | -0.131 s (-0.6%) |
| Physical text-prefill tok/s | 28,658 | 28,862 | +0.7% |

The 32-page gate appeared better: its text transformer was 5.0% faster and its
pipeline was 1.4% faster. The 128-page run showed that this did not scale. The
full text graph gained only 0.6%, which was smaller than runtime variance and
did not offset the higher surrounding prefill wall time.

## Accuracy

- 32-page gate: 32/32 Markdown files byte-identical.
- 128-page run: 126/128 Markdown files byte-identical.
- Two pages changed slightly because the larger packed matmuls changed numeric
  rounding. The path was rejected for performance before metric evaluation.

## Disposition

Commit `43725ec` implemented the experiment. Commit `52641ad` removed its full
feature surface. The accepted baseline remains CPU-side MRoPE at commit
`501cc02`, plus its documentation commits.

## Artifacts

- 32-page gate: `tmp/11_mineru_2_5_pro_inference/opt4_packed_nz_text_n32_43725ec/`
- 128-page run: `tmp/11_mineru_2_5_pro_inference/opt4_packed_nz_text_n128_43725ec/`
- Baseline: `tmp/11_mineru_2_5_pro_inference/opt3_cpu_mrope_n128_501cc02/`
