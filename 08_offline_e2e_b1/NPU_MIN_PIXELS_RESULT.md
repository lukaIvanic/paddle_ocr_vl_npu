# 910B recognition `min_pixels` comparison

Observed on 2026-07-16 from source commit `e5af678`, using physical NPU 2 as
logical `npu:0`. Both settings ran the same five OmniDocBench pages through
real layout inference and the run-scoped compiled B=4 continuous decoder. Each
setting was repeated twice with the same code and TorchAir cache.

The control retained the installed PaddleOCR-VL 1.6 configuration:

```text
min_pixels = 112896
max_pixels = 1003520
patch_size = 14
merge_size = 2
```

The experiment passed `--preprocessor-min-pixels 56448`. This halves the area
floor, not each spatial dimension. Since the resize factor is
`patch_size * merge_size = 28`, the nominal minimum projected image-token count
falls from 144 to 72. `max_pixels` remains `1003520`.

## Verification

The override is applied to a copied runtime configuration rather than editing
the model directory. Every result records the model default, requested
override, effective minimum and maximum, patch/merge sizes, resize factor, and
nominal minimum image tokens under `configuration.preprocessor`.

The runtime evidence agrees with that configuration:

| Measurement across 160 recognized crops | Default | Half-area | Change |
|---|---:|---:|---:|
| Projected image tokens | 30,184 | 21,183 | -29.82% |
| Minimum projected tokens | 144 | 72 | -50.00% |
| Median projected tokens | 161 | 89 | -44.72% |
| Prompt tokens, including image placeholders | 32,264 | 23,263 | -27.90% |
| KV-prefix bytes copied into decode slots | 594,690,048 | 428,783,616 | -27.90% |
| Crops whose projected shape changed | 0 | 126 of 160 | 78.75% |

The unchanged 34 crops were already above the old minimum or constrained by
the maximum. Both detected table crops were in this group, so this sample says
nothing about table-accuracy sensitivity to a lower floor.

## Performance

| Setting | Run | Run wall | E2E output tok/s | Prefill wall sum | CPU crop preprocess | Vision device | Text device | Decode wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Default | 1 | 25.3866 s | 283.653 | 14.1358 s | 1.8720 s | 6.9093 s | 6.9669 s | 5.9352 s |
| Default | 2 | 23.0970 s | 311.771 | 12.8984 s | 1.6631 s | 6.2957 s | 6.3934 s | 5.8994 s |
| Half-area | 1 | 23.0894 s | 311.485 | 12.6253 s | 1.2714 s | 6.2160 s | 6.1699 s | 5.9902 s |
| Half-area | 2 | 23.8514 s | 301.533 | 13.1614 s | 1.3748 s | 6.4514 s | 6.4374 s | 5.9779 s |
| Default mean | | 24.2418 s | 297.712 | 13.5171 s | 1.7675 s | 6.6025 s | 6.6802 s | 5.9173 s |
| Half-area mean | | 23.4704 s | 306.509 | 12.8934 s | 1.3231 s | 6.3337 s | 6.3037 s | 5.9841 s |
| Mean change | | **-3.18%** | **+2.95%** | **-4.61%** | **-25.14%** | **-4.07%** | **-5.64%** | +1.13% |

The deterministic preprocessing and traffic reductions are decisive. The E2E
gain is not: individual wall-time ranges overlap, and the earlier historical
default run was 22.8657 seconds. The half-area setting therefore produced a
modest mean improvement in this paired sample, not a proven 3.18% speedup.

Decode is expected to remain flat. Its batch size, cache length, weights, and
compiled graph are identical. The half-area outputs contained 7,192 generated
tokens versus 7,201 for the default, which also makes raw E2E tok/s an imperfect
configuration-only comparison.

## Output fidelity and accuracy risk

Both default runs reproduced the historical default output exactly. Both
half-area runs also reproduced each other exactly. Comparing settings:

- 146 of 160 crops (91.25%) had identical text and complete token-ID sequences.
- 14 crops changed; all 160 retained the same stop reason.
- Mean normalized character similarity over all crops was 98.95%.
- Changes occurred in six formulas, four text blocks, two vision footnotes, one
  footer, and one paragraph title.

Ten changes were presentation-level differences such as line wrapping,
full-width versus ASCII parentheses, Roman `IV` versus `Ⅳ` as one Unicode
numeral, or LaTeX spacing and bracing. Of the four more substantive changes,
two removed or completed text that appears linguistically better, one inserted
a duplicated Chinese character and is a likely regression, and one matrix
formula was materially rewritten with mixed errors in both settings.

This is output fidelity, not a valid OmniDocBench accuracy score. Several
detected formula crops are subdivisions of larger ground-truth text blocks,
and Experiment 08 still lacks PaddleX's final merging and structured-output
postprocessing. The evidence rules out a broad collapse on these pages, but it
does not establish that quality is unchanged. Small formulas are the clearest
risk: 37 of 42 formula crops were resized and six changed output.

## System implications

- Only eager crop preprocessing and vision/text prefill receive smaller input.
  The compiled continuous-decode graph does not need recompilation.
- Position IDs, multimodal RoPE indices, prompt length, and the cache-length
  guard are recomputed from the effective resized grid, so the shorter prompts
  flow naturally through the existing path.
- Fewer prompt tokens reduce valid-prefix admission traffic by 27.90%, but each
  staged request still allocates a full fixed-length 2,048-token static cache.
  The override therefore does not reduce that cache allocation's HBM footprint.
- The ready reservoir is bounded by request count (`4B = 16`), not bytes, so a
  smaller prefix does not increase its capacity.
- Layout inference, boxes, page scheduling, and skipped-region policy are
  unchanged. Both settings produced 179 layout regions, 160 recognized crops,
  and 19 skipped regions in the same order.

The half-area floor is useful as a tunable performance/quality point, but these
results are not strong enough to make it the default. A broader, properly
postprocessed benchmark should decide that policy, with formula and small-text
results reported separately.

## Result artifacts

```text
tmp/08_offline_e2e_b1/five_pages_uniform/cross_page_b4_min_pixels_pair_default/run.json
tmp/08_offline_e2e_b1/five_pages_uniform/cross_page_b4_min_pixels_pair_default_repeat/run.json
tmp/08_offline_e2e_b1/five_pages_uniform/cross_page_b4_min_pixels_56448/run.json
tmp/08_offline_e2e_b1/five_pages_uniform/cross_page_b4_min_pixels_56448_repeat/run.json
```
