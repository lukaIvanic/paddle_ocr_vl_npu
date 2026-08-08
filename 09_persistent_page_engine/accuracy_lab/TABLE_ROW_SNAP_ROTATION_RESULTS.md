# Snapped and orientation-normalized table row drafts

## Scope

This experiment applies two image-only changes to the eight-band row-draft
strategy:

1. Rotate the 12 manually verified sideways table crops by 90 degrees clockwise.
   This affects row drafts only. The whole-table target remains unchanged.
2. Move each of the seven internal uniform-band cuts to a nearby horizontal rule
   or low-ink separator row.

The 910B run used commit `d367f64`, all 665 OmniDocBench v1.6 tables, one
cross-table B8 recognition schedule, and the same whole-table target tokens as
the earlier `uniform_8` experiment.

## CPU boundary validation

- Tables: 665
- Rotation ground truth applied: 12 of 12
- Upright tables: 653
- Boundaries identical to the reviewed visualization: 649 of 653
- Remaining differences: four one-pixel tie choices. Crop dimensions and the
  other 4,567 upright internal cuts agree.

The row-only fast path reduced proposal preparation from 203.3 seconds to 62.4
seconds. It skips the unrelated ruled, whitespace, edge, and hybrid detectors.

## OCR and stitched-table results

| Metric | Uniform 8 | Snapped + rotated |
|---|---:|---:|
| Row requests | 5,320 | 5,320 |
| Recognition wall | 256.6 s | 266.1 s |
| Real vision tokens | 2,731,772 | 2,753,456 |
| Row output tokens | 468,749 | 533,630 |
| Rows ending at KV capacity | 33 | 54 |
| Stitched Page-TEDS | 0.498909 | 0.569959 |
| Stitched structure Page-TEDS | 0.552366 | 0.611991 |

Rotation itself did not cause the extra long drafts. Across the 12 rotated
tables, row output fell from 18,637 to 15,645 tokens and no row reached KV
capacity in either run. The extra long drafts came from upright snapped crops:
33 to 54 KV-capacity rows.

## Practical draft acceptance

The practical matcher uses the generated target prefix only. These values use
the same matcher and cost model in both runs.

| Verification block | Accepted tokens/call, old | Accepted tokens/call, new | Target coverage, old | Target coverage, new | Ideal target-decode speedup, old | Ideal target-decode speedup, new |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 3.894 | 4.365 | 0.7804 | 0.7989 | 4.354x | 4.756x |
| 16 | 5.082 | 5.934 | 0.8195 | 0.8403 | 5.086x | 5.752x |
| 32 | 5.889 | 7.094 | 0.8384 | 0.8606 | 4.895x | 5.688x |

For the 16-token verifier:

| Cohort | Tables | Accepted/call, old | Accepted/call, new | Coverage, old | Coverage, new | Projected total speedup, old | Projected total speedup, new |
|---|---:|---:|---:|---:|---:|---:|---:|
| All | 665 | 5.082 | 5.934 | 0.8195 | 0.8403 | 1.401x | 1.385x |
| P75+ latency | 167 | 4.934 | 5.939 | 0.8201 | 0.8460 | 1.809x | 1.945x |
| P90+ latency | 67 | 4.567 | 5.652 | 0.8107 | 0.8411 | 1.961x | 2.147x |
| Original latency over 2 s | 48 | 4.419 | 5.409 | 0.8053 | 0.8341 | 1.991x | 2.184x |
| Rotated | 12 | 1.692 | 3.724 | 0.6155 | 0.7759 | 1.307x | 1.716x |
| Upright | 653 | 5.589 | 6.140 | 0.8321 | 0.8443 | 1.407x | 1.370x |

The all-table projected speedup falls slightly because the cost model charges
for every long row-draft token. The acceptance mechanism itself improves. The
slow-table cohorts, which are the latency target, improve in both acceptance and
projected total speed.

## Oracle upper bound at block 16

| Cohort | Accepted/call, old | Accepted/call, new | Coverage, old | Coverage, new |
|---|---:|---:|---:|---:|
| All | 9.674 | 10.210 | 0.8991 | 0.9042 |
| P75+ latency | 10.170 | 10.871 | 0.9057 | 0.9118 |
| P90+ latency | 10.118 | 10.968 | 0.9066 | 0.9136 |
| Rotated | 6.226 | 8.844 | 0.8570 | 0.8929 |

## Conclusion

Keep rotation normalization. It produces a large and clean gain on all 12
sideways tables.

Keep boundary snapping as the current draft-quality candidate. It improves
practical acceptance from 5.08 to 5.93 tokens per 16-token call and raises the
stitched Page-TEDS by 0.071. Before production use, bound or trim long row-draft
generations so that the 21 additional KV-capacity rows do not erase the
acceptance gain on the full corpus.
