# Header-aware U1-U32 adaptive cut evaluation

Date: 2026-08-09  
Device: Ascend 910B2  
Code under test: `fd9926e` for the three-body-row selector and `fd8644f` for the denser two-body-row selector  
Cohort: all 143 tables whose saved whole-table B1 generation E2E is greater than 1 second.

## Result

The explicit U1-U32 cut-fit selector is a large improvement over the rejected
maximum-U policy. The denser header-aware version reaches 1.110x aggregate
speedup over B1, preserves 141/143 saved native B1 token sequences, and keeps
the stitched draft Page-TEDS slightly above fixed U8.

It is not a new default. Fixed U8 is still faster at mean, P50, P75, P90, P95,
and maximum latency. The dense adaptive selector is better only at P99: 4.757 s
versus 5.780 s. One remaining extreme table gives it a 6.747 s maximum, slightly
worse than U8's 6.483 s maximum.

## Selector implemented

Both selectors explicitly record a trial for every U from 1 through 32. A trial
above the image-derived context cap is rejected before cut-layout work. Each
feasible trial:

- starts from the normalized whole-table image;
- combines ruled-line, whitespace, row-edge, and hybrid row evidence;
- groups existing safe detector boundaries near a header-weighted target;
- uses snapped header-weighted uniform cuts only when the safe boundary pool
  cannot form the requested layout;
- rejects any layout whose retained cut crosses Otsu ink;
- rejects bands that are too short for the measured character height;
- keeps the first cut after the detected header plus 1.1 character heights;
- gives the first band at least 1.5 times the body-band target weight.

The two evaluated context budgets were:

| Strategy | Header context | Body context | U P50 | U P75 | U P90 | U P95 | U P99 | U max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `adaptive_cutfit_32_header_snapped` | about 2 visual rows | about 3 visual rows per band | 6 | 10 | 13 | 14 | 17 | 18 |
| `adaptive_cutfit_dense_32_header_snapped` | about 2 visual rows | about 2 visual rows per band | 9 | 14 | 19 | 20 | 26 | 26 |

The dense selector used safe detector-boundary grouping on 137/143 tables and
header-weighted snapped cuts on 6/143. It retained zero ink-crossing cuts on all
143 tables. The first-band/body-height ratio was 1.515 at P50 and 1.524 on
average. Per-row projected image tokens were P50 96, P90 182, P95 205, P99 300,
and maximum 986.

Selector and split CPU time is included in draft E2E. For the dense selector it
was 0.333 s at P50, 0.476 s at P95, and 0.630 s maximum.

## Draft quality

All TEDS values use the same 143 tables and 125 pages.

| Draft | Page-TEDS | Sample TEDS | Structure-only Page-TEDS |
|---|---:|---:|---:|
| U8 snapped | 0.791554 | 0.798882 | 0.845308 |
| U16 snapped | 0.696407 | 0.700067 | 0.750143 |
| Rejected maximum adaptive U | 0.658645 | 0.671857 | 0.706842 |
| Header-aware, 3 body rows | 0.820767 | 0.827714 | 0.868551 |
| Header-aware, 2 body rows | 0.798383 | 0.804537 | 0.850353 |
| Final dense adaptive-U + adaptive-K output | 0.941803 | 0.939364 | 0.967314 |

Dense adaptive drafts beat U8 on 72 tables, tie on 15, and lose on 56. The
mean per-table TEDS delta versus U8 is +0.00566. The less-dense selector has the
better draft: it beats U8 on 77 tables and has a +0.02883 mean delta.

## Draft latency

Seconds per table. Percentiles use nearest rank.

| Draft | Mean | P50 | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| U8 | 0.548 | 0.455 | 0.593 | 0.864 | 1.065 | 1.125 | 2.849 |
| U16 | 0.684 | 0.542 | 0.737 | 1.219 | 1.571 | 1.657 | 3.366 |
| Header-aware, 3 body rows | 1.070 | 0.967 | 1.203 | 1.403 | 1.792 | 2.963 | 3.343 |
| Header-aware, 2 body rows | 0.993 | 0.895 | 1.110 | 1.360 | 1.675 | 2.063 | 2.448 |

The dense run used KV768 for 141 split tables. The two safe U1 fallbacks,
`page_000283_table_box_id_1` and `page_000284_table_box_id_0`, used KV1536 after
an explicit warmup. Their native generated token IDs were merged back into the
143-table dataset order before verification.

## Composed speculative latency

Composed latency is measured row-draft generation plus live target prefill and
adaptive-K verification/decode. Setup and compilation are excluded. Adaptive K
starts at K16, doubles after a fully accepted call, halves after a rejected
call, and is bounded to K8/K16/K32/K64.

| Lane | Mean | P50 | P75 | P90 | P95 | P99 | Max | Aggregate speedup | Faster tables | Exact saved IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original whole-table B1 | 2.029 | 1.573 | 2.380 | 4.034 | 4.806 | 4.900 | 5.292 | 1.000x | - | - |
| U8, K16 | 1.459 | 1.059 | 1.670 | 2.512 | 3.733 | 5.780 | 6.483 | 1.391x | 129/143 | 141/143 |
| U16, K16 | 1.642 | 1.264 | 2.103 | 3.067 | 3.804 | 4.715 | 5.578 | 1.235x | 106/143 | 139/143 |
| Rejected maximum U, adaptive K | 2.316 | 1.976 | 2.754 | 3.873 | 4.798 | 6.070 | 6.291 | 0.876x | 46/143 | 141/143 |
| Header-aware 3-row, adaptive K | 1.840 | 1.480 | 2.113 | 3.015 | 3.916 | 6.460 | 7.020 | 1.102x | 82/143 | 140/143 |
| Header-aware 2-row, adaptive K | 1.827 | 1.453 | 2.059 | 3.347 | 3.926 | 4.757 | 6.747 | 1.110x | 82/143 | 141/143 |

The denser drafts reduced draft latency, but their lower consistency increased
verification work. Proposed-token acceptance fell from 51.57% to 48.50%, and
target calls increased from 21,216 to 23,272. Target-call reduction therefore
fell from 8.07x to 7.35x. This explains why dense improves P99 while slightly
worsening P90 and P95 relative to the three-body-row selector.

The dense exact-saved mismatches were `page_000263_table_box_id_7` and
`page_000273_table_box_id_1`. Exactness is against saved native B1 token IDs.
No generated text was decoded and re-encoded for drafting, stitching, or
verification.

## Dense tail examples

| Request | U | Draft s | Target path s | Composed s | B1 s | Speedup | Target calls | Accepted fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| page_000279_table_box_id_0 | 9 | 2.063 | 4.683 | 6.747 | 4.806 | 0.712x | 1,159 | 21.28% |
| page_000279_table_box-fy04hrwa | 6 | 1.818 | 2.939 | 4.757 | 3.449 | 0.725x | 708 | 25.97% |
| page_000271_table_box_id_1 | 13 | 1.574 | 3.023 | 4.597 | 4.847 | 1.054x | 717 | 32.42% |
| page_001595_table_box_id_1 | 12 | 1.360 | 3.229 | 4.589 | 4.683 | 1.020x | 723 | 28.78% |
| page_000283_table_box_id_1 | 1 | 1.686 | 2.649 | 4.335 | 4.878 | 1.125x | 670 | 44.84% |

## Decision

Keep fixed U8 as the production default.

Keep `adaptive_cutfit_dense_32_header_snapped` as the better adaptive research
lane. It validates the main design ideas: search U1-U32, preserve a larger
header band, use only safe snapped cuts, and avoid forced maximum U. It turns
the rejected 0.876x maximum-U policy into a 1.110x lane and fixes that policy's
draft-quality collapse.

The next useful optimization is not more visual rows. It is reducing the
selector's 0.33 s median CPU cost and learning when an extra safe band will
reduce draft time without lowering speculative acceptance. The remaining
worst case is verifier divergence, not crop geometry.
