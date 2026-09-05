# Image-derived length hints for ordinary B2 admission

Read-only exploratory review; no estimator or admission-policy change has been
made to the serving path. The estimator source/data are the user's separate
`/Users/lukaivanic/.codex/visualizations/2026/09/05/01a072ff-83ee-7cc2-90f4-7a6773e974c0/table_image_estimator/`.
Files inspected: `fast_image_cells.py`, `cell_oracle.py`, `image_cells.py`,
`fast_report.json`, `fast_rows.csv`, `image_cell_rows.csv`,
`cell_oracle_report.json`. Features use archived viewer images; production
input equivalence and server CPU cost remain unverified.

The two-feature image model (components + approximate filled-cell bands) has
page-grouped five-fold out-of-fold token predictions on 657 EOS-complete older
B1 outputs. Eight truncated labels are excluded from that regression, not
from any serving benchmark. Median error46.370 tokens, P90 error212.682,
Spearman0.94781. Cell-count-only privileged prediction median error105.848;
cells plus text length33.799. These are exploratory estimates after feature
development, not an untouched final validation set.

The fast file marks 361 cases below256 predicted tokens, with three actual
lengths over512 and none over1024. Two misses are nearly empty grids:
page_000506_table_7 (546 tokens, prediction80.93) and table_9 (541,100.32).
Both have408 enclosed rooms and13 horizontal grid lines. A conservative
image-derived structure floor of `rooms + max(0, grid_rows-1)` where
`confident_grid` is true removes those two from the short group. The remaining
miss is page_000631_table_0 (580,213.80). The fast timing (2.45ms median,
6.72ms P99 on <=802816-pixel local assets) excludes grayscale conversion and
the richer grid pass; it must not be quoted as the combined serving cost.

## Join against actual ordinary B2 requests

Reference: `table_vision_3b222363_20260905/cache_reload/development/results.jsonl`,
the frozen random100, C2, 2.892613 tables/s, P95 1.943158s. Join by canonical
table ID, use saved OOF predictions with the structure floor above, and count
actual native `token_ids` in these newer responses (never re-encode text).
55/100 are predicted below256; none of those produced more than512 tokens.

Match `other_prefill_spans.other_request_id` to response IDs after removing
the engine UUID suffix after `:`. This is diagnostic bookkeeping only.

| Running request | Actual latency s | Actual / predicted tokens | During-decode interruptions | Short newcomers among them |
|---|---:|---:|---:|---:|
| page_000272_table_box_id_1 | 4.325 | 3025 /1937 | 7 | 4 |
| page_000275_table_box_id_1 | 4.136 | 2852 /1355 | 10 | 6 |
| page_000287_table_box_id_8 | 2.542 | 1768 /1888 | 3 | 1 |
| page_000263_table_box_id_7 | 2.350 | 1532 /758 | 5 | 5 |
| page_001187_table_0 | 2.237 | 1553 /1425 | 4 | 4 |

For page000263 the short newcomers arrive during host spans starting at
0.209,0.921,1.416,1.835,2.116s of request age, lasting respectively
0.081,0.081,0.035,0.061,0.063s. Their actual outputs contain458,328,282,135,446
native tokens. These are host no-new-decode-submission spans, **not** exclusive
NPU stalls. No latency savings or counterfactual P95 are calculated by
subtracting them; changed admission changes all subsequent scheduling.

## Implication

Ordinary B2 is a natural first test, independent of speculation. Prepare the
newcomer's image/features on the CPU worker, then decide whether to defer its
NPU prefill while the existing request keeps decoding. Both requests count
toward C2 throughout, including the newcomer waiting. A future policy needs
bounded deferral and fallback for underestimated lengths; a negative
`predicted_total - generated_so_far` is evidence that the estimate failed,
not evidence of imminent EOS. Protecting every very long request until it
finishes would waste throughput even when the2s target is already impossible.

First test an explicitly non-qualifying oracle in real C2 to price the scheduling
opportunity, or test an image-derived conservative short classification if
preferred. Neither simulation/subtraction nor this reviewed prediction table
may enter the qualifying runtime. Any deployable estimator must consume real
request pixels and freeze its coefficients/policy before validation gates.

For the actual target, C3 is attractive after the B2 control: current B4/C3
already achieves3.671 tables/s but P95 2.214s, leaving throughput headroom for
selective deferral. B2 already meets P95 but lacks throughput. Neither claim
proves that a deferral policy improves the tradeoff; that requires measurement.
