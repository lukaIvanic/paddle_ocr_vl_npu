# Real table speculative-decoding runtime

## Result

The B1 D16 target runtime works across all 665 OmniDocBench v1.6 table crops.
It reduces 267,413 ordinary target decode steps to 41,896 target calls, or
6.383x fewer calls. The composed pipeline improves the slow-table tail, but it
does not yet improve total corpus time.

The representative run uses:

- the rotated and boundary-snapped eight-band row drafts;
- measured B8, KV768, packed-vision draft generation;
- the column-aware exact CPU matcher with reversible cursor and width patch;
- one compiled B1, D16, KV4096 PromptFA verifier;
- ordinary B1 IncreFA fallback when no draft candidate exists.

## Accuracy

| metric | ordinary B1 | speculative | delta |
|---|---:|---:|---:|
| Token-exact tables | 665 / 665 reference | 661 / 665 | -4 tables |
| Sample TEDS | 0.944385 | 0.944223 | -0.000162 |
| Page-TEDS | 0.948580 | 0.948345 | -0.000236 |
| Structure Page-TEDS | 0.973811 | 0.973466 | -0.000345 |

The four changed tables are deterministic. Their first token differences occur
at target positions 7, 114, 130, and 951. They are numerical differences
between multi-token PromptFA verification and ordinary one-token IncreFA. The
input contract and KV progression remain valid. The Page-TEDS change is 0.024
percentage point.

## Target runtime

| metric | result |
|---|---:|
| Generated target tokens | 267,413 |
| Target calls | 41,896 |
| Speculative calls | 36,668 |
| B1 fallback calls | 5,228 |
| Accepted draft tokens | 225,517 |
| Accepted tokens per speculative call | 6.150 |
| Accepted / proposed | 39.19% |
| Target-call reduction | 6.383x |
| Target prefill + decode sum | 239.07 s |

## Composed per-table latency

The composed latency is a sum of two measured stages. Draft generation comes
from the optimized per-table B8/KV768 run. Target prefill and D16 verification
come from the real runtime run. No rate model is used.

| percentile | ordinary B1 | composed speculative | change |
|---|---:|---:|---:|
| P50 | 0.463 s | 0.596 s | slower |
| P75 | 0.884 s | 0.944 s | slower |
| P90 | 1.642 s | 1.345 s | 18.1% faster |
| P95 | 2.449 s | 1.740 s | 29.0% faster |
| P99 | 4.847 s | 3.720 s | 23.3% faster |
| Maximum | 5.292 s | 6.260 s | slower |

Across the corpus, ordinary B1 totals 498.86 s. The composed speculative path
totals 522.14 s, or 0.955x baseline speed. It is faster on 246 of 665 tables.
The draft stage costs 283.07 s and the target stage costs 239.07 s.

One matcher failure is the new maximum:
`page_001595_table_box_id_1` needs 1,177 target calls and takes 6.260 s
composed, compared with 4.683 s for ordinary B1. A small-table bypass and a
confidence gate for weak anchors are the next practical controls. They must be
image- or prefix-derived; they must not use future target tokens.

## Validation history

- One-table compile smoke: 148 / 148 tokens exact.
- Sixteen-table live comparison: 16 / 16 exact, including rejection and
  fallback-heavy cases. Target model-device time improved 2.573x.
- Full ruled-draft diagnostic: 658 / 665 token exact.
- Full optimized snapped-draft run: 661 / 665 token exact and the accuracy and
  latency results above.

Artifacts on Blue Zone:

- Runtime output:
  `tmp/09_persistent_page_engine/table_spec_decode_full_snapped_kv768_1dfeb9d`
- Optimized draft output:
  `tmp/09_persistent_page_engine/table_row_per_table_kv768_pack2304_355db8b`
- Ordinary B1 target and score corpus:
  `tmp/09_persistent_page_engine/table_spec_full_d1e6d00/whole/whole`

