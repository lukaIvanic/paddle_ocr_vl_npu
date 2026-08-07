# Table row drafts and offline speculative-decoding results

## Result

Ruled-row drafts are the only useful starting point. They dominate the other
four split modes in stitched Page-TEDS, draft cost, and speculative speed. A
legal prefix-only matcher accepts 84.6% of target decode tokens with K16. The
supplied cost model projects a 1.67x total speedup over B1 full-table OCR. On the
slowest p75+ tables, it projects a 1.95x speedup.

The full-table target model remains authoritative. Row-stitch Page-TEDS measures
the draft quality only. It does not predict the accuracy of a correct
speculative target runtime.

## Exact corpora

The experiment covers all 665 OmniDocBench v1.6 table crops.

- B1 target: 1,557,204 real vision tokens and 268,080 output tokens.
- Ruled drafts: 3,808 rows, 2,379,928 real vision tokens, and 319,353 output
  tokens in the optimized cross-table run.
- Every target and row draft retains its exact token IDs.
- The reusable compressed corpus is
  `tmp/09_persistent_page_engine/table_spec_full_d1e6d00/exact_token_corpora.tar.gz`.
- Archive SHA-256:
  `7d6e700d5db77764ae32bc5797608fa4f842219d4a7bcd16b0ed2cd91b3acc65`.

The archive is 25 MB and remains on Blue Zone. It is not committed to Git.

## Split-strategy comparison

The projected draft cost uses 30,000 vision tok/s and 6,000 draft decode tok/s.

| strategy | rows | real vision tokens | output tokens | projected draft cost | stitched Page-TEDS | best legal block | projected total speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| ruled | 3,808 | 2,379,928 | 319,353 | 132.56 s | 0.83840 | K32 | 1.678x |
| hybrid | 6,623 | 2,708,792 | 319,065 | 143.47 s | 0.77203 | K16 | 1.534x |
| whitespace | 7,121 | 2,749,536 | 336,083 | 147.67 s | 0.70894 | K16 | 1.509x |
| selected | 7,464 | 2,838,048 | 363,623 | 155.21 s | 0.71963 | K16 | 1.457x |
| row-edge | 9,019 | 3,063,312 | 735,212 | 224.65 s | 0.55474 | K16 | 1.114x |

The same-run B1 target corpus scores 0.94858 Page-TEDS. All TEDS evaluations
completed without errors. Row-edge had three bounded TEDS timeouts because of
degenerate draft outputs.

## Ruled-row block sweep

The baseline cost is 409.35 seconds:

- target vision: 51.91 seconds at 30,000 tok/s;
- target decode: 357.44 seconds at 750 tok/s.

The speculative cost includes ruled draft vision, ruled draft decode, target
vision, and weighted target verification. It excludes CPU, HTTP, text prefill,
and runtime implementation overhead.

| block | accepted / proposed | target decode coverage | accepted / speculative call | target decode speedup | projected total | total speedup | tables faster |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.879 | 0.457 | 0.88 | 1.84x | 378.96 s | 1.080x | 383 / 665 |
| 8 | 0.590 | 0.803 | 4.62 | 4.87x | 258.57 s | 1.583x | 555 / 665 |
| 16 | 0.422 | 0.846 | 6.46 | 5.99x | 244.92 s | 1.671x | 566 / 665 |
| 32 | 0.272 | 0.868 | 7.98 | 6.08x | 243.99 s | 1.678x | 556 / 665 |

K16 is the recommended first runtime. K32 improves the whole-corpus projection
by only 0.4%. K16 helps ten more tables and is marginally better on the p75+
subset.

## Slow-table subset

The saved B1 worker-latency p75 is 0.885 seconds. This gives 167 p75+ tables.

| block | target decode coverage | accepted / speculative call | target decode speedup | projected total speedup | tables faster |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.795 | 4.31 | 4.66x | 1.830x | 157 / 167 |
| 16 | 0.836 | 5.86 | 5.61x | 1.950x | 156 / 167 |
| 32 | 0.858 | 7.08 | 5.60x | 1.949x | 154 / 167 |

The legal K16 matcher is slower than baseline on 99 of 665 tables. Forty-three
of those targets contain at most 64 tokens, so draft overhead cannot amortize.
Eight have a draft runaway of at least 4,000 tokens. Only 11 of the 167 p75+
tables are slower. Among those 11, four have target coverage below 50%, two
have more than twice as many draft tokens as target tokens, and one is a draft
runaway.

The worst ruled K16 cases are:

| request | target tokens | draft tokens | target coverage | total speedup |
|---|---:|---:|---:|---:|
| `page_000918_table_6` | 55 | 4,055 | 0.907 | 0.131x |
| `page_000921_table_8` | 64 | 4,058 | 0.937 | 0.167x |
| `page_000911_table_box_id_5` | 160 | 8,174 | 0.742 | 0.200x |
| `page_000697_table_9` | 71 | 4,101 | 0.871 | 0.200x |
| `page_000861_table_5` | 288 | 4,275 | 0.359 | 0.250x |

The first low-risk policy improvement is to bypass row drafting for obviously
small tables and reject detected draft runaways. Phase 4 can develop a better
gate after the Phase 3 target runtime exists.

## Oracle ceiling

An oracle matcher that can inspect future target tokens is an upper bound, not
an implementable policy. Ruled K32 reaches 92.75% accepted target coverage,
10.97x target-decode speedup, and 1.879x projected total speedup. On p75+
tables, its projected total speedup is 2.309x. This bounds the remaining value
of matcher improvements above the legal K16/K32 result.

## Optimized row-draft execution

The first all-table runs opened and drained one recognizer schedule per table.
Ruled decode slot use was only 20.5%, and effective decode was 765 tok/s. The
sum of table recognition times was 702.11 seconds.

The cross-table B8 run submits all 3,808 ruled rows through one continuous
recognizer schedule:

- recognition wall: 189.88 seconds, 3.70x faster;
- CPU image loading, row analysis, and crop materialization: 207.43 seconds;
- stitching: 0.41 seconds;
- total after model setup: 400.04 seconds;
- model setup: 31.53 seconds;
- decode slot use: 98.94%;
- raw decode: 3,506 tok/s;
- effective decode: 3,369 tok/s;
- vision prefill: 55.16 device-seconds;
- text prefill: 19.03 device-seconds.

The measured 3,369 effective decode tok/s is below the supplied 6,000 tok/s
projection. The projection remains useful as the requested hardware-rate model.
The actual run also exposes a separate 207-second CPU preparation problem. That
CPU path is not part of the current speculative matcher analysis.

Cross-table scheduling preserves exact row token IDs for 661 of 665 tables.
The four differences are small batch-numeric changes. Re-simulating the
cross-table drafts changes aggregate acceptance by less than 0.01 percentage
point.

## Artifacts

- Full target and five-strategy draft root:
  `tmp/09_persistent_page_engine/table_spec_full_d1e6d00`
- Full five-strategy simulation:
  `tmp/09_persistent_page_engine/table_spec_full_d1e6d00/simulation_all`
- Optimized ruled run:
  `tmp/09_persistent_page_engine/table_row_cross_table_ruled_full_b5bb429`
- Optimized ruled simulation:
  `tmp/09_persistent_page_engine/table_row_cross_table_ruled_full_b5bb429/simulation`

The simulator is `scripts/table_speculative_simulator.py`. It uses only target
prefix tokens for the legal matcher. The `oracle_global` lane is clearly marked
as a future-token upper bound.
