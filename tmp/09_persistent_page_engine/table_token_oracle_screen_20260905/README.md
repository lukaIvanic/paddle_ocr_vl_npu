# Perfect output-length oracle: preliminary routing screen

CPU-only analysis of the same 100 distinct random tables, 2026-09-05.
No serving code was changed, no inference was run, and no generated text was
re-encoded. `analyze.py` counts saved ordinary-B1 native output IDs, including EOS.
The four inputs have identical selection-file SHA-256:
`1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85`.

## Rule

Keep the current height eligibility, then speculate only when the oracle's
ordinary-B1 output-token count is at least T. The decision does not use arrival
order, companion, latency, acceptance, or per-table exceptions. The oracle is
an experimental stand-in for an image-only estimator, not a production feature.

## Evidence of unnecessary speculation

Of the 61 currently speculative requests:

| Native B1 tokens | Tables | Spec faster in sequential comparison | Net sequential seconds saved by spec |
|---|---:|---:|---:|
| <256 | 19 | 1 | -4.621 |
| 256–511 | 17 | 6 | -1.074 |
| 512–767 | 12 | 11 | 2.367 |
| 768–1,023 | 5 | 4 | 1.026 |
| 1,024–1,535 | 4 | 3 | 0.527 |
| >=1,536 | 4 | 4 | 6.526 |

Two illustrative current C2 tails have identical B1/spec output IDs:

| Table | Tokens | Ordinary B1 | Spec C1 | Current spec C2 |
|---|---:|---:|---:|---:|
| page_000921_table_9 | 90 | 0.260 s | 1.195 s | 2.827 s |
| page_000217_table_box_id_0 | 138 | 0.347 s | 1.278 s | 2.511 s |

Their ordinary B1 values are NOT predictions of their routed C2 latency.

## Threshold screen

| Rule | Spec requests | Historical regular-B2 >2 s cases retained | Sequential request-time saving vs current routing |
|---|---:|---:|---:|
| Current height rule | 61 | 6/6 | — |
| Add >=256 tokens | 42 | 6/6 | 4.621 s |
| Add >=512 tokens | 25 | 6/6 | 5.695 s |
| Add >=768 tokens | 13 | 6/6 | 3.328 s |
| Add >=1,024 tokens | 8 | 6/6 | 2.302 s |

The projected sequential distribution is obtained by replacing newly filtered
requests' spec-C1 latency with their saved ordinary-B1 latency; the 39 existing
ordinary routes keep their contemporaneous C1 measurements. These are mixed-run
offline projections, not measured routed runs:

| Rule | Mean | P95 | P99 | Maximum |
|---|---:|---:|---:|---:|
| Current C1 measured | 0.584 s | 1.278 s | 1.489 s | 2.596 s |
| >=512 projected C1 | 0.527 s | 1.245 s | 1.489 s | 2.596 s |
| >=1,024 projected C1 | 0.561 s | 1.261 s | 1.489 s | 2.596 s |

No new C2 throughput/P95 is inferred by splicing latencies from different runs.
In C2, rerouting changes request completions, companions, shared graph calls,
prefill interference, and CPU overlap. Removing short-table drafts also adds
ordinary decode work and changes which kinds of work can batch. Consequently,
neither saved sequential seconds nor C1 percentile changes can be directly
translated to C2 improvement.

## Next empirical comparison

The evidence supports testing T=512 and T=1,024 against the current C2 baseline
(2.033 tables/s, P95 2.514 s) using the same sample/order, client cap 2 and all
existing optimized model settings. Admission decisions would consult only the
frozen oracle length; generation would still use the real image and model,
never saved output tokens. Warmup remains outside timing, all request latency
inside, no table exceptions or arrival-dependent routing.

The 512 threshold removes 36 drafts while retaining most sequential speculation
benefits. The 1,024 threshold asks the user's narrower question: reserve
speculation for the longest requests and let ordinary decoding batch the rest.

Do not treat either threshold as validated on unseen data. A perfect length
estimator does not predict acceptance: page_000263_table_box_id_7 has 1,532 B1
tokens yet takes 1.930 s in ordinary B1 versus 2.596 s in spec C1. In particular,
the sweep's 1,536 cutoff happens to exclude this failure and is not grounds to
claim a robust tail solution. A later full-corpus/held-out check and an actual
image-estimator cost/error study are separate tasks.

Full-precision values, all tested thresholds, source paths, and all 100 matched
records are in `analysis.json`, reproducible with `python3 analyze.py`.
