# Full versus trimmed LM head: four real-serving runs

Completed 2026-09-11 on one Ascend 910B2, physical NPU6. Runtime source
`2c04ecc7c343153403356638c9d1474e6ebad447`. These are experiment-19 serving
measurements, not isolated matrix-multiplication tests or 310P results.

## End-to-end results

Every row uses the same historical random-100 seed-1 sample/order from all
665 tables. Closed-loop client concurrency equals physical decode batch size;
there is no batch-filling wait. Latency is actual client submission to complete
response. Payload preparation and one real-request warmup precede measurement.

| Batch / concurrency | Decode head | Tables/s | Mean s | P50 s | P90 s | P95 s | P99 s | Max s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B2/C2 | Trimmed 16,384 | 2.957706 | 0.663769 | 0.394391 | 1.309363 | 1.889210 | 4.027872 | 4.277204 |
| B2/C2 | Full 103,424 | 2.833003 | 0.692523 | 0.417800 | 1.364497 | 1.993757 | 4.257685 | 4.446377 |
| B8/C8 | Trimmed 16,384 | 6.090250 | 1.174274 | 0.694363 | 2.384824 | 3.237947 | 7.461281 | 8.374919 |
| B8/C8 | Full 103,424 | 5.645766 | 1.256543 | 0.744877 | 2.606625 | 3.527710 | 8.195074 | 8.948973 |

Removing trimming reduced throughput by 4.216% at B2 and 7.298% at B8.
Mean increased 4.332% / 7.006%; P95 increased 5.534% / 8.949%.
This is one controlled run per variant, not a variance study or a sustained
open-loop capacity measurement. Decode device timing was enabled equally in all
four runs; do not compare directly to historical instrumentation-off results
as if this were another baseline reproduction.

All 400 requests succeeded and stopped at EOS. Every measured run generated
41,120 tokens in total; the throughput difference is not explained by a lower
total output-token count. Images, input token counts, KV4096, first-token
selection, scheduler and preprocessing were unchanged between head variants.

## Decode event metrics

The service summary covers **100 measured requests plus one real warmup**.
Unlike the client latency/throughput above, these aggregate graph metrics include
that warmup. Device-event intervals can include submission gaps; they are not
pure kernel-active time. Physical token slots include unused batch slots.

| Batch | Head | ms/call | Calls/s | Physical slots/s | Useful decode tokens/s |
|---|---|---:|---:|---:|---:|
| B2 | Trimmed | 1.166181 | 857.500 | 1715.000 | 1492.933 |
| B2 | Full | 1.233882 | 810.450 | 1620.901 | 1412.014 |
| B8 | Trimmed | 1.523285 | 656.476 | 5251.808 | 3570.423 |
| B8 | Full | 1.738481 | 575.215 | 4601.718 | 3147.222 |

## Outputs and vocabulary coverage

Within each batch, 98/100 native token streams and formatted outputs match.
The same two tables differ at both batch sizes:

- `page_000542_table_0`: 355 trimmed tokens versus 356 full tokens. Header text
  differs. Full output contains IDs 23464 and 29785, absent from the trimmed map.
- `page_001417_table_2`: 69 trimmed tokens versus 68 full tokens. Two Chinese
  characters differ. Full output contains ID 95474, absent from the trimmed map.

These are real text differences, not tokenization-only changes. Ground-truth
quality/TEDS was not re-evaluated here; neither variant is declared more accurate.
Across B2 versus B8 within a head, 99/100 native streams match, so this should
not be presented as universal batch-size output identity either.

## Compilation, ownership and reproduction

- 19 distinct recorded graph cache paths: 10 vision, five text prefill, four
  decode (B2/B8 x trimmed/full). The first process built 16; later variants
  reused prefills and built their respective decode graph.
- Each variant used a compilation/warmup process, then a fresh cached process
  with another real-request warmup before the 100 measured requests.
- Initial vision setup: 191.283 s; text prefill: 143.349 s. Later prefill cache
  reloads were about 5.4 s vision first-call total and 3.3 s text first-call total.
- All 331 ownership snapshots were clean; only the matching owned worker(s)
  occupied physical NPU6. Every owned server stopped; final device check was free.
- A single source fingerprint was retained throughout. No source changes or
  cache-key bypasses occurred during the run.
- Ordered-ID SHA256 (newline-joined IDs):
  `944d66f46da6db1aa55d59fe31f82d2e57f3f414911bdb2a7e7926f3f64efd0b`.
- Runtime was committed/pushed locally, deployed by checksum-verified Git bundle
  and `git pull --ff-only` after direct GitHub pull failed. No tracked remote
  files were hand-edited.
- Retrieved archive SHA256:
  `0698d4c1d2dfd28b5307793446ddbe4cf1722bf96f99874ff0ff675af13fdb29`.

`driver.py` reuses the frozen `be691de1` client and ownership/warmup harness.
Every phase preserves exact commands, readiness, server logs, client results,
service summaries, exit statuses and ownership records. Raw artifacts were
retrieved unchanged; `comparison.json` is derived with:

```sh
python3 tmp/19_table_ocr_serving/lm_head_ab_20260911/analyze.py \
  tmp/19_table_ocr_serving/lm_head_ab_20260911
```

This comparison does not choose the permanent product vocabulary. The trimmed
head remains the default pending that discussion; `--full-decode-lm-head`
selects the full head. First-token LM-head/argmax remains full and outside the
text-prefill graph in both cases. Both decode graphs return greedy native IDs.
