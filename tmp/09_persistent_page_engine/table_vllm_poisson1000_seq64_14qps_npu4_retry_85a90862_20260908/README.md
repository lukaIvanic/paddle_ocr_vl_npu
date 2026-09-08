# vLLM-Ascend: matched 1,000-request Poisson sweep

Completed 2026-09-08 on one physical NPU4, Ascend 910B2. Runner source commit
85a90862. Setup began 19:55:24 CST; final server shutdown and NPU release completed
21:49:41 CST (13:55:24–15:49:41 Zagreb), 1 h 54 min 17 s total.

## Contract and validity

- PaddleOCR-VL-1.6, FP16, vLLM 0.23.0+empty / vLLM-Ascend 0.23.0rc1.
- Maximum active sequences 64; shared per-forward token budget 16,384;
  per-request prompt-plus-output context 4,096. Dynamic execution batches.
- FULL_AND_PIECEWISE compilation, captured decode batch sizes 1 through 64;
  async scheduling and chunked prefill enabled.
- Prefix caching disabled and multimodal processor cache zero. Image preprocessing
  retains min_pixels=28224 and max_pixels=802816. No resolution changes.
- One server load, two complete real warmups outside measurement. Each rate has
  1,000 requests; all responses drain before starting the next rate.
- Exact saved global-shuffle sequence from the custom Pareto run, covering all
  665 distinct table IDs. Replays b2/qps1's schedule, scaling offsets by the rate.
  All original custom schedules agree with this scaling within 6.9e-13 seconds.
  Sequence hash: `97a1f87dd18ace0833f6d66796ce6868845b04880292d0f632f3575c3693caa9`.
- All 14,000 requests completed: zero HTTP/client errors; zero KV preemptions;
  10 context-limit stops in each run, always the same request occurrences as
  the matched custom B2/QPS1 control. These stops remain in metrics and accuracy.
- All 1,908 recorded NPU-ownership snapshots were uncontaminated. Final snapshot
  contains no device process; direct npu-smi and container process checks agree.
- Earlier NPU4 preflight failure and interrupted NPU6 attempt are excluded.

## Latency and throughput

Latency starts at the actual client request attempt and ends on full HTTP
response, including queueing. Scheduled-arrival latency and dispatch lag are also
saved. Input crops and JSON payloads are prepared outside the arrival timeline in
both implementations, as in the existing custom benchmark. This is an endpoint
latency benchmark, not page detection/cropping latency.

Completed throughput is 1,000 divided by first-window-start to last response,
including drain. Offered QPS is not achieved throughput or proof of stable
capacity. The shared Poisson realization yields actual arrival rates about
96.05% of target (e.g. 7.684 at target8); dispatch lag is separately recorded.

| Offered QPS | Completed tables/s | Mean s | P95 s | P99 s | Max s | Peak arrivals in rolling 1s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.958 | 4.340 | 14.917 | 30.686 | 37.124 | 6 |
| 1.5 | 1.431 | 4.823 | 16.966 | 33.655 | 38.339 | 7 |
| 2 | 1.899 | 5.340 | 18.622 | 38.425 | 41.772 | 8 |
| 2.5 | 2.344 | 6.172 | 21.505 | 45.691 | 49.726 | 8 |
| 3 | 2.779 | 7.129 | 25.829 | 52.861 | 58.984 | 10 |
| 3.5 | 3.197 | 8.243 | 28.624 | 62.459 | 69.686 | 11 |
| 4 | 3.594 | 10.740 | 38.059 | 82.260 | 91.364 | 12 |
| 4.5 | 3.975 | 14.827 | 47.618 | 94.802 | 112.752 | 13 |
| 5 | 4.141 | 18.182 | 51.571 | 99.320 | 112.616 | 14 |
| 5.5 | 4.170 | 25.287 | 60.352 | 106.814 | 121.298 | 14 |
| 6 | 4.256 | 30.344 | 67.024 | 109.332 | 124.552 | 15 |
| 6.5 | 4.086 | 41.021 | 87.388 | 126.004 | 141.259 | 16 |
| 7 | 4.267 | 42.545 | 86.274 | 121.123 | 137.517 | 16 |
| 8 | 4.364 | 48.098 | 94.804 | 126.506 | 142.294 | 19 |

Mean native queueing is below 0.004 s through target4, then grows to 1.976 s
at4.5, 5.108 s at5, 11.514 s at5.5, 16.713 s at6, 26.653 s at6.5,
28.660 s at7 and 34.405 s at8. At8, outstanding requests peak at408 and the
server needs 99.017 s to drain after the final arrival.

Thus throughput levels near 4.1–4.4 tables/s under heavier offered load while
latency and queues grow substantially. The roughly 100-request screening result
was not an adequate estimate of the longer-run latency distribution.

Matched custom Pareto anchors (best measured custom P95 at each offered rate):

| Offered QPS | Custom batch | Custom P95 s | vLLM P95 s | Custom completed tables/s | vLLM completed tables/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1.908 | 14.917 | 0.960 | 0.958 |
| 3 | 4 | 2.479 | 25.829 | 2.880 | 2.779 |
| 6 | 8 | 3.776 | 67.024 | 5.734 | 4.256 |

Custom source: `outputs/poisson-frontier-20260907/results.json` at repository
root. This compares the existing custom per-rate frontier against one fixed
vLLM configuration; it is not a same-batch-size comparison.

## Evidence and accuracy

`results.json`/`results.csv` contain exact aggregate metrics. Each `budget16384/qps*`
folder contains its command, immediate per-request records, native generated IDs,
text, schedule, summaries, logs, and before/after native server metrics.

No generated text was encoded. Versus the custom B2/QPS1 control, the vLLM1-QPS
run matches 952/1,000 native streams and 959/1,000 raw texts. Across vLLM rates,
0–20 raw outputs differ from the vLLM1-QPS reference; capped request sets are
identical. Ground-truth results are recorded in `accuracy/comparison.json`.

Official scoring completed: 706 unique GT/prediction pairs, zero scoring errors
and zero timeouts. Repeated request occurrences remain in the aggregation;
these are matched-sequence scores, not a deduplicated canonical 665-table score.

| Metric | Custom B2/QPS1 | vLLM range across 14 rates |
|---|---:|---:|
| Page-TEDS | 95.43735% | 95.41996–95.44322% |
| Sample-mean TEDS | 95.01588% | 95.00986–95.03894% |
| Page structure-TEDS | 97.78462% | 97.75898–97.78791% |

Aggregate accuracy is essentially unchanged, but individual outputs are not
identical. The largest observed per-table TEDS decrease is
`page_000273_table_box_id_1`: 98.730% custom versus 82.941% vLLM in the
QPS4 run, with merged-cell markers changing `<lcel>` to `<ecel>` and `<xcel>`
to `<ucel>`. This changes normalized HTML structure, not merely tokenization.
The largest increase is `page_001398_table_box-7mwax454`: 39.244% to 57.926%,
with different cell delimiters and dotted content. Other differences include
punctuation, text, and cell placement; all per-run score deltas and raw-text
diff spans are preserved. The weather-table minus signs remain a small
vLLM improvement. Small aggregate differences do not prove every individual
difference harmless or establish its numerical cause.

The existing accuracy helper is
`../table_vllm_poisson_sweep100_budgets_qps1to8_324dd85a_20260908/compare_accuracy.py`.
Use `--count 1000 --run-dir <this-run>` for both stages: `--prepare-only` under
pipeline_py312, then `--score-prepared` under omnidocbench_py310. Official output
postprocessing and the existing OmniDocBench scorer are reused; raw IDs/text
remain unchanged. Scoring starts after measured inference and server shutdown.
