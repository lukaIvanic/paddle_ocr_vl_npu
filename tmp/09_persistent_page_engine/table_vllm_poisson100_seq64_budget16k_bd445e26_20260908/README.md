# vLLM-Ascend: sequence capacity and Poisson arrivals

All eight 100-request tests completed successfully on physical NPU4 (910B2),
2026-09-08 19:07:10–19:19:13 CST including setup. Model PaddleOCR-VL-1.6;
vLLM 0.23.0+empty / vLLM-Ascend 0.23.0rc1. Runner commit bd445e26.
The NPU was free before launch and after shutdown; all 210 ownership samples
were uncontaminated. No request errors, truncations, or preemptions.

## Matched comparison

Both columns use a 16,384 shared forward-token budget, 4096 per-request context,
async scheduling, chunked prefill, FULL_AND_PIECEWISE compilation, and no
cross-request prefix/processor cache reuse. All integer decode batch shapes
through each cap are captured. Images and preprocessing settings are unchanged.
Maximum active sequences is an admission cap, not a fixed execution batch.

Each test replays the exact same first100 globally shuffled requests and scales
the same exponential arrival intervals by offered QPS. Sequence fingerprint:
`d874b593a394c604c833bde5530cdb6f6454e99f6d4b5dc3262ab6ed3703fb3d`.
Two complete warmup requests per server precede timing. Each measured test drains
fully before the next. Latency is actual client request attempt to complete HTTP
response, including connection and queueing; scheduled-arrival latency is also saved.

| Offered QPS | Completed tables/s, max16 | Completed tables/s, max64 | P95 s, max16 | P95 s, max64 |
|---:|---:|---:|---:|---:|
| 1 | 0.917 | 0.915 | 10.916 | 11.353 |
| 2 | 1.536 | 1.535 | 13.099 | 13.049 |
| 3 | 1.919 | 1.921 | 13.722 | 13.445 |
| 4 | 2.135 | 2.196 | 15.836 | 15.881 |
| 5 | 2.139 | 2.420 | 17.351 | 17.004 |
| 6 | 2.183 | 2.516 | 17.868 | 18.232 |
| 7 | 2.228 | 2.628 | 17.883 | 18.760 |
| 8 | 2.220 | 2.701 | 19.472 | 18.854 |

Completed throughput counts the whole finite window including drain; these are
not claims that sustained arrivals at offered QPS are supportable. At offered8,
max64 still drains for 24.387 s after the last dispatch.

At offered8, mean latency decreases 10.889→8.404 s and maximum decreases
34.902→26.891 s. Native average queue time decreases 5.9385→0.0267 s/request.
Thus the larger admission cap helps throughput and mean/max latency, but P95
does not improve consistently. Server logs are periodic snapshots, not exact
occupancy peaks; saved request events provide exact client outstanding counts.
The max64 run reaches 68 outstanding requests at offered8.

## Why the 16k budget

The prior [24-run budget sweep](../table_vllm_poisson_sweep100_budgets_qps1to8_324dd85a_20260908/results.json)
compared 4096/8192/16384 at max16. It showed no consistent decisive winner, so
16k follows the user's agreed tie-break. All 24 runs completed without errors,
truncations or preemptions; 642 ownership samples were uncontaminated.

## Accuracy: same 100-request subset

Compared with the custom B2/QPS1 run's matched first100 outputs (96 distinct
table IDs), all 32 vLLM tests have identical decoded text and identical TEDS.
Some use different native token sequences for identical text. No generated text
was encoded. Each unique (table ID, predicted HTML) pair was scored once using
the existing OmniDocBench scorer: 97 pairs, zero errors/timeouts. Scores below
retain all100 request occurrences and aggregate Page-TEDS by source page.
These are subset results, not full-corpus scores.

| Metric | Custom pipeline | All vLLM configurations |
|---|---:|---:|
| Sample TEDS | 95.129504% | 95.133959% |
| Page-TEDS | 95.701839% | 95.703628% |
| Sample structure-only TEDS | 98.253009% | 98.253009% |
| Page structure-only TEDS | 98.052324% | 98.052324% |

99/100 decoded outputs match custom exactly. The one substantive difference is
`page_001375_table_4`: vLLM retains nine minus signs missing from custom, all nine
agree with GT. This table's TEDS improves 96.598822%→97.044291% (+0.445469 pp).
Both systems still miss other signs. Tokenization-only differences occur in
`page_001175_table_2` without changing decoded text or TEDS.

Detailed scores, diffs, and the reproducible CPU-only comparison script are in
the [prior sweep's accuracy folder](../table_vllm_poisson_sweep100_budgets_qps1to8_324dd85a_20260908/accuracy/comparison.json).
Prepare with the pipeline Python using `compare_accuracy.py --prepare-only
--extra-run tmp/09_persistent_page_engine/table_vllm_poisson100_seq64_budget16k_bd445e26_20260908`,
then run `--score-prepared` using `/workspace/venvs/omnidocbench_py310/bin/python`.
Scoring ran after all measured inference and server cleanup, without installing
dependencies or altering inference environments.
