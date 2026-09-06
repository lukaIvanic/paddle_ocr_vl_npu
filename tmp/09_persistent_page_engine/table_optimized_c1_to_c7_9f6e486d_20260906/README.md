# Optimized ordinary table serving: 1,000-request batch/concurrency sweep

2026-09-06, one Ascend 910B2, physical NPU6. Inference commit
`9f6e486d`. Requested order: B1/C1, B2/C2 reconfirmation, B3/C3, B4/C4,
B6/C6, B7/C7, B8/C7, then the user-added B8/C8.

| Configuration | Tables/s | Mean (s) | P95 (s) | P99 (s) | Max (s) |
|---|---:|---:|---:|---:|---:|
| B1/C1 | 1.917 | 0.521 | 1.657 | 3.363 | 3.430 |
| B2/C2 | 3.302 | 0.604 | 1.953 | 3.897 | 4.195 |
| B3/C3 | 3.931 | 0.760 | 2.472 | 5.064 | 5.548 |
| B4/C4 | 5.128 | 0.776 | 2.540 | 5.109 | 5.612 |
| B6/C6 | 5.943 | 0.997 | 3.257 | 6.556 | 7.884 |
| B7/C7 | 6.508 | 1.059 | 3.472 | 7.016 | 8.021 |
| B8/C7 | 6.468 | 1.065 | 3.583 | 7.118 | 7.992 |
| B8/C8 | 6.986 | 1.124 | 3.768 | 7.673 | 8.544 |

Each row is one complete run, not selected percentile minima. QPS means achieved
closed-loop completed responses per second, including initial fill and final
drain. This is not an offered-QPS arrival test. B2 reconfirms the earlier
3.311–3.312 tables/s, P95 1.941–1.944 s results.

## Fixed contract

- Production `serve_crop_ocr_api.py` and `table_closed_loop_api_client.py`.
- Same seed-3 manifest in every lane: all 665 tables shuffled once, then
  335 distinct tables sampled from that same corpus.
- Manifest SHA256:
  `fcf572b443303fb449913a12d58989eaf59b5e563e45fa6dac6629e194b7fa62`.
- `combined_apply_complete_layer_prefetch1_rope_lut_packed_mlp`, IncreFA,
  FP16, greedy ordinary argmax, same native-ID 16,384-row vocabulary map.
- Linear vision patch projection, vision attention weight padding,
  setup GC collection/freeze with GC still enabled, optional decode events off.
- Vision buckets 256,384,512,640,768,1408,1920,2048,2944,4096;
  text buckets 128,256,512,1024,1152; unchanged 28,224/802,816 pixel limits.
- KV4096, max-new-tokens4096, no interruption cap, no routing/proposals.
- A full-request warmup before each measurement. CPU payload preparation,
  model load and compile outside timing. Full HTTP submission-to-response
  latency includes server-side preparation, prefill, decode, control and output.
- Server loaded separately for B1/B2/B3/B4/B6/B7/B8. B8/C7 and B8/C8 share
  the same server and compiled graph, with another warmup before C8.
  Their combined service summary is under `b8c7/service.json` (2,002 requests).

## Audit and outputs

`analysis.json` and `analyze.py` verify the full manifest, all eight configuration
contracts, actual in-flight timelines, identical input shapes/vision-token
counts, and direct-host NPU ownership samples. Maximum sampled ownership gap
was four seconds; every sampled measured window contained only the identified
owned worker. Parent and worker PIDs were manually inspected between runs.

Every lane returned 1,000 responses with no HTTP errors: 988 EOS and the same
12 KV4096-cap stops. All 1,000 remain in latency and throughput denominators.
EOS-only throughput is recorded separately in `analysis.json`.

C1, C2 and C3 match all 1,000 native token streams exactly. Relative to C1,
B4 differs on 9 requests, B6 on 4, and B7/B8C7/B8C8 on 7 each. These are
recorded numerical-output differences, not a claim of newly evaluated Page-TEDS
parity. No generated output was re-encoded. Native outputs are preserved in
each `measured/results.jsonl`; progress was flushed immediately per response.

All warmup/client/server exit codes are zero. All owned inference jobs and the
read-only monitor were stopped. `final_cleanup.log` confirms NPU6 free at
2026-09-06 20:28:56 CST. No follow-on experiment was launched.

The workbook's latest block includes these eight rows plus the previously
selected higher-P95 B5/C5 run. B2's new control is the highest-P95 of its three
clean validations, so its complete row replaces the prior selected B2 row.
