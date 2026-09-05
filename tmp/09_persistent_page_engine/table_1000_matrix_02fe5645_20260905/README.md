# Optimized ordinary table serving: matched 1,000-request comparison

Requested 2026-09-05. This is the explicit six-lane comparison, not a claim
that the separate serving goal's staged validation has passed.

## Fixed protocol

- Ascend 910B2, physical NPU 6 only. One API/model worker at a time.
- Inference source: `02fe5645b53f03c8bde0fd4284a3c2ac0c1649de`.
- Ordinary decode: B1/C1, B2/C2, B4/C3, B4/C4, B8/C8, B16/C16.
- Same `table_closed_loop_api_client.py --set random --count 1000
  --shuffle-seed 3` input sequence in every lane. All 665 distinct tables,
  then 335 distinct selections from the same corpus. Manifest SHA256:
  `fcf572b443303fb449913a12d58989eaf59b5e563e45fa6dac6629e194b7fa62`.
- Independent requests, immediate response-driven refill up to C. Static decode
  batch B retains idle slots. No oracle, interruption cap, or speculative route.
- Crop PNG payloads are prepared in RAM before timing. Submission-to-response
  latency includes all server-side work and waiting. Whole-run throughput includes
  initial filling and final drain. No artificial per-request delay.
- Setup/cached graph loading and one complete `--set warm --count 1` request
  are outside timing. Every response and its native token IDs are flushed to JSONL.

## Locked model and execution

`serve_crop_ocr_api.py`, PaddleOCR-VL-1.6 FP16, TorchAir,
`combined_apply_complete_layer_prefetch1_rope_lut`, greedy native-ID selection,
`presets/table_compact_vocab/b1_verifier_topfreq_16384.json`, KV4096 and output
limit 4096. Input bounds 28,224 / 802,816 pixels, unchanged crop pixels and
resampling. Scheduling metrics and normal decode device events are enabled.

Vision buckets: 256,384,512,640,768,1408,1920,2048,2944,4096.
Text buckets: 128,256,512,1024,1152. Vision attention-weight padding is enabled.
The current CPU image decoding/preparation overlap and bit-exact normalization
lookup are retained. Compact control, disabled device events, loop-boundary
prefetch, oracle admission and interruption caps are not enabled.

## Results and audit

Run `python analyze.py` after copying completed lane records and the host monitor.
`analysis.json` retains unrounded percentiles, rates, output stop reasons, input
and concurrency checks, token differences and observed NPU ownership.

The first B1 run completed 1,000 responses in 573.0199595440645 s:
1.7451399089059154 responses/s, P95 1.8107100192282815 s. There were no HTTP
errors. 988 requests ended at EOS and 12 reached the unchanged KV4096 boundary
(eight unique tables, four repeated). All 335 repeat streams were token-identical.
All capped responses remain in the latency distribution. A returned-response
rate is not proof that every OCR output completed normally, nor goal acceptance.

The initial terminal disruption occurred before measured requests. Its failed
preload and server-start logs are preserved; no measured subset was discarded.
The B1 measurement used `server_restart.log` and `measured/`, not the aborted
`client/` directory.

B2 completed 1,000 responses in 337.8616728449706 s:
2.9597911819339586 responses/s, P95 2.182853424537461 s. All 1,000 native
streams exactly matched B1, including the same 12 KV-limit stops. All 335
repeated B2 streams were identical. Input crop dimensions, input tokens and real
vision-token counts matched B1 for every occurrence. The client peak was two.

B4/C3 completed 1,000 responses in 263.3439779350301 s:
3.797314857325923 responses/s, P95 2.622647292126202 s. Stop counts remain
988 EOS / 12 KV-limit. Six occurrences (five unique tables) differ in native IDs
from B1, with unchanged stream lengths. Direct raw-output review found:

- `page_000766_table_2`: one space after a math delimiter.
- `page_001035_table_28`: fewer leader dots in one label.
- `page_000635_table_0` (twice): different native IDs, identical raw decoded text.
- `page_000281_table_box_id_1`: one extra closing parenthesis.
- `page_000496_table_4`: two occurrences of `Rhysotritia` become `Rhyosritia`.
  This is a name spelling change, not a formatting-only difference.

The model/options and input shapes are unchanged apart from batch geometry;
these small deterministic differences are consistent with batch-dependent
numerical drift. Logits were not captured, so that causal explanation is not
independently proved. No output was dropped, normalized for comparison, or
re-encoded. All 335 within-B4/C3 repeated streams reproduce exactly.

B4/C4 used the same warmed B4 server after a separate full-request warmup,
changing only the client cap. It completed 1,000 responses in
213.96090900502168 s: 4.673750941937389 responses/s, P95
2.790147225355028 s. All 1,000 streams exactly match B4/C3, with the same stops.
The B4 service summary covers both client runs and both warmup requests, so its
lifetime timing counters must not be attributed to either client run alone.

B8/C8 completed 1,000 responses in 156.94978585198987 s:
6.371464571114747 responses/s, P95 4.142976103140971 s. Stops remain
988 EOS / 12 KV-limit. Only `page_000521_table_2` differs from B1 (two
occurrences): full-width parentheses replace ordinary parentheses and two
literal line breaks disappear. Cell wording/numbers remain unchanged; 435
native tokens become 431. The other 998 streams match B1. All repeats match.

B16/C16 completed 1,000 responses in 125.66911503497977 s:
7.957404647288651 responses/s, P95 6.435093457967739 s. Stops remain
988 EOS / 12 KV-limit. Three streams differ from B1: the two occurrences of
`page_000521_table_2` described for B8, plus the single space after the formula
delimiter in `page_000766_table_2` described for B4. All 335 repeat streams match.

| Lane | Mean (s) | P50 (s) | P90 (s) | P95 (s) | P99 (s) | Max (s) | Responses/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1/C1 | 0.573 | 0.357 | 1.245 | 1.811 | 3.700 | 5.060 | 1.745 |
| B2/C2 | 0.674 | 0.411 | 1.528 | 2.183 | 4.435 | 4.684 | 2.960 |
| B4/C3 | 0.787 | 0.484 | 1.740 | 2.623 | 5.279 | 6.044 | 3.797 |
| B4/C4 | 0.851 | 0.512 | 1.962 | 2.790 | 5.694 | 6.403 | 4.674 |
| B8/C8 | 1.232 | 0.719 | 2.822 | 4.143 | 8.466 | 9.357 | 6.371 |
| B16/C16 | 1.933 | 1.122 | 4.511 | 6.435 | 13.823 | 15.154 | 7.957 |

All six manifests are byte-identical. The admission-cap reconstruction passes
for every lane. All frozen non-batch model settings match; crop dimensions,
input-token counts and real vision-token counts match B1 for every occurrence.
Zero HTTP errors, timeouts or unsent requests occurred. No lane satisfies both
3 responses/s and P95 below 2 s, even before accounting for incomplete outputs.

## Ownership and cleanup

Manual host-to-container PID mappings are in `host_npu6_monitor.log`. Each
measurement is bracketed by monitoring and every observed in-window snapshot
contains only its expected worker. There are 104/61/48/38/28/23 in-window
samples for the six lanes, respectively; maximum observed sampling gap is 6 s.
Sampling cannot exclude an unobserved sub-interval process.

Only the identity-checked owned API parents received SIGTERM. After B16, the
host at 2026-09-06 06:01:07 CST reported NPU6 free. The final API/worker and
owned monitor PIDs are absent; connecting to container port 8767 returns
ECONNREFUSED (111). The container lacks `ss`, so port release was checked with
a direct socket connection instead. No other user's process was signalled.

## Workbook

The prior workbook's temporary directory was missing. `workbook.mjs` reconstructs
the user's supplied screenshots and historical vLLM data, preserving their visible
custom values and rounding. It appends the new comparison below the old tables.
The larger new workload is explicitly distinguished from the historical random
100; no controlled speedup is claimed between unmatched workloads.

The workbook is saved durably under
`outputs/01a0735d-f277-7262-b1d0-b87d6db95456/Table OCR latency comparison.xlsx`.
