# B8 decode / C5 admission: extra decode capacity did not win

Runtime/client source `7d7256fc`, one910B2, physical NPU6. Ordinary crop API,
same packed-MLP/linear-patch/GC-freeze/no-decode-profile-event candidate as the
passing B2 milestone and preceding B4 development tests. Same native16384
vocabulary, greedy policy, images/token counts, KV4096 and stopping policy.
No speculative decoding or table-specific prediction/routing is involved.

Hypothesis: after B4/C5 improved occupancy but left the fifth request waiting
for a decode slot, allow that request to decode in a static B8 arena. There
are at most5 complete requests in the entire pipeline, with unused graph rows
left idle. No waiting to fill a cohort. Compare the same frozen development100
seed1 order, with a full warm request outside timing.

| Configuration | Tables/s | P95 seconds |
|---|---:|---:|
| B4/C5 control |4.742527029517923|2.3707526877638876|
| B8/C5 |4.680882905431254|2.8311503162374723|

B8/C5 completes100/100 EOS, no errors or unsent requests, wall21.363491037976928s.
Native outputs match the B4/C5 control100/100, with unchanged crop sizes and
input/real-vision-token counts. The test fails the second milestone's5 tables/s
development gate. It therefore does NOT proceed to seed2 or1000 validation.

Recorded launched-request iteration histogram: active5=29490, active4=8312,
active3=1881, active2=274, active1=1163. These are per-request counters and
include completion look-ahead; they are not a direct count of physical calls.
The result measures the complete serving effect, not an isolated attention
kernel regression. A matched real-loop profile is needed before attributing
the loss to a particular operator. Simply raising padded capacity has not
closed the measured throughput gap.

`command.txt` records the full argument array; the server launch appends
`--freeze-setup-gc --no-decode-device-timing --service-summary-output <service.json>`.
It reaches READY after65.389117s recognizer setup. Client command is
`table_closed_loop_api_client.py --api-url http://127.0.0.1:8767/v1/ocr
--set random --count 100 --shuffle-seed 1 --max-in-flight 5 --output-dir <client>`.
Warmup uses `--set warm --count 1 --max-in-flight 1` and a separate output dir.
PNG payload preparation precedes request timing; all preparation/prefill,
decode, queue/control and response latency remains included.

Host parent2023648/worker2023650 map to container2548263/2548265. Full command,
`/proc` NSpid mapping and physical ownership were manually verified. Four
periodic NPU6 snapshots bracketed by the full monitor cover the measured
window; only2023650 appears. This is sampled monitoring, not proof against an
arbitrarily short invisible outside process. Warmup/client/server exit0.
The owned parent was stopped gracefully after the test, the service summary
was saved, and `npu-smi` showed NPU6 empty. The owned monitor1972167 was also
stopped; neither it nor the server PIDs remained. No other user's process was
signalled or changed.

The next investigation should profile the remaining B4/C5 execution costs,
not continue a blind concurrency sweep. Its existing4.7425 tables/s needs
about5.4% throughput improvement while keeping P95 under3s. First milestone
evidence remains separately preserved in `table_packed_noevents_23d5518c_20260906`.
