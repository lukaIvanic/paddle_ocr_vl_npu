# Calibrated C1 and two-table speculative serving

Ascend **910B2, physical NPU 6**, 2026-09-05. Final implementation commit:
`b0a5c960`. This is an opt-in path in the existing HTTP API, not a new default.
All owned benchmark servers and monitoring loops are stopped. Direct-host checks
at **2026-09-05 15:03:40 UTC** found no process on NPU 6 and no remaining
`serve_table_speculative_api` process.

## Outcome

C1 performance was recovered in the actual two-slot implementation. C2 works,
and the tested implementation fixes improve it substantially, but **it does not
beat regular B2 on throughput/P95**. It also does not meet the broader
3 tables/s and P95 <2 s aspiration.

All columns below use the exact same 100 distinct all-corpus tables and dispatch
order. Latencies are client HTTP wall time in seconds, not composed estimates.

| Metric | Original spec API, C1 recheck | Final two-slot API, C1 | First calibrated C2 | Retained-lookahead C2 | Final background-CPU C2 |
|---|---:|---:|---:|---:|---:|
| Completed tables/s | 1.708 | 1.709 | 1.495 | 1.782 | 2.033 |
| Mean | 0.585 | 0.584 | 1.337 | 1.121 | 0.982 |
| P50 | 0.518 | 0.525 | 1.038 | 0.843 | 0.787 |
| P90 | 1.069 | 1.108 | 2.716 | 2.328 | 1.913 |
| P95 | 1.251 | 1.278 | 2.995 | 2.504 | 2.514 |
| P99 | 1.455 | 1.489 | 4.852 | 3.834 | 2.849 |
| Maximum | 2.465 | 2.596 | 8.293 | 7.220 | 5.016 |
| Requests >2 s | 1 | 1 | 21 | 15 | 8 |
| Successful / requested | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |

Directories, respectively: `original_recheck`, `async_c1`, `identity_c2`,
`retained_c2`, `async_c2`. The same two-slot worker was separately rechecked
at C1 before each C2 change: `identity_c1` (1.697 tables/s, P95 1.285 s),
`retained_c1` (1.711, 1.271 s), and `async_c1` above.

Historical anchors were 1.686509 tables/s, P95 1.281931 s, P99 1.522730 s;
the earlier restored original API (`original_fixed`) was 1.735458 tables/s,
P95 1.274381 s, P99 1.440144 s. The final original-API recheck is a fresh
contemporaneous control, not selection of the fastest original run.

### Regular B2 reference

The saved `../table_closed_loop_random100_seed1_3a745ba_20260903/b2` run has
the identical selection-file hash and dispatch order. It is a historical
reference, not a new regular-B2 rerun in this experiment.

| Metric | Regular B2 | Final spec C2 |
|---|---:|---:|
| Completed tables/s | 2.483 | 2.033 |
| Mean | 0.793 | 0.982 |
| P50 | 0.527 | 0.787 |
| P90 | 1.492 | 1.913 |
| P95 | 2.151 | 2.514 |
| P99 | 4.915 | 2.849 |
| Maximum | 5.081 | 5.016 |
| Requests >2 s | 6 | 8 |

Spec C2 improves this reference's P99, but regular B2 has the better
throughput/P95 tradeoff. All 100 final C2 requests were slower than their matched
final C1 request; the throughput gain comes from overlapping requests, not from
making individual requests faster.

## Frozen contract and correctness

- Selection hash (SHA-256 of every measured `tables.jsonl`):
  `1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85`.
  The original and speculative samples have identical IDs, order and metadata.
- 39 ordinary height-routed requests and 61 speculative requests. Height
  threshold remains 384 px; U8 snapped cuts, pixel preprocessing, overlap and
  padding are unchanged.
- Optimized Q1 IncreFA, complete-layer prefetch, RoPE lookup, and the same
  frequency-selected 16,384-row LM head/native-ID map. Target KV4096; draft
  KV768. Optimized manual post-RoPE verifier; independent K7/15/31/63, initial 15.
  No token suppression, matcher-policy changes, or generated-text re-encoding.
- **100/100 final outputs are native-token-identical** to the historical
  optimized-spec stream in every retained measured run. Final C2 also matches
  its paired C1 100/100. All measured outputs ended in EOS; no errors or
  length/cache truncations were dropped.
- In the corrected `identity_c1` and final paths, every speculative table's
  split boundaries and draft rotation match the original saved preparation.
  Small prefill/batch numerical differences may change intermediate drafts or
  acceptance counts; final-token parity alone is not proof of draft identity.
- The benchmark API still uses the baseline's pre-existing source-orientation
  metadata. This experiment did not create a general-purpose orientation
  detector or introduce a new table-specific optimization.

### Whole-pipeline concurrency

C1 means one outstanding complete-table request anywhere in the pipeline.
C2 means at most two total, including CPU preparation—not two decoding tables
plus a third being prepared. The client refills only after a response and does
not wait to pair arrivals. The server reserves preparation against the same
two slots.

Independent reconstruction from dispatch/completion intervals found peak 1
for every C1 control and peak 2 for every C2 run, with zero outstanding at end.
The final server log independently records 100 C1 admissions all at count 1,
and 100 C2 admissions only at counts 1 or 2. Warmup labels are excluded from
these counts. A long table can have several *successive* companions without
ever exceeding concurrency two.

Stable KV arenas keep separate request slots and positions. Changing K does
not copy the historical KV cache. Outputs are committed independently;
completion does not wait for the other table. Before reuse, outstanding KV
writers are drained. Prefills are prioritized when ready; otherwise the
least-recently-served request runs, joined immediately by compatible work.

## What was fixed

1. **Compiler dispatch:** B1/B8 decoder shapes shared a Dynamo code identity.
   The earlier diagnostic explicitly reported expected B8 versus actual B1.
   `96f6291c` isolates code identity per B/KV without changing model operations.
2. **Host-call overhead:** remove production per-call timing events, reuse
   RoPE views, and overlap matcher construction with target prefill.
3. **Real pinned memory:** installed torch-npu returned
   `empty_like(pinned_tensor).is_pinned() == False`. Explicit pinned result
   buffers restore genuinely asynchronous D2H. Runtime checks guard this.
4. **Q1 feedback:** restore queue-depth-one decode/token-copy overlap for
   drafts and ordinary decoding. Verifier fallback remains synchronous.
   Cache-boundary lookahead is guarded; uncommitted physical calls are
   counted separately from algorithm steps.
5. **Preprocessing identity:** the reference replaced source ID with HTTP ID
   before the baseline's orientation lookup. Exactly two speculative samples
   consequently lost their 90-degree rotation. Preserve source ID through
   preprocessing, then assign unique runtime IDs. One affected table needed
   503 verifier calls rather than the original 149; after correction it used 145.
   This is why output parity alone was insufficient for calibration.
6. **C2 phase switching:** retain pending Q1 work across independent phases
   instead of discarding it. Drain before conflicting writes, admission or
   retirement. Service-lifetime discarded lookaheads fell from 9,123 to 1,274;
   these counters include the C1 control and warmups, not just measured C2.
7. **CPU preparation:** one background worker prepares admitted requests
   while unrelated model work continues. The total table-slot cap remains two.
   This improved throughput/max, but did not materially improve P95.

The first three reference variants and the pinned-buffer variants still had
the orientation bug. Their timing records are diagnostic history, not valid
frozen-preprocessing C2 comparisons. No such run was used to authorize C2.
C1 was established using the corrected `identity_c1` first.

## What actually batches

Final measured C2, excluding warmups:

| Work | Singleton graph calls | Paired graph calls | Request-steps in paired calls |
|---|---:|---:|---:|
| Draft (B8 / B16 Q1) | 5,342 | 1,377 | 34.0% |
| Ordinary (B1 / B2 Q1) | 3,961 | 637 | 24.3% |
| Verifier (B1 / B2, all Q including fallback) | 4,539 | 240 | 9.6% |

A paired graph advances two independent requests; its time is counted once
globally and once in each participant's latency. Graph sizes and attention
implementations are recorded in the service metadata.

Different phases still run separately. Two verifiers with different Q also run
separately; neither request is forced to change its K. The phase ledger records
both mixed-phase exposure and same-phase/different-Q fragmentation. These are
**host scheduling intervals**, not an isolated device profile or claims of
simultaneous NPU-stream execution.

## Final tail

| Table | Final C1 (s) | Final C2 (s) | Added latency (s) |
|---|---:|---:|---:|
| page_000263_table_box_id_7 | 2.596 | 5.016 | 2.420 |
| page_000921_table_9 | 1.195 | 2.827 | 1.632 |
| page_000275_table_box_id_1 | 1.346 | 2.663 | 1.317 |
| page_000272_table_box_id_1 | 1.478 | 2.595 | 1.118 |
| page_000261_table_box_id_3 | 1.334 | 2.573 | 1.240 |
| page_000217_table_box_id_0 | 1.278 | 2.511 | 1.233 |
| page_001270_table_box_id_1 | 1.243 | 2.090 | 0.847 |
| page_000199_table_box_id_0 | 1.179 | 2.066 | 0.887 |

For the five requests above the measured P95, foreign preparation/prefill
action time is 0.465–0.971 s, and other foreign action time is 0.338–1.003 s.
The longest request spends about 1.950 s in its verifier host actions,
plus 0.971 s in foreign preparation/prefill and 1.003 s in other foreign actions.

These explain measured contention, not removable latency estimates. Residual
client time (0.098–0.511 s for these five) includes pre-admission/HTTP delay and
uninstrumented orchestration; it is **not subtracted** or relabeled as kernel
time. Background CPU spans overlap main-thread actions and are non-additive.
Prefill execution remains serialized on the NPU. Further gains would require
a separate scheduling/batching/prefill investigation, not another demonstrated
small wrapper bug. No further experiment was started.

## Reproduction and evidence

Each new measured directory has `command.txt` with the source commit and
exact invocation. The final serving command is the existing
`serve_table_speculative_api.py --interleaved-tables 2 --allow-compile`;
the original API remains default when the interleaved flag is omitted.

For each server: two complete external warm requests, then the unchanged
`table_closed_loop_api_client.py --set random --count 100 --shuffle-seed 1`
at max-in-flight 1. C2 is run only after its C1 check, with two external C2 warm
requests before its measured max-in-flight 2 run. The pinned C1 repeat is retained
as an additional same-process repeat. The contemporaneous original control
disables its extra built-in cold request (`--cold-request-id ''`) and uses the
same two external full-request warmups. The earlier historical/original_fixed
controls had an additional built-in cold request; that difference is not hidden.

All model compilation/cache loading and warm requests are outside measured
client timing. Crop/payload construction follows the unchanged crop-in-RAM
client contract. Server PNG decoding, CPU preprocessing, queueing, prefill,
decode, host control and response completion remain in client latency.
JSONL results are flushed after each response. No requests/outliers were
removed; development controls are retained. The earlier cold `control/`
request records remain on the validation container; its server/client logs and
service metadata are included here. It is not used as a calibrated comparison.

`comparison.json`, generated by `analyze.py`, contains full-precision
distributions, selection hashes, interval concurrency checks, native parity,
all 100 matched C1/C2 latency changes, tail accounting and batching counters.
`*_service.json` holds actual graph contracts and service-lifetime counters.
`*_server.log`, `*_client.log`, warmup directories and `host_logs/` preserve
the raw evidence. Eight direct-host monitor logs show only the expected owned
worker PID in every sampled NPU 6 snapshot; sampling was approximately every
five seconds, not a claim to detect arbitrarily short unseen jobs.
Manual checks established the owning parent/worker and device freedom before
and after each server.

CPU validation: 13 scheduler/acceptance/accounting tests, five Q1
pipeline/identity tests, and one existing API metrics test pass. These tests
validate control logic, not NPU performance; the real 100-table runs supply
inference evidence. No custom kernel, model math or new inference-policy
optimization was introduced.
