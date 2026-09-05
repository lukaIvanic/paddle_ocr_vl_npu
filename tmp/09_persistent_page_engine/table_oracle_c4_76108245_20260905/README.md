# Four-table speculative serving with the >=1,024-token oracle

Ascend 910B2, physical NPU 6. Inference source: `76108245`.
Measured on 2026-09-05 UTC (2026-09-06 CST). Same random-100 all-corpus
sample/order and oracle as the preceding C2 experiment. No model/routing policy
change beyond enabling four simultaneous complete table requests.

## Result

| Metric | Previous C2 server | Four-slot server at C2 | Four-slot server at C4 |
|---|---:|---:|---:|
| Completed tables/s | 2.470 | 2.480 | 2.862 |
| Mean | 0.809 s | 0.805 s | 1.393 s |
| P50 | 0.615 s | 0.609 s | 1.007 s |
| P90 | 1.630 s | 1.606 s | 2.868 s |
| P95 | 1.995 s | 1.997 s | 3.418 s |
| P99 | 2.356 s | 2.368 s | 4.199 s |
| Maximum | 4.707 s | 4.758 s | 7.180 s |
| Requests >2 s | 5 | 5 | 26 |
| Speculative / ordinary | 8 / 92 | 8 / 92 | 8 / 92 |
| Successful, EOS-terminated | 100 | 100 | 100 |
| Native outputs identical to previous C2 | 100 | 100 | 100 |

The four-slot implementation reproduces C2 first. Raising client concurrency
to four then improves completion throughput by 15.4%, but worsens P95 by 71.1%.
This implementation does not meet either 3 tables/s or P95 <2 s. This is not a
claim about an upper limit on speculative serving or about a fully optimized
non-adjacent batching implementation.

Historical regular B4 on the identical sample/order achieved 3.241 tables/s,
P95 3.350 s, P99 7.289 s, maximum 7.785 s. Current oracle C4 improves that
historical P99 but not throughput/P95. Regular B4 was not rerun here.

## Observed bottleneck evidence

All 1,818 measured verifier request-steps at C4 still used B1 graphs: Q1 203,
Q8 1,202, Q16 177, Q32 109, Q64 127. The corresponding C2 counts are identical.
Thus this sparse eight-speculation workload did not obtain batched verification
in the measured C4 run. That does not mean the B4 verifier is unimplemented:
the four-speculative-request warmup exercised padded multi-request verification.

Ordinary B4 scheduled steps in the measured C4 run:

| Real requests in the B4 step | Scheduled steps |
|---|---:|
| 4 | 2,970 |
| 3 | 4,014 |
| 2 | 352 |

59.5% of these steps used padding. These counts are separable from the service
lifetime because C2 cannot use B4 ordinary steps and all warmup requests were
speculative. They are logical scheduled steps, not the additional uncommitted
Q1 lookahead graph launches. Measured C4 draft request-steps were 2,154 in B8
and 356 in padded B32; do not claim the fully useful B32 warmup as measured
four-request draft batching.

The worst request, `page_000263_table_box_id_7`, changed from 4.758 s at C2 to
7.180 s at C4. Its own attributed host actions stayed nearly unchanged:
2.949 -> 2.934 s. Foreign preparation/prefill rose 0.358 -> 1.422 s, and other
foreign actions rose 1.171 -> 2.532 s. Contention from companions clearly
increased. These are host-action attribution fields, not isolated kernel
measurements; background CPU spans and device events must not be added again.

Prefix preservation is not free. Across the entire service lifetime (C2,
C4, and warmups), target preservation ran on 4,547 physical calls and copied
248,868,864 bytes total, while draft preservation ran on 344 calls and copied
158,662,656 bytes. Those byte counts include save plus restore. No new profiler
was run, so neither launch overhead nor its contribution to P95 has been
isolated. Further improvement needs a separately scoped investigation.

## Implementation contract

- Four complete requests maximum, including queued background CPU preparation;
  the client refills after individual responses, without a cohort requirement.
- Existing greedy selection, compact vocabulary, height rule, frozen oracle
  threshold, U8 cuts, KV4096 target / KV768 draft, and independent adaptive
  K7/15/31/63 remain unchanged. No generated text encoding or saved-token replay.
- Shared graph sizes are powers of two: target B1/B2/B4 and draft B8/B16/B32.
  Verifiers retain Q8/Q16/Q32/Q64. Different phases/queries still run separately.
- Compatible non-adjacent owners use the smallest covering contiguous cache
  view. Example: slots 0 and 3 use B4, with dummy slots 1 and 2. Slots 1 and 2
  use a contiguous B2 view and need no padding. Three owners use B4.
- Dummy rows use position zero. Their first Q KV positions are saved and
  restored around every graph launch, including uncommitted Q1 lookahead.
  This protects unrelated active requests without moving full historical KV
  caches. Save, graph, restore and subsequent compute share the compute stream;
  restoration is enqueued before the output event. Real slots/positions and
  acceptance remain independent. No custom operator or model math was added.
- Prefix buffers are allocated during setup. View lists are cached by the
  protected slots/query. Contiguous C2 execution has no prefix-copy work.
- The original API remains default; four-slot serving is explicitly opt-in.

## Run order and evidence

One resident four-slot server. Additional shape initialization covered B4 Q1,
B32 Q1, and B4 Q8/Q16/Q32/Q64 before READY. No separate synthetic benchmark was
used as a performance result.

1. Two complete C2 warmup requests using `--set a --count 2`.
2. Identical random-100 C2 calibration; inspect performance and native parity.
3. Four complete C4 warmup requests using `--set a --count 4`.
4. Identical random-100 measured C4, then graceful shutdown.

Warmup requests finish under the normal API stop rules; three of the four
long C4 warmup requests reach the existing KV limit, not EOS. The first two
warmup outputs match their C2 warmup tokens exactly. No warmup latency is used
in the measured distribution. All 100 measured requests in each run end in EOS.
Warmup and measured client commands, source commit and machine are recorded in
`c2/command.txt` and `c4/command.txt`. No per-table untimed preparation occurs
inside measurement. The unchanged client preloads crop payloads before timing;
server image decoding, preparation, waiting, prefill, decode and response remain
inside client wall latency. Progress flushes after every response.

Selection SHA-256 for both runs and the previous control:
`1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85`.
Client intervals and 100 server admissions per measured run independently
confirm caps 2 and 4 respectively. Every measured route matches the prior
1,024-token oracle route. Outputs are 100/100 native-token-identical.

Direct-host NPU 6 owner was worker PID 1728658, parent 1728656; corresponding
container server PID was 2448277. Manual checks identified the process, and
the five-second-sleep monitor covers both measured windows without another NPU
owner. Sampling is not continuous process tracing. The server exited cleanly;
NPU 6 was free at 2026-09-05 16:45:56 UTC. The owned monitor was then stopped.

CPU tests passed: 14 scheduling tests (including all C4 slot subsets), seven
pipeline/identity tests (including dummy-prefix restoration and Q1 lookahead),
and three API/oracle tests. CPU tests do not establish NPU performance or
bytewise NPU KV parity; the real-run native-output comparison is separate.

`analyze.py` regenerates full-precision distributions and all per-table changes
in `comparison.json`. Raw client/server logs, warmups, service graph contracts,
batch composition and prefix-copy counters are retained. Service counters
include warmups and C2 unless specifically separated above.
