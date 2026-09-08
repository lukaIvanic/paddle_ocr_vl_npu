# Detailed hybrid timing — one 910B2

Inference commit `4a01bda5`, 2026-09-09, physical NPU 6 on
`liteserver-c001-4`. Instrumentation only: no scheduling, CPU worker-count,
NPU-concurrency, model, resolution or cache-root changes. Reference:
`de71a457` staged-frontend run.

## Validation

- 29 local policy/timing tests passed with Python 3.13.
- First-64 timing-on/off controls: **52.237 / 51.782 seconds**.
  Observed on/off wall difference **+0.88%**, one pair on a shared machine,
  not a statistically established overhead estimate.
- Both controls preserve all 1,015 crop token sequences, text and stop reasons.
  Timing-on also matches the previous 64-page reference.
- Full run: exit 0, **1,651 pages / 637.722 seconds = 2.5889 pg/s**.
  Setup: 41.805 seconds. Diagnostic trace export: 6.714 seconds, outside
  processing; setup and post-run diagnostic reporting are not in processing pg/s.
- Full 30,557 crop IDs, metadata, token sequences, text and stop reasons match
  the accepted reference exactly: 28,125 UniRec and 2,432 Paddle requests.
- All 1,651 page JSON files and all 1,651 Markdown files match byte-for-byte.
  This is output parity, not a new ground-truth accuracy evaluation.
- Decode calls remain 14,933 UniRec / 9,217 Paddle; useful UniRec slot
  utilization 88.869%, active Paddle slot utilization 69.650%.
- Peak Torch allocation remains 22.289 GiB.
- Exclusive owner scopes sum to 637.722051331 seconds with **zero partition
  error**. The original processing boundary differs by about 13.5 microseconds.

## Additive owner-wall breakdown

These groups sum disjoint exclusive scopes, not nested inclusive timers.
Exact membership is recorded in `accounting.json`.

| Area | Seconds |
| --- | ---: |
| UniRec prefill | 206.608 |
| Paddle prefill | 85.907 |
| Layout/detection/publication | 126.027 |
| UniRec decode, excluding completion/output scopes | 111.726 |
| Paddle decode, excluding completion/output scopes | 62.116 |
| Completion conversion, publication and page/crop output | 20.392 |
| Explicit shared waits | 20.294 |
| Coordinator control and remaining root-scope time | 4.651 |
| **Total** | **637.722** |

This is **host-owner elapsed attribution**, not CPU-active time and not a
device-aware proof that every second is removable. Calls can include NPU
execution/waits. In-engine asynchronous work can overlap host callbacks.
The full-corpus time is close to the prior 635.853-second run; that cross-run
difference alone does not isolate instrumentation overhead.

## What the new measurements clarify

**Layout versus publication.** Detection/H2D/metadata/D2H and the existing
yield fence occupy 124.556 seconds on the owner. The enclosing advance/publish
method contributes only another 1.394 exclusive seconds. CPU input and
post-layout/crop work execute on their own persistent workers, not inside
that detection timer.

**Exposed CPU dependencies.** All 2,450 shared waits total 20.294 seconds.
Wait p50/p99/max: 3.196 / 109.658 / 288.831 ms. The largest observed joint
prerequisite set is `page.crops + unirec.cpu_preparation`: 14.455 seconds.
Only-page-input waits total 0.534 seconds; only-page-crop waits 2.270 seconds.
Other joint sets account for the remainder. A joint set is counted once:
it does not establish how much speeding up one particular worker would save.

CPU service wall totals are 74.724 seconds input, 74.854 seconds page crops,
103.504 seconds UniRec crop preparation, and 44.277 seconds Paddle crop
preparation. These overlapping worker totals must **not** be added to E2E
or interpreted as recoverable critical-path time.

**Output is measurable, but not all file I/O.** The 20.392-second output group
includes completion conversion/publication and assembly as well as writes.
Page assembly+emission inclusive wall is 14.341 seconds, within which page
files cost 9.132 seconds. Markdown/image build+write takes 6.640 seconds;
JSON encoding/writing takes 1.895/0.438 seconds. Crop trace encoding/writing
takes 1.479 seconds. These last numbers are already inside the output total.
Page assembly+emission p50/p99/max: 3.413 / 106.414 / 264.661 ms.

**Yield fences are not Python bookkeeping.** Paddle decode-yield fences total
13.473 seconds over 3,654 turns, p50/p99/max 4.570/4.760/5.869 ms.
UniRec's equivalent fences total 0.225 seconds. These are existing waits,
not new synchronization added by the instrumentation.

**UniRec step diagnostics are properly bounded.** Its 14,933 decode steps
total 91.533 seconds for graph/submission/token-selection/CPU-read elapsed,
p50/p99 6.102/6.349 ms. The token-selection/read wait subinterval totals
86.571 seconds and includes waiting for preceding device execution: it is
not an 86-second pure copy cost. The 30.722-second scheduler subinterval
includes retirement/admission/completion callbacks; it is nested, not an
additional exclusive cost.

**Host and device-event envelopes are different.** UniRec vision/text host
calls total 145.165/54.302 seconds; their existing device-event envelopes
total 155.420/45.093 seconds. Asynchronous work can cross host-call boundaries.
Do not mix those two bases into one additive phase table. Nor do vision-call
p50/p99 values represent individual kernels or fixed buckets: a vision encode
call can process a variable crop set. First-use costs remain in processing.

## Raw evidence and limits

`run_summary.json` retains count/total/mean/p50/p99/max for owner scopes,
CPU service, CPU queue residence, ready residence and UniRec step diagnostics.
`metrics.json` adds the usual throughput/token-density/slot summaries.
`accounting.json` records group membership and raw trace coverage.

The full trace has 413,914 events (138,210,584 bytes), including exactly 33,859
identified CPU jobs, with matching queue/service/ready counts and no negative
durations. Observed workers: one page-input worker, one page-crop worker, one
UniRec preparation worker and one Paddle preparation worker, unchanged.
Raw host intervals share a monotonic clock; existing Paddle device-event
reconstructions are also included. The inherited `Continuous decode scheduler`
span includes cooperative pauses and is context, not exclusive device work.

Full trace, predictions and logs remain under the remote output root in
`command.txt`. The downloaded trace is also at
`/tmp/hybrid-timing-4a01bda5/hybrid1651/timing_trace.json` on the authoring Mac.
No kernel-level critical-path reconstruction or 310P validation is claimed.
