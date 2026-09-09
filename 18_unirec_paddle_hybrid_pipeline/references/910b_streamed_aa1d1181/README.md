# Streamed UniRec within the hybrid: one 910B2, first 384 pages

All runs completed on physical NPU 6 using existing caches. The final streamed
path is `aa1d1181`; its model source is byte-identical to `0f2abda9`. No new
vision/text/decode graph definitions, weights, resolution or generation limits
were introduced. Exact commands, roots, counters and timing distributions are
in [comparison.json](comparison.json).

## Results

| First-384 configuration | Processing s | pg/s | Torch allocated peak GiB | Reserved peak GiB | Whole-device sampled peak GiB |
|---|---:|---:|---:|---:|---:|
| Serial control, `0f2abda9` | 180.168 | 2.1313 | 16.3339 | 17.6445 | 21.4531 |
| Streamed, inherited global text fences, `0f2abda9` | 165.782 | 2.3163 | 17.2966 | 19.5488 | 23.4072 |
| Streamed, direct packed export, `aa1d1181` | 162.929 | 2.3569 | 17.2966 | 19.5977 | 23.4561 |

The final path reduced processing wall by 17.240 s (9.57%) and increased
throughput by **10.58%** against the fresh control. The direct export variant
was another 1.75% faster in throughput than the first streamed variant. These
are single runs on a shared machine, not statistically established speedups.

Setup was 42.620 / 44.864 / 43.300 s respectively and is outside processing
wall. Real-page first-use work remains inside processing; these are NOT the
standalone UniRec benchmark's warmup-excluded hot-service rates. No full-1651
streamed run or new ground-truth accuracy evaluation was performed.

Both candidates matched **all 4,346 crop records** against the control:
3,657 UniRec and 689 Paddle, with identical token IDs, text, stop reasons,
page/block/prompt metadata and routing. Every run drained both models' ready
storage; the streamed vision/text buffer returned to zero. Owner timing
partitions closed with zero error. Existing standalone entrypoints and hybrid
default execution remain unchanged.

## What changed versus the ineffective lanes-only test

- The lanes-only experiment supplied one page's crops at a time. Streamed
  execution accumulates CPU-ready crops across pages up to the existing
  128-record capacity. The 384-page candidates used 30 vision windows,
  averaging 121.9 crops / 13.63 page fragments per window, p50 128 crops.
- Vision retains experiment12's `BoundedVisionOwner` and its shape planning,
  stable per-key streams, graph objects and early-output callbacks. Bucket
  execution count fell **1,718 -> 1,269**, while useful vision bucket-row
  utilization rose **65.86% -> 94.73%**. All 3,657 real crops were retained;
  the nine existing eager-overflow crops remained on their existing route.
- Four persistent CPU processes, each with eight resize threads, were actually
  observed in both 384-page candidates. CPU queued/running/completed capacity
  remains 128. The earlier first64 prototype lazily started its pool and only
  one process performed work; explicit setup-time process readiness fixed that.
- Three persistent stage threads allow UniRec vision, text prefill and decode
  to overlap. Text starts when a page fragment's vision outputs are ready,
  without waiting for all vision keys in the window. The same continuous
  decode iterator runs bounded step bursts on its own stream. Ready events
  and `record_stream` protect cross-stream KV/input lifetimes.
- All stage submissions stop at a cross-model turn boundary. In-flight work
  is joined and the device fenced before layout/Paddle runs. Active decode
  slots survive; there are no corpus-wide recognizer phases.

This is intentionally not the standalone service verbatim. PPv3/Paddle retain
crop geometry and assembly; bounded UniRec turns must yield the NPU to Paddle
and layout. Page fragments fit the ready capacity, retaining greedy page-local
text packing. The final first384 text-source packing fraction was 32.60%,
versus the control's 34.37%; vision packing improved, but text packing did not.

## Text-prefill synchronization

The reused cohort convenience method called whole-device synchronization
before and after every text pack. The final streamed export instead invokes
the **same existing `PackedUniRecTextPrefillRuntime.run` graph and segment
redistribution**, stacks compact cross-KV exactly as direct admission expects,
and records a ready event. It does not build unused decode masks/start tokens
or call the globally synchronizing convenience wrapper. Active decode still
uses the existing admission copy and event handling.

An intermediate optional argument in the model file (`aa8ab556`) was NOT
NPU-tested or adopted: its host-only change would unnecessarily invalidate
whole-file cache fingerprints. `aa1d1181` restores the original model file
byte-for-byte and places the integration in experiment18. The remote launch
checked model blob equality before running. User-owned local model edits were
not staged, reverted or included in any inference result.

## What still costs time

Control UniRec prefill+decode owner time was 79.429 s; final streamed UniRec
owner time was **52.682 s**, down 26.747 s. Pipeline savings were smaller:
total coordinator wait grew **1.036 -> 10.739 s**, and other stages varied.
The serial global owner layout time was 34.015 s versus 33.244 s in the final
candidate; Paddle prefill/decode remained about 65 s combined.

The 0f2abda9/aa1d1181 wait-detail snapshot still inspected only the first page,
although streamed readiness depended on a cross-page window. It labelled most
of the extra wait `ready_before_wait`. Therefore the printed
`exposed_cpu_dependency_wait_s` under-attributes these runs; **do not quote it
as their complete CPU critical-path wait**. Total `shared.wait` is measured
correctly. The follow-up metadata-only fix snapshots the actual cross-page
request list; it does not change the readiness rule or execution policy.

The final vision/text/decode worker maximum envelopes were 16.649 / 15.406 /
15.655 s. A long outlier remains despite removing the text-wrapper global
fences. Its cause is NOT established by these aggregate spans. Worker-host
overlap was 21.596 s, but it is not proof of equivalent device-kernel overlap
or time saved; do not sum the worker spans into pipeline wall.

## Memory and correctness limits

Both control and candidates use UniRec ready64 / Paddle ready32, unchanged
active B128/B64, self/cross/decode lengths and Paddle ready-row length1536.
The final sampled device baseline/peak/increase was **3425 / 24019 / 20594 MB**,
against control **3425 / 21968 / 18543 MB**. Total device peak increased by
2051 MB (~2.003 GiB), despite keeping the reduced ready pools. Independent
sampling reported no errors. These are sampled whole-device totals, not
additional memory on top of baseline. Torch/counter scopes overlap.

UniRec ready-row count and the separate 128-record in-flight vision/text buffer
must not be conflated. Their boundedness does not prove 310P memory fit.
The pipeline remains opt-in and **the 310P handoff is not changed to use it**.

Earlier validation: two pages at `26c922c3` matched all 25 crop records;
first64 at `84060541` matched all 1,015 records, at 53.224 s serial versus
49.697 s streamed. Local policy tests: 45 total, six torch-dependent skips.
All 45 passed in the 910B environment before the final run, including compact
cross-KV segment/token-accounting checks. Additional tests cover early producer
publication, stage overlap, one operation per stage, direct error propagation,
full-arena alternating ties, cross-page readiness and safe source closure.

## Reproduce

After `source npu-setup`, set a new output `RUN_ROOT` and `PAGE_LIMIT=384`, then:

```bash
bash 18_unirec_paddle_hybrid_pipeline/run_910b_streamed_unirec_matrix.sh
```

The script runs serial then streamed with the same half-ready settings. It
prints parity differences rather than silently changing settings to seek a
match. Inspect them before interpreting speed. The runtime switch is
`--unirec-streamed --unirec-vision-lanes 4`; CPU defaults are 4 processes x 8
threads. No cache reset or fresh cache root is needed.
