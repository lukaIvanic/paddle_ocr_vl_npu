# Staged CPU layout/crop preparation — one 910B2

Inference commit `de71a457`, 2026-09-08, physical NPU 6. The reference is the
previous persistent recognition-CPU run, `b511f81b`, on the same device. Existing
caches and all model, resize, routing, grouping and decode settings were retained.
Layout graph capture remained disabled. These are individual runs on a shared
machine, not repeated statistical performance estimates.

## What changed

Experiment 18 now calls Paddle's existing staged frontend methods:

- Persistent CPU input worker: `decode_page` + `preprocess_decoded_page`.
- Coordinator: `detect_preprocessed_page`, including H2D, detection, selected
  metadata/mask processing and completed D2H, followed by the NPU yield fence.
- Persistent CPU crop worker: `prepare_detected_page` and the unchanged
  routing-specific Paddle text resize.

Each CPU stage has one reserved slot, including finished-unconsumed results.
Publication stays FIFO. Outstanding frontend futures remain upstream work until
publication; full decode retains priority, and the owner never blocks waiting
for a CPU future when other eligible work exists. No NPU-over-NPU concurrency
or background transfer submission was added. Recognition preparation remains
on its existing bounded workers; page assembly/output writing stays synchronous.
Experiment 09's implementation and standalone behavior were not edited.

## Full 1,651-page result

| Measurement | Recognition-CPU baseline | Staged frontend |
| --- | ---: | ---: |
| Processing wall | 704.214 s | 635.853 s |
| Processing throughput | 2.344 pg/s | 2.597 pg/s |
| Including measured setup | 2.213 pg/s | 2.433 pg/s |
| Layout/publication owner wall | 243.845 s | 126.233 s |
| Shared wait wall | 0 s | 24.012 s |
| UniRec prefill owner wall | 191.940 s | 208.937 s |
| Paddle prefill owner wall | 77.428 s | 83.967 s |
| UniRec decode owner wall | 126.969 s | 126.493 s |
| Paddle decode owner wall | 62.550 s | 63.328 s |
| UniRec useful decode slot utilization | 88.869% | 88.869% |
| Paddle active decode slot utilization | 69.650% | 69.650% |
| Peak Torch allocated | 22.289 GiB | 22.289 GiB |
| Peak Torch reserved | 23.619 GiB | 23.619 GiB |

Throughput increased 10.75%, saving 68.361 seconds. The owner-side frontend
reduction was 117.613 seconds, but shared waits and increased recognition
prefill wall offset part of that saving. `shared.wait` can wait for frontend or
recognition CPU completion; it is not exclusively a layout wait counter.

New frontend worker service wall was 75.397 seconds for input and 85.760 seconds
for post-layout/crops. Coordinator detection service took 125.101 seconds.
All four page counters (submitted/input/detected/crops) equal 1,651; one input
worker and one crop worker were observed, with high-water two frontend pages.
CPU service spans overlap each other and owner work and must not be added to
E2E. The existing `frontend_stage_s` fields also contain nested spans, including
overlapping per-page totals.

The more detailed CPU totals include 58.449 seconds image decode, 12.610 seconds
detector-input preprocessing, 44.768 seconds structural/polygon postprocessing
(including 26.848 seconds polygon work), and 40.374 seconds crop/request
preparation. These are service measurements, not claims of fully hidden time.

UniRec/Paddle decode calls remained exactly 14,933/9,217. Real and padded
recognition token counts and UniRec packing utilization were unchanged.
Exclusive decode-turn useful rates were 13,428.9/6,445.5 tok/s respectively.
UniRec's 91.851-second execution timer includes token selection and blocking
CPU reads; Paddle's 55.999-second timer uses device events for graph + argmax.
Do not compare these as identical device-only measurements.

UniRec's vision event envelope increased from 140.098 to 159.561 seconds, while
Paddle vision-device time stayed approximately 44 seconds. Host contention is
a possibility, not established by this run; the envelope includes submission
gaps and transfers, not just kernels. No CPU-thread tuning was bundled here.

## Correctness and controls

- Full run exit 0; all 1,651 expected Markdown and JSON files present.
- Exact same 30,557 unique crop IDs and metadata.
- All 28,125 UniRec and all 2,432 Paddle token sequences, texts and stopping
  reasons matched `b511f81b` exactly. No new ground-truth evaluation was needed
  for this parity check, and no new accuracy score is claimed.
- All saved non-content page fields matched on every page: geometry, labels,
  polygons, reading order, IDs, page metadata and model settings. The check
  compares entire page JSON after removing only each block's `block_content`.

| First-64 route | Baseline wall | Staged frontend wall | Exact token parity |
| --- | ---: | ---: | ---: |
| Hybrid | 54.478 s | 51.832 s | 1,015/1,015 |
| All-UniRec | 41.349 s | 38.824 s | 1,015/1,015 |
| All-Paddle | 45.955 s | 45.919 s | 1,015/1,015 |

Every control also preserved all non-content page fields. Twenty-three local
tests passed under Python 3.13, covering bounded futures, persistent workers,
coordinator-only detection, worker exceptions and closed-input drain. They are
CPU policy tests; the four runs above provide the real NPU validation.

See command.txt, metrics.json, controls.json, geometry_comparison.json and
prediction_comparison.json. Full traces and page artifacts remain under the
remote output root in command.txt. No 310P validation is claimed.
