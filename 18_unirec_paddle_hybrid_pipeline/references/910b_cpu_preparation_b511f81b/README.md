# Persistent recognition CPU preparation — one 910B2

Inference commit `b511f81b`, 2026-09-08, physical NPU 6. Existing compiled
caches, model presets, page-local prefill selection and decode capacities were
retained. Only the coordinator submits NPU work; one persistent CPU worker per
recognizer prepares bounded host inputs. Layout/preparation and page output
writing are still synchronous on the coordinator.

All three first-64 controls and the full 1,651-page hybrid completed. Nineteen
local policy/worker tests passed; those tests are not inference validation.

## Full-corpus result

The reference is `d7de347f`'s full shared page-prefill run, on the same physical
910B2. These are individual measured runs, not repeated confidence intervals.

| Measurement | Reference | Persistent CPU preparation |
| --- | ---: | ---: |
| Processing wall | 811.249 s | 704.214 s |
| Processing throughput | 2.035 pg/s | 2.344 pg/s |
| Including measured setup | 1.931 pg/s | 2.213 pg/s |
| Layout + crop preparation | 231.227 s | 243.845 s |
| UniRec prefill turn wall | 279.914 s | 191.940 s |
| Paddle prefill turn wall | 110.873 s | 77.428 s |
| UniRec decode turn wall | 124.803 s | 126.969 s |
| Paddle decode turn wall | 64.161 s | 62.550 s |
| UniRec useful decode slot utilization | 88.869% | 88.869% |
| Paddle active decode slot utilization | 69.653% | 69.650% |
| Peak Torch allocated | 22.289 GiB | 22.289 GiB |

Throughput improved 15.2%, with 107.035 seconds less processing wall. Prefill
turns saved 121.419 seconds combined; layout increased 12.618 seconds. Do not
attribute every delta to CPU overlap alone: per-chunk producer setup was also
removed, Paddle FIFO pack membership now uses already-prepared inputs, and
run-to-run effects are not isolated.

UniRec executed exactly the same 14,933 decode graph calls, source token counts,
58.214% text-prefill density and 75.740% vision-row utilization as the reference.
Paddle also retained 9,217 decode calls; real vision tokens were unchanged,
while padded vision tokens decreased from 2,598,272 to 2,585,856. Thus the gain
is not explained by a changed decode batch or reduced recognition resolution.

CPU worker service wall was 105.102 seconds for UniRec and 46.110 seconds for
Paddle. These spans overlap coordinator work and must not be added to E2E or
interpreted as CPU-active time. The full run never selected `shared.wait`.
One worker thread was observed per model; submitted = consumed = 28,125 and
2,432 respectively. Storage high-water counts were 128/128 and 64/64. This
bound covers submitted plus completed-unconsumed CPU preparation, not all raw
page/crop state. No new memory-budget policy was introduced.

Useful decode throughput per exclusive decode turn was 13,378.6 tok/s for
UniRec and 6,525.7 tok/s for Paddle. Engine execution timings have different
bases: UniRec graph + selection + blocking read took 92.144 seconds; Paddle's
device-event graph + argmax took 54.958 seconds. They are not two equivalent
device-only measurements. Detailed rates and distributions are in metrics.json.

## Output checks

- Exit code 0; all 1,651 Markdown and JSON output stems matched the dataset.
- Exact same 30,557 unique crop IDs and metadata as the reference.
- All 28,125 UniRec token sequences, texts and stopping reasons matched.
- Paddle: 2,424/2,432 token sequences matched (750/751 tables and
  1,674/1,681 formulas). All stopping reasons matched.
- Some differences are formatting, but not all: one formula changes a
  superscript symbol, and one table changes a mathematical expression in a
  header. No claim of unchanged ground-truth accuracy is made; no new accuracy
  evaluation was run.
- Length/repetition/KV-cap stop counts were unchanged.

## First-64 controls

| Route | Reference wall | New wall | Exact token sequences |
| --- | ---: | ---: | ---: |
| Hybrid | 60.781 s | 54.478 s | UniRec 741/741; Paddle 272/274 |
| All-UniRec | 44.777 s | 41.349 s | 1,015/1,015 |
| All-Paddle | 47.816 s | 45.955 s | 1,011/1,015 |

Each control completed all 64 pages / 1,015 crops. All stopping reasons match
its corresponding reference. `controls.json` records the metrics and mismatches.

## Remaining scope

Layout + crop preparation is now the largest exclusive stage (243.845 seconds,
34.6% of processing wall). Splitting its CPU work from NPU layout, and moving
page output writing off the coordinator, remain later steps. This result does
not add NPU-over-NPU concurrency or claim 310P validation.

See command.txt for exact invocations. Full traces and page artifacts remain
under the documented remote output root; compact summaries and the full-run log
are committed here.
