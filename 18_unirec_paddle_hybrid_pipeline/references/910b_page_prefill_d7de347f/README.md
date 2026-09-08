# Shared page-prefill validation on one 910B2

Inference commit `d7de347f`, physical NPU 6, 2026-09-08. See
[command.txt](command.txt) for commands, settings and remote paths. Existing
cache roots, kernels, image preprocessing, decode priorities and capacities
were retained. No 310P validation is claimed.

Both adapters now consume page-local crop groups up to available ready storage,
instead of clipping each prefill to current decode vacancies. Large pages resume
in bounded chunks. Existing model-specific packing remains inside each chunk.
The 128-request UniRec / 64-request Paddle ready capacities are separate from
active decode slots. Paddle retains its preallocated 32 staging slots.

## Full-corpus before/after

| Metric | Previous `9373eba8` | Page prefill `d7de347f` |
|---|---:|---:|
| Pages | 1,651 | 1,651 |
| Processing E2E wall | 928.063 s | 811.249 s |
| Processing pages/s | 1.779 | 2.035 |
| Measured setup | 42.383 s | 43.874 s |
| Pages/s including measured setup | 1.701 | 1.931 |
| UniRec text-prefill useful density | 12.3% | 58.2% |
| UniRec physical text-source tokens | 15,867,720 | 3,354,120 |
| UniRec S1320 text-prefill calls | 12,021 | 2,541 |
| UniRec vision row utilization | 41.3% | 75.7% |
| UniRec prefill wall | 357.530 s | 279.914 s |
| Paddle prefill wall | 141.943 s | 110.873 s |
| UniRec active decode-slot utilization | 88.87% | 88.87% |
| Paddle active decode-slot utilization | 69.48% | 69.65% |
| Peak Torch allocation | 22.200 GiB | 22.289 GiB |

Throughput improved **14.4%**, processing wall decreased **12.6%**. These are
single-run comparisons, not confidence intervals. Timing scopes match: processing
includes layout, recognition, assembly and synchronous output writing; first-use
work within processing is not subtracted. Imports/dataset enumeration precede
the measured setup timer. Peak Torch memory is not total physical-device memory.

The main UniRec device-envelope saving is **vision: 213.138 -> 141.661 s**.
Despite 4.73x better text packing density, the **text-prefill envelope changed
only 44.477 -> 43.164 s**. That envelope includes host submission/extraction and
transfers, not just kernels; the remaining cost has not been isolated. Do not
claim a 4.73x text-prefill latency speedup from padding counts.

Paddle vision device time: 52.371 -> 46.629 s; text-prefill device time:
17.687 -> 13.831 s. Current real-token rates are 49,359 vision patches/s and
43,888 text-input tokens/s. UniRec envelope rates are 13,783 real vision-output
tokens/s and 45,236 real text-source tokens/s; the two models' vision units and
timing scopes differ.

Decode-turn useful rates: UniRec 13,611 tok/s; Paddle 6,362 tok/s. These include
admission/copy/bookkeeping and completion callbacks, excluding other phases.
Narrower UniRec graph+selection+CPU-read time is 92.183 s; Paddle graph+argmax
device time is 57.104 s. Do not compare these as the same timing scope.

## Output validation

All four runs exited 0. The three 64-page controls each have all 64 expected
Markdown/JSON stems; the full run has all 1,651, with no missing or extra stems.
The full trace contains 30,557 unique request IDs and unchanged routing/metadata:
28,125 UniRec text crops, 751 Paddle tables, 1,681 Paddle formulas.

- Full run vs previous full hybrid: **28,125/28,125 UniRec token rows exact**;
  **2,429/2,432 Paddle token rows exact**. All 751 tables are exact. The three
  differences are formulas (one changes an overset symbol, others change LaTeX
  text/spacing). These are not asserted to be accuracy-neutral.
- Hybrid64 vs previous full hybrid: **1,015/1,015 token rows exact**.
- Hybrid64 vs current single-model controls: UniRec **741/741** exact;
  Paddle **272/274** exact.
- New all-UniRec64 vs old all-UniRec64: **1,015/1,015** exact.
- New all-Paddle64 vs old all-Paddle64: **1,009/1,015** exact.
- All stop reasons matched in each comparison. Full run retains 65 UniRec
  length caps, 9 Paddle KV-cap stops and 2 Paddle repetition stops.

No ground-truth accuracy evaluation was run. Token agreement is a regression
anchor, not an OmniDocBench accuracy score.

The first `aadf77a4` smoke caught premature Paddle EOF with three queued crops.
It completed only 63/64 pages despite the then-zero exit code and is excluded
from valid performance evidence. `d7de347f` makes EOF depend on draining the
source queue and adds an explicit page/crop completion assertion. The restarted
controls and full run passed that assertion. Fifteen local tests cover routing,
surplus-ready priority, bounded large pages, EOF and report/trace checks.

## Artifacts

- [metrics.json](metrics.json): derived full-run metrics.
- [run_summary.json](run_summary.json): original full-run counters/settings.
- [prediction_comparison.json](prediction_comparison.json): full token comparison.
- [controls.json](controls.json): three control summaries and comparisons.
- [hybrid1651.log](hybrid1651.log): full run log.

Full predictions/traces remain at the remote root in `command.txt`. Local trace
copies are under `/tmp/hybrid-pageprefill-d7de347f/`. Reproduce analysis with
`summarize_run.py RUN_DIR` and `compare_traces.py REFERENCE_TRACE CANDIDATE_TRACE`.
