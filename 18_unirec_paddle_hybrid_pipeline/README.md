# Experiment 18: continuous UniRec + Paddle page pipeline

One PP-DocLayoutV3 frontend feeds two resident recognizers on one NPU.
Experiment 09 owns Paddle layout/cropping/assembly and inference; experiment
12 owns UniRec inference. This experiment owns routing and cross-model turns.

```sh
python 18_unirec_paddle_hybrid_pipeline/run_pipeline.py \
  --input /workspace/datasets/OmniDocBench/images \
  --dataset-json /workspace/datasets/OmniDocBench/OmniDocBench.json \
  --text-model unirec --table-model paddle --formula-model paddle \
  --unirec-model-path /path/to/unirec \
  --openocr-root /path/to/OpenOCR \
  --unirec-vision-cache /existing/vision/cache \
  --unirec-decode-cache /existing/decode/cache/parent \
  --limit 2 --output-dir /new/run/output
```

Set all three model arguments to one recognizer to load only that recognizer.
No YAML, label-specific overrides, or fallback recognizer exists. Unknown
prompts fail rather than being silently assigned to text. Chart/image/seal
handling remains the owned Paddle frontend's existing skip policy.

## Scheduling

Full arenas (including ready requests reserving slots) take priority. Ties
alternate UniRec/Paddle. A turn runs up to `--decode-steps` graph calls (32
initially), stopping earlier when slots need refill. When neither arena is
full, execute existing prefill groups for pending crops, otherwise advance
layout. Partial arenas drain only when submitted upstream work for that model
has exhausted. `PageInbox` supports independent arrivals and explicit close;
the benchmark submits filenames then closes input, not the recognition arenas.

No page decode cohorts and no corpus-wide text/table phases. Prefill takes a
page's assigned crops up to the available ready-storage capacity, independent
of decode vacancies. Larger pages resume on a later refill. Both adapters use
this shared rule, then apply their existing vision/text grouping. Ready capacity
is UniRec B128 / Paddle B64 at the default batches; Paddle retains its existing
additional 32 private-KV staging slots. Active decode slots are separate from
ready storage. Full decode remains eligible when active + ready >= batch size.
CPU-ready pending crops are prefilled before more pages are examined; while
preparation is outstanding, layout may advance instead. There is one compute owner; NPU work
is synchronized at cross-model yield boundaries. The engines retain pending
token-copy objects, slot epochs, EOS handling and KV update ordering.

The existing blocking decode entrypoints consume the new iterator boundaries
internally. Standalone callers do not opt into cross-model synchronization.
Model definitions and their compiled cache keys are unchanged by those seams.

## Persistent CPU preparation

Each selected recognizer owns one run-lifetime CPU worker. Submitted and finished
CPU requests share a bounded capacity (UniRec 128 / Paddle's existing CPU-prep
capacity, 64 at B64). The coordinator pumps these queues independently of NPU
ready-KV capacity. UniRec reuses its resize/RGB path; Paddle reuses `_prepare_cpu`
and its FIFO ready-input grouping, without spawning per-chunk producers.
FIFO/page boundaries remain intact. Since Paddle now groups fully CPU-ready
inputs, pack membership is no longer affected by its old impatient producer's
instantaneous availability; numerical comparisons are required.

Only the coordinator stages transfers and submits layout, prefill and decode.
If one model's CPU inputs are pending it can run other eligible work. When no
work is eligible it waits on an event signalled by CPU completion or page input;
in-flight CPU requests continue to count as upstream work, including after input
closure. Worker errors propagate, and workers are joined during shutdown.
`cpu_preparation` summaries record consumed requests, high-water storage and
worker service wall time (overlapping work, not additive to E2E). Coordinator
`shared.wait` measures exposed waiting.

## Staged page preparation

One persistent input worker reads/decodes the next page and builds Paddle's
unchanged CPU detector input. Detection, H2D, selected-mask processing and D2H
remain on the coordinator, with an NPU fence before yielding. One persistent
post-layout worker consumes CPU-owned results using Paddle's existing polygon,
crop/merge and request-building methods, including routing-specific text resize.
The frontend retains its existing internal CPU crop/mask workers.

There is one slot for each CPU stage (at most two frontend pages in flight,
including finished-but-unconsumed futures). Publication remains FIFO; raw page
input and recognition-ready storage retain their existing behavior. Pending
frontend futures count as upstream work until the page is published, so neither
temporary CPU gaps nor a closed input can cause premature partial drain. The
coordinator waits only when no eligible NPU/frontend action exists. Full decode
and ready recognition prefill keep priority over layout. No separate NPU layout
stream, transfer worker, new capture mode or page decode cohort is introduced.

`page_preparation` records stage counts, worker service wall, detector owner
wall and the existing frontend timing fields. CPU spans and page totals overlap;
they are not additive to coordinator wall. Page output writing remains synchronous.

## First implementation scope

Paddle keeps its production greedy vision / packed text prefill, B64/KV4096,
PSE-sentinel decode, and 0.5 text-crop scaling. UniRec uses the K20 production
vision kernels, its existing packed text prefill, NZ decode and B128/C1320/S2048.
UniRec vision runs on the shared owner, not four concurrent vision lanes.
Both consume Paddle's region geometry/merge policy. UniRec receives unscaled
crop pixels and uses its own resize/normalization and output conversion.

Setup is separate from pipeline wall. First-use compilation remains visible
inside pipeline wall; no hot-throughput claim should be made from a cold smoke.
Per-crop traces retain the model, task prompt, token IDs and content. Do not
compare token IDs across tokenizers. The initial writer is synchronous.

CPU policy tests are not inference validation. 910B and 310P validation must
be recorded separately; no 310P result is claimed by this implementation.

The [initial 910B validation](references/910b_first64_c507cb15/README.md)
completed 64 pages in all-UniRec, all-Paddle and hybrid configurations. All
741 hybrid text token sequences matched the all-UniRec control; 272/274
specialist sequences matched all-Paddle. Full-corpus accuracy and optimized
throughput are not yet established. The first-64 set is formula-heavy.

```sh
python3.12 -m unittest discover -s 18_unirec_paddle_hybrid_pipeline/tests -v
```
## Timing interpretation

`action_wall_s` measures serialized coordinator service time by model/phase.
Engine-wide elapsed timers span cooperative pauses and must not be interpreted
as exclusive decode time. Slot utilization uses each engine's active/effective
token slots divided by its physical token slots. Paddle's
`decode_model_and_argmax_device` is a device-event measurement; UniRec's
`decode_s` includes graph execution, token selection and the blocking CPU read.

`prefill_tokens` retains real and padded token counts. Paddle's
`prefill_device_s` sums existing stage events (shared packs counted once).
UniRec's new `*_envelope` event intervals surround the production vision and
text-prefill calls: they include transfers and host-submission gaps, **not just
kernel-active time**. They resolve at the existing prefill synchronization;
no additional synchronization or model/compile-cache change is introduced.
UniRec source tokens and Paddle vision patches have different definitions and
must not be compared as identical units. First-use cache/compile costs inside
the processing window remain included in E2E time.

The [full 1,651-page 910B run](references/910b_full1651_9373eba8/README.md)
completed at 1.779 pages/s (1.701 including measured setup). It records
per-engine slot utilization, token rates, prefill density and device timing.
Accuracy evaluation is still separate and has not been run for this hybrid.

The subsequent [shared page-prefill run](references/910b_page_prefill_d7de347f/README.md)
completed all 1,651 pages at 2.035 pages/s, improving UniRec text-prefill density
to 58.2% and vision row utilization to 75.7%. Full token comparison preserved
all UniRec text and Paddle tables; three Paddle formulas differed. Both model
adapters use the page-prefill rule above with their existing ready capacities.

The [persistent CPU-preparation run](references/910b_cpu_preparation_b511f81b/README.md)
completed all 1,651 pages at 2.344 pages/s (+15.2% versus page-prefill), with
serialized NPU execution. Both workers stayed within their existing capacities;
all UniRec predictions matched, while seven Paddle formulas and one table
differed. Layout/crop preparation and output writing remain synchronous.

The next [staged frontend run](references/910b_layout_cpu_de71a457/README.md)
completed all 1,651 pages at 2.597 pages/s (+10.75%), with exact token parity
for every crop and identical saved layout geometry. CPU input and crop work now
run on the persistent stages described above; NPU work remains serialized and
output writing remains synchronous. Shared waits and increased recognition
prefill wall offset part of the owner-side layout reduction; the report records
both rather than treating all background CPU service time as saved E2E time.
