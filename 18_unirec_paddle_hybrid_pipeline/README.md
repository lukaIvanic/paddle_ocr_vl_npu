# Experiment 18: continuous UniRec + Paddle page pipeline

310P execution brief: [compact ready-KV retry, smoke64 then full1651](WORK_SERVER_310P_COMPACT_READY_KV_E2E.md).
It reuses the previous attempt's caches, verifies local assets and independently
monitors device memory with the smaller B64/S1536 ready pool; no 310P
validation is claimed until the work agent reports its run.

One PP-DocLayoutV3 frontend feeds two resident recognizers on one NPU.
Experiment 09 owns Paddle layout/cropping/assembly and inference; experiment
12 owns UniRec inference. This experiment owns routing and cross-model turns.

## Opt-in streamed UniRec integration

`--unirec-streamed --unirec-vision-lanes 4` enables cross-page vision supply
and overlapping UniRec vision/text-prefill/decode inside an exclusive UniRec
turn. This is experimental; it does not change the default or the 310P brief.
Use `run_910b_streamed_unirec_matrix.sh` for matched serial/streamed trials,
starting with first64 then first384, existing caches and half-ready capacities.

The CPU preparation pool follows standalone UniRec's process/thread structure:
four persistent spawn processes, eight persistent resize threads each. The
existing crop/RGB/resize contract is retained; PPv3/Paddle still own geometry,
so importing standalone's entire OpenDoc crop builder would be incorrect.
`--unirec-cpu-workers` and `--unirec-cpu-threads` adjust this opt-in pool only.
CPU queued/running/finished crops remain bounded at the active batch capacity
(128 by default). Process startup belongs to setup, not the first timed page.

Vision accumulates CPU-ready crops across pages up to that same 128-record
budget, flushing the available tail when no submitted upstream pages remain.
Experiment12's `BoundedVisionOwner` performs the unchanged shape planning and
multi-key stream dispatch. Completed key groups publish early through its
existing callback; text prefill can begin for complete page fragments before
the whole vision window finishes. Each fragment is at most the selected ready
capacity; the existing page-local greedy text packs and compact cross-KV export
are used. Ready-KV credits include in-flight text exports. Decoding consumes
the existing ready-event dependency and cooperative continuous iterator, not
a new model-forward or cache-copy implementation.

Three persistent stage threads overlap vision, text and decode. Only the owner
mutates page assembly and stage scheduling; workers never block on downstream
queue capacity. At the decode-step budget, stop NEW submissions, join in-flight
operations, then fence the device before Paddle/layout can run. Decode arenas
survive turns. A turn can also return when it needs more upstream layout/CPU
work; no batching timer or corpus-wide recognizer phase is added.

This is a bounded shared-owner adaptation, not the standalone service verbatim:
standalone can run its stages indefinitely, whereas the hybrid must hand the
NPU back to Paddle/layout. Model kernels, shape presets, generation limits and
existing cache roots remain unchanged. Standalone 09/12 entrypoints are unchanged.

`unirec.stream` is the non-overlapping owner interval. Worker stage envelopes
overlap and are reported in `streamed_execution` and `overlapping_worker_scopes`;
never add them to owner wall or call them kernel occupancy. Streamed vision's
host-envelope token rate is labelled separately from device-event rates.
Window sizes, page-fragment counts, buffer high-water, CPU process/thread
counts, ready-KV peaks, external memory and crop parity must accompany timing.
The two-page 910B smoke at `26c922c3` passed all 25 crop records against the
existing control; larger-workload throughput validation is pending.

Experimental `--unirec-vision-lanes 1|2|4` reuses experiment 12's persistent
`BoundedVisionOwner` with in-memory prepared inputs, all graphs retained and
no same-key graph cloning. Default `0` retains the validated sequential path.
Only vision calls within a UniRec prefill turn can overlap; all lanes finish
before text prefill or any layout/Paddle/decode turn. CPU worker counts, crop
grouping, routing and ready-KV capacities are unchanged. The lane executor and
streams persist until shutdown; no extra model-weight copies or spool writes
are introduced. Existing cache roots and compiled model code are unchanged.

`vision_runtime.hybrid_lanes` records joined turn wall time, concurrent group
widths and per-key lane host spans. Lane spans overlap and are not additive to
the critical path, nor are they pure device-kernel timings. The existing owner
scope around `vision.encode` measures the joined interval. More lanes can need
more activation/workspace memory; 310P fit is not inferred from 910B.

`run_910b_vision_lanes_matrix.sh` runs sequential control/1/2/4-lane comparisons
on identical first-N pages (`PAGE_LIMIT=64` initially), with independent device
memory sampling and exact per-crop token/text/stop parity gates. Run only after
`source npu-setup`, with `RUN_ROOT` naming a new output directory. It never
clears or changes the graph cache roots. All 64/384-page cases passed on 910B;
four lanes improved the 384-page rate by only 1.9% while increasing sampled
whole-device peak memory by 1.71 GiB. Default remains `0`; this is not a 310P
recommendation. See [measured comparison](references/910b_vision_lanes_abe23c9b/README.md).

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

## Paddle ready-KV storage

Ready request counts can be varied independently of active decode and CPU
preparation using `--unirec-ready-capacity` and `--paddle-ready-capacity`.
Defaults remain 128/64 at the usual batch sizes. A half-ready experiment uses
`--unirec-ready-capacity 64 --paddle-ready-capacity 32`; Paddle private rows
default to its selected ready capacity, retaining length1536. CPU preparation
remains 128/64 and active decode remains B128/B64. A full smaller reservoir
gets an admission turn through the existing engine copy path, yielding before
decode so more prefill can fill the remaining active slots. Standalone engine
behavior stays unchanged unless cooperative refill is explicitly selected.
UniRec `compact_ready_kv` reports logical live/high-water cross-KV rows/bytes,
excluding temporary prefill tensors and allocator retention.
[First-384 910B validation](references/910b_half_ready_3a45bf83/README.md)
passed: all 4,346 crop records matched the fresh original-capacity control;
sampled whole-device peak fell 22.323 -> 21.454 GiB, with 185.50 -> 184.08 s
pipeline wall time. CPU capacities and active decode arenas were unchanged.
Defaults remain unchanged; this is not yet a 310P memory-fit validation.

[Full 910B validation](references/910b_readykv_8b60af95/README.md): all 30,557
crop outputs and all 1,651 page JSON/Markdown files matched exactly. Peak
allocation fell 22.29 -> 17.30 GiB, at 2.61 pg/s versus 2.59 pg/s previously.

The hybrid requests one ready row per Paddle decode slot (64 by default), with
1,536 positions per ready row. The active decode arena remains B64 x 4,096;
generation limits, preprocessing and grouping are unchanged. Override only for
an explicit comparison with `--paddle-ready-cache-rows 96
--paddle-ready-cache-length 4096`. The pool now reports allocation, live/high-water
leases, releases, row length and maximum observed prompt length in `ready_kv_pool`.

Admission still runs through experiment 09's `DecodeArena.admit`, using its
existing `torch._foreach_copy_`, controls, and release-event protocol. A compact
source row is copied into the corresponding destination prefix; the remaining
finite suffix stays masked by the real cache position and is overwritten as
decode progresses. The active arena is zero-initialized as before. Source rows
are reusable only after the existing release-event dependency.

Packed prefill keeps its existing scratch graphs and prefix redistribution.
Individual prefill uses one reusable B1 x 4,096 scratch cache to retain the
compiled call shape, then copies the valid prefix to the leased ready row.
Prompts exceeding the ready-row length raise explicitly; they are never truncated
or silently rerouted. The previous run's largest Paddle prompt was 1,036 tokens.
Experiment 09 standalone constructor defaults retain their original pool sizes.

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

Validated on all 1,651 pages on 910B2: see the
[timing validation and accounting report](references/910b_timing_4a01bda5/README.md).
All 30,557 crop predictions and all page JSON/Markdown files remained identical.

Detailed timing defaults on; `--no-detailed-timing` is the instrumentation-cost
control, not a scheduling preset. `detailed_timing.owner_exclusive` is a checked
non-overlapping partition of the owner `pipeline` scope. Each label reports
count, total, mean, p50, p99 and max seconds. `owner_inclusive` contains nested
durations and must not be summed. Remaining scope time is retained explicitly,
not inferred to be idle CPU or accelerator work. The original `wall_s` boundary
remains unchanged; the encompassing timing scope differs by a small call-boundary
overhead. Detailed trace export occurs after processing and is timed separately.

`wait_blocker_sets` counts each observed wait once, including joint prerequisites.
It snapshots missing CPU crop requests and page input/post-layout futures at wait
entry. A ready-before-wait race is labelled separately. This is exposed waiting
under the current policy, including wake/scheduling latency, **not** proof that
speeding up one named worker would save the entire joint wait. `cpu_service`,
`cpu_queue_residence` and `cpu_ready_residence` are overlapping per-job elapsed
distributions. They are not additive to owner time. Raw job intervals and page/
crop identifiers are saved in `timing_trace.json` using Paddle's existing shared
host-clock recorder; its viewer can open that JSON. Existing Paddle frontend/
prefill/decode trace spans are included, but device envelopes are not exact
kernel occupancy or a globally synchronized device critical-path proof.
In particular, the inherited `Pipeline / Continuous decode scheduler` span
is Paddle's iterator lifetime, including cooperative pauses for other stages.
It is context, not exclusive Paddle execution; use the `Hybrid owner` scopes
and their checked exclusive totals for coordinator accounting.

Output scopes separate completion conversion, page assembly/emission, Markdown
and image building/writing, JSON encoding/writing and crop tracing. Decoder
advancement and yield fences are measured separately. UniRec's existing step
callback supplies input-build, submission, token-read/wait and scheduler
distributions; those are nested host intervals, not extra exclusive costs.
No new NPU synchronization is added. Vision/text/decode models and cache keys
are not changed by instrumentation.

Legacy engine timers that span generator suspension are moved out of their
normal timing dictionaries into `cooperative_pause_inclusive_legacy_timing`.
Do not interpret those numbers as exclusive decode or scheduler overhead.
The original engine objects and standalone reporting behavior are unchanged.

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
