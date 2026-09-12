# Experiment 19: table OCR serving

## Reading the HTTP service

`serve.py` starts with the manually reviewed `ServeConfig`, followed by `main()`.
Argument parsing is at the bottom, immediately before the script entrypoint.
The configuration declaration and its comments are unchanged by the structural passes.

- `main()` starts inference, runs HTTP serving, and coordinates shutdown.
- Process A: `HttpServer` and `HttpRequestHandler` receive
  images and send responses. `InferenceConnection` also lives in A: it owns the
  queues and process handle for B, matches results to waiting requests, and
  handles worker startup/shutdown messages. It is not another process.
- Process B: the small `run_inference_process` entrypoint constructs an
  `InferenceWorker` inside B. Its `run`, `pull`, `closed`, `emit_result` and
  `emit_error` methods replace the nested request-source class and callbacks.
  The existing recognizer still owns model execution and OCR scheduling.

`HttpServer.run()` owns signal registration and the serving loop; its named
shutdown callback replaces the nested function. `HttpServer.close()` stops
inference before closing the HTTP server, including when saving the summary
raises. `InferenceConnection.stop_inference_process()` now also writes the
shutdown summary, keeping file serialization out of `main()`.

The old `_State`/`submit`/`_dispatch` names are replaced with
`InferenceConnection`/`recognize`/`_receive_results_and_status`. Each waiting HTTP
request still has its own one-result queue; the shared dictionary lock is not
held during inference waits. An HTTP timeout still does not cancel work in B.
The nonfunctional HTTP `/v1/drain` endpoint is removed (now 404). Graceful
shutdown is retained as `stop_inference_process()`: send the end-of-input marker,
wait for the final summary, and join or terminate the worker using the existing
timeouts. The summary is written after the worker has stopped.

Validation includes CPU tests for out-of-order replies, startup failure, request
timeouts, queue rejection, graceful/forced shutdown, HTTP responses and summary
output. A fake OCR worker also exercises actual spawned-process communication.
These are service-plumbing checks, not NPU inference or performance validation.

## Explicit server paths

`ServeConfig`, above `main()` in `serve.py`, lists the public options and their
defaults. The CLI requires `--model-path`, `--graph-cache-directory` and
`--log-folder`. Paths are resolved from the launch working directory; none
are derived from the repository or given machine-specific defaults.

All compiled graphs live under the single supplied cache directory, in
`decode/`, `vision_prefill/` and `text_prefill/`. Stage-specific cache keys and
head separation are unchanged. Existing caches are not moved automatically:
an empty new location will require compilation when the server next runs.

`--run-eagerly` runs the same NPU stages without TorchAir compilation.
`--metrics-level` is `basic`, `scheduling` (default), or `detailed`: ordinary
request metrics, additional scheduling instrumentation, or both plus NPU decode
event timings. The old independent timing flags and three cache-directory flags
are replaced, not retained as aliases. Historical launch commands require their
recorded source revision.

The required `--log-folder` receives `service_summary.json` on shutdown, when
the worker returns a summary. It is not a live log or a
crash-safe record; continuous logging remains a separate planned discussion.

The server no longer calculates `HERE`/`EXPERIMENT_ROOT`/`REPO_ROOT` or inserts a
directory into `sys.path`. Launch it normally with Python: Python makes the
script directory importable, and the spawned worker inherits that import path.

## Output formatting

Post-generation repetition truncation and math-delimiter rewriting are removed.
The table-only HTTP endpoint calls `convert_otsl_to_html` directly, keeping the
original text if conversion returns nothing. The obsolete normalization wrapper
and label branch are removed. HTTP responses still retain `raw_text` and native
token IDs. No decoder stopping rule changed.

The HTML conversion behavior is unchanged: it resolves merged cells, pads
short rows, escapes HTML-sensitive cell text, and trims cell-edge whitespace.
There is no general whitespace/newline collapse. Cell-edge trimming changed no
outputs in the saved 665-table check. Removing math rewriting changes 132 of
those formatted outputs; repetition removal had affected none. This is an
intentional output-formatting change, not a new inference or quality benchmark.
The converter now reads as three steps: parse/pad rows, resolve cell origins and
spans, then emit HTML. Descriptive names replace `anchors`/`owner`/`info`, and
cell creation is consolidated. The parser remains source-identical. All 1,000
saved conversions (665 unique tables) match the previous converter exactly.
All 33 CPU tests pass, including 1,000 deterministic mixed/malformed OTSL cases and exact
preservation of non-table whitespace, math, currency and repetitive content.

## Fixed checkpoint implementation

The product no longer reads model or preprocessor configuration files, constructs
model-config objects, or forwards architecture settings through constructors.
The verified PaddleOCR-VL-1.6 dimensions, RoPE values and token IDs are explicit
constants in the modeling files. Crop processing fixes 14x14 patches, 2x2 merging,
one temporal frame, and the effective serving limits of 28,224–802,816 pixels.
RGB conversion remains conditional on the actual image mode; resize dimensions
remain input-dependent. Kornia resizing and FP32 normalization before FP16 vision
retain their existing operation order.

Model input files are data only: the local `model.safetensors`, `tokenizer.json`
(including vocabulary/merges), and the bundled native-token mapping for the
60,416-row head. Full 103,424-row decoding remains supported. No `config.json`,
`preprocessor_config.json`, tokenizer-config or generation-config file is needed.
Operational arguments such as device, model directory, batch size and endpoint
remain configurable. TorchAir's compiler API configuration is unchanged.

At the user's request, checkpoint-config hashes were removed from all three
stage cache keys and metadata, without a replacement model fingerprint. The
now-unused model-directory forwarding was removed too. Existing source/version
hashes and head-specific cache separation remain. No caches were renamed,
deleted or bypassed; broader cache policy is deferred.

This change passes 33 local CPU tests: full-size model parameter-shape parity on
the meta device (no weight allocation), rotary-buffer parity, a mocked weight
loader with no config files, image-grid checks across 70 size pairs, RGB/RGBA/L
patch-layout checks with an identical resize test double, and the existing
arithmetic, call-order and scheduler controls. Historical config classes are
loaded from Git only by reference tests, never by the product. This revision has
**not** been compiled or benchmarked on an NPU.

## Current preprocessing implementation

Kornia-RS bicubic resizing and uint8 CPU patch preparation are the sole path.
The NPU normalizes the transferred uint8 patches immediately before vision.
This selects the existing combined path tested in eight 100-table comparisons:
[preprocessing comparison](../tmp/19_table_ocr_serving/preprocess_options_20260911/README.md).
Those are closed-loop HTTP results, not a new Poisson-load validation. The same
endpoint and preprocessing run under either arrival schedule. No scheduler,
pixel limits, vision-token budget or model graph is changed by this default edit.

The resize/normalization selectors, Pillow-resize branch, CPU normalization/LUT
helpers and their runtime forwarding/state have been removed. Transfers always
carry uint8 patches; the existing FP32 rescale/subtract/divide followed by FP16
conversion runs unconditionally before vision. Operator order is unchanged.
`preprocess_pil_image` returns uint8 patches. The unused file-path wrapper
`preprocess_image` has been removed; serving decodes uploaded bytes on its CPU worker.
Historical A/B launchers require their recorded source revision; their removed
constructor arguments are no longer accepted by the product implementation.

This cleanup is CPU-tested against the selected historical source and prefill
call trace; it is not another NPU benchmark. The subsequent fixed-checkpoint
cleanup above removes the RGB/resize/temporal configuration switches. The
deployed preprocessor was read directly and specifies `do_convert_rgb=true`,
`do_resize=true`, and `temporal_patch_size=1`. Target resize dimensions and actual
image modes remain input-dependent.

**Simplified source `0976fa33` compiled and tested on one 910B2 on 2026-09-11.**
The cached-process B8 / 6-QPS / 1,000-request replay has exact output and workload
parity with step 2. P95 is 3.790 s versus 3.780 s; mean is 1.318 s versus
1.284 s (+2.7%). The mean increase is retained, not declared measurement noise.
See the step-3 validation section below.

**New cleanup after that validation (not yet NPU-tested):** text-weight NZ
conversion now fails setup on non-NPU weights, cast exceptions, or a returned
non-NZ format. Native/mixed-format continuation is removed. Decode KV writes
use only the direct `scatter_update_` calls; the unused CPU-indexing branch is
removed. IncreFA/scatter label constants, label helpers, readiness fields and
cache-name tags are removed, not replaced with other selectors. `FRACTAL_NZ = 29`
remains the named storage-format code. Physical weight-format statistics remain.
Source/cache-name changes may require compilation on a future NPU run; no cache
is renamed or bypassed, and no NPU execution is part of this edit.

The subsequent checkpoint-specific text sweep is also CPU-tested only:

- The deployed checkpoint's `config.json` was read directly: text `use_bias=false`,
  `hidden_act=silu`, hidden size 1024. Q/K/V and gate/up inputs share that width.
  Packed projection construction no longer validates arbitrary mixed widths or
  supports optional biases; text MLP calls SiLU directly. Vision GELU is unchanged.
- `torch_npu` is imported once at module scope. CPU tests supply their own fake
  module explicitly; production has no CPU fallback. Redundant internal NPU and
  tensor-shape guards are removed from the text computation/preparation path.
- Decode always constructs its future-slot mask. There are no optional supplied
  masks/KV positions or scalar-position broadcast inputs; the scheduler supplies
  one position per slot. Prefill always receives its padding mask.
- Layer 0 explicitly uses RMSNorm, later layers AddRMSNorm. Prefetch remains
  before normalization and all retained operator calls keep their order.
- Text prefill has one fixed bucket tuple `(128, 256, 512, 1024, 1152)` and one
  padding policy. Bucket parsing, `auto`/`none` options and selectable text/decode
  backends are removed. Smallest-fitting-bucket selection remains request-dependent;
  overflow still uses the same unpadded prefill stage, not a new implementation.
- Decode source hashing now hashes this actual text file, not deleted experimental
  modules. Fixed dtype/weight-format forwarding and old preset cache-name tags
  are removed. Source hashes still change naturally; cache reuse is not forced.
- Model loading requires a local path. Hugging Face download resolution and the
  default hub ID are removed; checkpoint loading/filtering itself is unchanged.

There are 24 passing CPU tests. In addition to arithmetic and call-trace parity,
they compare every prefill route from 1 through 4096 tokens, padded tensors at
bucket boundaries and overflow, and constructor compilation/warmup call order
using a fake compiler. These changes are not covered by the historical NPU
result below and will need a later agreed compilation/validation pass.

### Latest vision and single-crop cleanup (CPU-tested only)

Vision directly uses the checkpoint's GELU/tanh, MLP zero-extension to 4352,
joint D72-to-D80 attention weights, linear patch projection and mandatory NZ
conversion. Backend/padding/format selectors and hypothetical checkpoint guards
are removed. Dynamic request lengths still select buckets or the existing
aligned overflow path. The projector's own GELU is unchanged.

`--eager` is the one compilation opt-out: it executes the same NPU stages,
weights, padding and scheduler without TorchAir wrapping. It applies to vision,
text prefill and decode; default execution is compiled. The test suite checks
that eager constructors do not invoke the compiler. No NPU eager run is claimed.

Prefill now carries `_PreparedCrop`, `_StagedCrop` and `_InFlightCrop`. Singleton
member lists, row layouts, profiled group routes, intermediate text-pack objects
and result lists are gone. CPU lookahead, free-slot admission, H2D stream events,
compute dependencies, first-token D2H synchronization and cache-lease ownership
are preserved. The legacy single-token concat is deliberately retained as an
operation, not optimized away during this structural pass.

The duplicate packed-projection setup call and its `hasattr` guards are removed:
projections are built once before NZ conversion. The 32-slot staging reserve
keeps its allocation behavior under the name `private_cache_staging_headroom`.
Unused `runtime_defaults.py` is deleted. Per-crop counters replace packing
statistics machinery, preserving the existing external summary schema with
constant compatibility fields until the separate metrics discussion. Internal
timeline stage keys and grouping labels now describe a crop, not a pack.

Output normalization, repetition handling and OTSL conversion were moved
verbatim into `crop_processing.py`; unused page-output/postprocess modules were
deleted. They remain recoverable from Git history. The formatting algorithm
and its text/formula label handling are unchanged.

New tests compare the original singleton prefill chain with the crop chain:
stage outputs, KV contents, first tokens, event/copy ordering and device-stage
accounting, for regular/uint8 preprocessing and timeline on/off. Formatting is
checked for exact source equality and output equality. Vision arithmetic and
weight preparation still match the reference on CPU; request routing is checked
through length 8192. These tests do not substitute for the deferred NPU run.

### Text-module reading order (CPU-tested only)

`text_prefill_and_decode.py` now introduces the shared model and KV storage,
then prefill computation, one-token decode computation, one-time decode
preparation, and finally runtime/bucket/compilation details. Stage `forward`
methods precede their implementation details; the prefill runtime's request
methods precede its long setup constructor. No new execution abstraction was
introduced.

Against the local pre-reordering snapshot, all 60 function/method definitions
retain byte-identical source, and the complete module AST matches after
normalizing declaration order. All 24 CPU tests pass. The historical prefill
source check now locates definitions by name rather than their position in a
file suffix; it permits only the stage method-order change, not body changes.
This source relocation changes the source hash and is not NPU validation or a
reason to bypass the compilation-cache identity checks.

### Reading order across the remaining main modules (CPU-tested only)

The same declaration-only pass now covers all five other main scripts:

- `vision_prefill.py`: shared model, embeddings, encoder computation, projector,
  weight preparation, and execution/bucket setup.
- `crop_processing.py`: crop preprocessing, prompt construction, output
  normalization and its helpers, then preprocessor configuration.
- `paddle_ocr_vl_1_6_modeling.py`: model composition, checkpoint loading, stage
  assembly, then cache and position-construction details.
- `serving_runtime.py`: request entrypoints, decode coordination, CPU preparation
  and prefill, completion, then persistent setup and diagnostics. The open-request
  admission helper and crop-state records follow the main recognizer.
- `serve.py`: startup and CLI, HTTP handling, request/result coordination,
  the NPU worker, then setup-GC and summary helpers. The `__main__` invocation
  remains after every definition.

All 105 function/method definitions across these files retain byte-identical
source against the local pre-reordering snapshots, including decorators and
signatures. The complete module ASTs match after declaration-order normalization.
The 24 CPU tests and HTTP CLI smoke pass. Physical-layout-dependent tests now
compare named definitions/methods; computation and behavior checks remain.
No new files or abstractions were added to the product, and no NPU run or
compilation was performed for this pass.

This self-contained runtime is copied from experiment 09 at
`be691de190ae099d1a9b0ba80865006b122ecc00`. It does not import experiment 09.
The original is preserved unchanged. The same B8 / 6 QPS 1,000-request benchmark
has now been repeated before any cleanup.

### Two supported decode vocabularies

- Default: the bundled **60,416-row** selected head.
- `--full-decode-lm-head`: the complete **103,424-row** checkpoint head.

There is no 16k mode, arbitrary vocabulary-path option, or expanded-head flag.
Both choices use the same decoder and scheduler and return greedy native IDs.
First-token selection after text prefill uses the full head in both modes.

All selected IDs, row ordering and checksum are self-contained in
`presets/table_compact_vocab/native_han_core_60416.json`: 60,352 reviewed protected
IDs plus 64 deterministic fillers, SHA256
`c730b5388f9871ead92e2cb484f8df81ba69518f1e8baeccc5a37f44c1514637`.
Loading does not tokenize text, read benchmark artifacts, import experiment 09,
or download anything. Full mode needs no separate ID list: its rows already use
the checkpoint's native vocabulary IDs. See the [vocabulary contract](presets/table_compact_vocab/README.md).

Decode cache identities remain `selected_vocab_<size>_<mapping-hash>` and
`full_vocab_<size>`. Defaulting to the already-tested 60k mapping does not change
its ID order, model operations or cache identity. The model source files and
vision/text-prefill cache identities are unchanged by this selection cleanup.
Readiness reports the actual head size and hash.

The 60k path was tested at `fb6532c1` on the same random-100 tables as the saved
16k/full controls: B2/C2 **2.927 tables/s, P95 1.928 s**; B8/C8 **5.788 tables/s,
P95 3.334 s**. Raw token streams matched the corresponding full-head control
100/100 at each B. This is not a general accuracy guarantee or a new 1,000-request
flagship validation. See [the results and raw evidence](../tmp/19_table_ocr_serving/lm_head_60416_20260911/README.md).
Historical 16k reports and reproduction scripts remain tied to their recorded
commits; they are not supported modes of the current product runtime.

## Structure

- `serve.py`: HTTP API, operational CLI, process lifecycle and readiness.
- `serving_runtime.py`: preparation, prefill and continuous-decode coordination.
- `paddle_ocr_vl_1_6_modeling.py`: checkpoint loading and model composition.
- `vision_prefill.py`: vision encoder/projector and its existing runtime.
- `text_prefill_and_decode.py`: both original text implementations, joined without
  changing computation; the prefill-only `_linear_tokenwise` helper is renamed
  `_prefill_linear_tokenwise` to avoid changing either implementation.
- `crop_processing.py`: image/prompt preprocessing and unchanged output formatting.
- `_support/`: temporary original dependencies, including the decode scheduler,
  checkpoint configuration and timing. Further
  consolidation is deferred until the simplified model passes NPU validation.
- `presets/`: the single frozen 60,416-row native-token vocabulary mapping.

No new scheduling, warmup or metrics design is introduced here. Synthetic constructor compilation, real
request warmup, asynchronous token transfer, KV4096 stopping behavior, CPU
lookahead, prefill admission and setup GC freeze remain intact.

## First simplification: text and vision model alternatives

The text module now directly implements the validated optimized serving computation.
`baseline`, `combined_apply`, the other experimental presets and 14 custom
operator-wrapper modules are removed. Decode computations no longer select
between the old normalization, rotary-factor or MLP implementations. The legacy
configuration class, preset dictionary, resolver, model arguments and CLI
selector are removed, along with their obsolete readiness fields.

The top-level `forward*` / `generate_ids*` reference APIs and direct vision/text
model forwards are removed. Parameter-owning modules and helpers used by the
staged runtime remain, with unchanged checkpoint names and shapes. HTTP serving
and a future Python API will use this same persistent engine. The public Python
submission/bulk-processing wrapper is deferred, not implemented here.

Text prefill now spells out FP32 softmax at each call site; the single-choice
getter and mode argument were removed. Decode's empty kwargs dictionary,
attention aliases and unused private packed-RoPE/KV arguments were also removed.
The other single-crop prefill computation is unchanged. An unreachable
decoder fallback was also removed: optimized decode uses AddRMSNorm.

Vision retains the BNSD PromptFA implementation, FP32 RoPE and the exact D72-to-D80
weight-padding computation. BSND/BSH alternatives, model-dtype softmax and the
masked sparse-mode-0 override were removed. There is no layout, sparse-mode,
softmax-mode or direct-forward attention environment configuration anymore.
The serving PromptFA function requires a mask, with BNSD and sparse mode 1
visible in the operator call. Input preparation supplies a mask even for an
exact-fill bucket. Manual vision attention and its selector are removed. The
activation-padding alternative and convolution patch-projection branch are now
removed: serving always prepares D80 weights and uses the validated linear
patch projection. Checkpoint parameter names/storage and the retained operation
order are unchanged. Runtime metadata now uses literal
values instead of calling removed getters; constructor compilation ordering is
unchanged.

Packed/batched text-prefill helpers and the batched/profile-guided vision router
are removed. The engine always admits and prefills independent crops; continuous
batched decode across requests is unchanged. Cache allocation now only creates
the zero-initialized separate K/V tensors used by this path: no `init_mode`,
packed storage or MHA-head override. The existing extra reserve of 32 private
cache slots/host token entries is deliberately retained, not retuned.

Cross-crop prefill packing is not part of this serving design. `be691de1`
explicitly served with `vision_packing="off"` and `text_packing="off"`; removing
unused packing machinery does not remove an optimization behind the chart.

The HTTP CLI now exposes model/device/batch size, host/port, queue/timeout/upload
limits, summary output and the existing three cache directories. The metrics
switches and cache-directory consolidation are deliberately deferred.
Internal values come directly from the validated serving contract, not a preset:

- FP16, TorchAir, KV4096 and the existing 4096 output-token ceiling/stopping rules.
- Frozen 60,416-row decode vocabulary mapped to native IDs, or the full head
  with `--full-decode-lm-head`; full checkpoint head for the first prefill token.
- Greedy argmax only; LaTeX preference/suppression policies and their unused
  scheduler buffers are removed.
- Input pixel bounds 28,224–802,816; vision buckets
  `256,384,512,640,768,1408,1920,2048,2944,4096`; text buckets
  `128,256,512,1024,1152`.
- Weight-padded vision attention, linear patch projection, setup GC freeze and
  the anchor's noncompact decode-control updates. No interruption-cap policy.

The endpoint accepts table crops only. The explicit full-head switch changes
the decode vocabulary; other operational controls do not change resolution,
token limits or model math. Text weights now require NZ;
vision-format choices and compiler machinery remain deferred. Readiness still
describes actual weight formats.

Image processing, vocabulary mapping, scheduling, startup warmup and GC remain
unchanged. Stage timings and queue/slot metrics remain; obsolete preset metadata
is removed, and prefill grouping metadata now describes independent crops.
The validated step-3 version preserved compiler source hashing and the active
graph-wrapper/cache-compile block. The later text sweep documented above removes
obsolete hash inputs and fixed cache tags, while retaining the graph-wrapper
construction and compile flags. Tests also check the unchanged vision key with
a fixed source hash. No cache identity is spoofed. The grouped text/vision cleanup was
compiled once in the step-3 validation, followed by a cached-process restart.
Future source changes may require compilation: discuss that explicitly before
performing further runs. Grouping the cleanup avoided a separate compile cycle
for each edit. This does not
guarantee that a discovered NPU failure could never require a further change.

CPU-only checks:

```sh
python3 -m unittest discover -s 19_table_ocr_serving/tests -p 'test_*simplification.py' -v
python3 19_table_ocr_serving/serve.py --help
```

The twenty-four tests use small CPU tensors and simulated NPU operations. For the
retained serving contract, B1/B2/B8, full/compact heads and two successive decode steps,
it checks exact operation traces/arguments, outputs and KV writes against
`dc755584`. Vision checks cover two D72 layers using the retained weight-padded
attention, FP16/FP32, exact and padded sequences, and B1/B2. They
compare outputs, prepared weights and operation-call traces. Tests also cover
removal of legacy forwards, checkpoint parameter identity, text-prefill outputs/KV, and required masks for
exact-fill and padded buckets. Source guards allow only explicit getter/softmax
inlining changes in preparation/runtime/compiler-adjacent code. Additional
checks compare the old independent-prefill branch with the cleaned execution,
verify the same open-request admission with the unused interruption cap removed,
and compare zero-initialized KV storage. Checks also compare fixed setup values
and the vocabulary hash with the saved B8 readiness record, and exercise HTTP
worker request/result wiring with a fake engine (no NPU or real OCR).
This is not real Ascend correctness, compilation or performance validation.
NZ setup tests cover conversion success, already-NZ weights, non-NPU weights,
cast exceptions, wrong returned formats and failures after a partial conversion.

The latest dead-plumbing pass removes two unread arguments, the rejected
`use_scatter_pa` selector, two write-only prefetch attributes and the single-step
prefetch construction loop. The active future-layer prefetch sequence is
unchanged. IncreFA still receives explicit `pse_shift=None` and
`actual_seq_lengths=None`, now directly at the operator call. The unused RoPE
alias, duplicate import and impossible recognizer-metadata branches are gone.
The full-head fallback, preprocessing alternatives, group machinery and
diagnostics are retained pending their separate discussions.

For subsequent read-only audits, Ruff's `ARG001`/`ARG002` rules flag unused
function/method arguments. Vulture can suggest unused names, but dynamic module
dispatch and registered buffers require manual review. Complexity scores are
navigation aids, not deletion criteria; an unexecuted branch is not necessarily
unreachable. No analysis tooling is added to serving dependencies.

### Remaining simplification decisions

- Define the Python interface over the same persistent engine later: blocking
  one-crop convenience, asynchronous submission and bulk-result iteration.
- NZ conversion and bucket padding are fixed for the supported checkpoint.
  Do not remove
  compatibility behavior based only on the flagship run exercising one case.
- Ascend operator behavior changes require review of the exact stack's supplied
  documentation. A 1,000-request regression run is not exhaustive edge coverage.

## Provenance checks

`relocation.json` is the frozen step-2 receipt: it maps copied source files to
their original Git blobs and new names. At step-2 commit `dc755584`, the CPU-only
migration check regenerates the expected mechanical
copy in memory and checks function/class bodies:

```sh
python3 19_table_ocr_serving/tests/check_relocation.py
python3 19_table_ocr_serving/serve.py --help
```

The historical relocation check intentionally no longer matches the simplified
working tree. Do not regenerate its receipt to disguise simplification as a
mechanical copy. Use the simplification tests above for this pass.

The receipt needs repository Git history, but the serving runtime does not.
`--materialize` is one-time bulk-copy tooling, not part of serving or warmup.
Do not use it after beginning intentional simplification.

## Runtime environment and benchmark

Use the existing 910B2 environment from the reproduction lock, with
`source npu-setup` before starting the worker. No packages need to be installed
or upgraded for this relocation. See `requirements.txt` for Python dependencies;
Ascend torch-npu, CANN and TorchAir must come from the already validated runtime.

The serving entrypoint no longer accepts internal implementation switches,
including `--decode-optimization`, `--dtype`, `--decode-backend`, `--token-selection`,
vocabulary overrides, pixel/token limits, bucket lists and vision toggles.
When adapting a historical validation command to 19, omit those switches: their
locked values are implemented directly. Preserve operational/load settings. Frozen
historical commands and receipts stay unchanged. Its cache
directory defaults use the experiment-19 namespace, so moving files cannot
silently masquerade as the old source fingerprint. The benchmark explicitly
passes the locked configuration and saves resolved readiness information.

The benchmark clients remain external validation tools: use the unchanged
historical clients and exact step-1 input sequence/Poisson schedule, not a newly
invented request generator. No benchmark metadata is used by the server.

Step-1 anchor: `tmp/09_persistent_page_engine/step1_b8qps6_20260910/README.md`.
It recorded 1.282711 s mean, 3.781594 s P95, 5.734013 completed requests/s,
zero errors and 1,000/1,000 native outputs identical to the original chart run.

## Step-2 validation

Runtime source commit: `dc755584`. Physical NPU6, same environment, historical
clients, 1,000-request sequence, Poisson schedule, warmup and B8 settings.

| Run | Mean (s) | P95 (s) | Completed requests/s |
|---|---:|---:|---:|
| Step 1, historical cached runtime | 1.282711 | 3.781594 | 5.734013 |
| Experiment 19, process that compiled fresh graphs | 1.360155 | 3.917355 | 5.730880 |
| Experiment 19, restarted with cached graphs | 1.283549 | 3.779594 | 5.733948 |

Both experiment-19 runs retained all 1,000 responses, zero errors, the same ten
KV-limit stops and **1,000/1,000 identical native token streams, raw text and
formatted outputs** compared with step 1. Schedules are byte-identical. Input
token counts, projected-image-token counts, crop dimensions and completion
reasons match for every occurrence. No new TEDS evaluation was run: complete
output equality establishes parity with the control.

The cached run differs from step 1 by +0.065% mean and -0.053% P95. Ownership
monitoring and direct-host checks confirmed no competing NPU6 process; our
server was stopped and NPU6 released after each completed benchmark.

**Startup qualification:** both runs used the same synthetic-constructor setup
and complete real-request warmup. The first process compiled new graphs; the
repeat loaded them from cache. The performance gap disappeared after restart
without code/settings changes. Setup froze 1,346,978 objects in the first
process versus 636,274 in the cached process (step 1: 636,312). This establishes
a startup-state difference, not proof that the object count itself caused the
gap. Keep the fresh-compile result; discuss startup behavior during step 3.

Evidence: `tmp/19_table_ocr_serving/STEP2_VALIDATION.md`, with all response,
configuration, command and ownership records in the adjacent run directories.

## Step-3 validation: simplified implementation

Runtime source `0976fa33`, physical NPU6 (Ascend 910B2), unchanged historical
clients, sequence, Poisson schedule and warmup. All 1,000 native token streams,
raw/formatted outputs, completion reasons, input-token counts, projected-image
counts and crop dimensions match the cached step-2 control. Zero errors; the
same ten KV4096-limit stops are retained. No new ground-truth score was computed.

| Metric | Cached step 2 | Cached step 3 |
|---|---:|---:|
| Mean (s) | 1.283549 | 1.317758 |
| P50 (s) | 0.862833 | 0.919765 |
| P95 (s) | 3.779594 | 3.790191 |
| P99 (s) | 7.259672 | 7.264818 |
| Completed requests/s | 5.733948 | 5.733937 |

P95/throughput closely reproduce the control; mean rose 2.7% and P50 rose 6.6%.
This single comparison does not establish whether those shifts are cleanup
effects or runtime variation. No additional tuning or repeat was performed.

The first process compiled and completed one real warmup, with zero measured
requests; setup took 397.929 s. It stopped before a second process loaded caches
(35.888 s setup), warmed identically and ran the measured replay. All 96 measured
phase ownership checks were clean; the final check confirms NPU6 released.

Full evidence and reproduction instructions:
`tmp/19_table_ocr_serving/STEP3_VALIDATION.md` and adjacent
`step3_b8qps6_0976fa33_20260911_*` directories. The original relocation receipt
remains unchanged. Deferred cleanup/performance discussions remain deferred.
