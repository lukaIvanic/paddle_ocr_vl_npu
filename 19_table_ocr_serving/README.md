# Experiment 19: table OCR serving

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

### Full versus trimmed decode-head comparison (CPU-tested only)

`serve.py --full-decode-lm-head` skips construction of the selected-row head
and its native-ID map. Omitting the flag preserves the 16,384-row trimmed
default. Both decode branches return greedy native token IDs from inside the
stage; the full branch no longer returns a logits tensor to the scheduler.
The checkpoint full head remains used for first-token selection after text
prefill in both variants. No first-token handling or scheduler change is included.

Serving uses separate decode cache roots: `full_vocab_<size>` versus
`selected_vocab_<size>_<mapping-hash>`. Vision and text-prefill roots are unchanged
between variants. Readiness records `full_decode_lm_head` and the vocabulary
metadata. This is a comparison switch, not a decision to retain trimming.

All 25 CPU tests pass, including full-head greedy IDs against the old logits,
unchanged transformer operation traces/KV writes, head setup/cache separation,
and CLI-to-worker selection. NPU compilation and throughput/latency measurements
for this comparison have not run yet.

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
- `presets/`: unchanged 16,384-row native-token vocabulary mapping.

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
- Frozen 16,384-row decode vocabulary mapped to native IDs; full checkpoint head
  for the first token after text prefill.
- Greedy argmax only; LaTeX preference/suppression policies and their unused
  scheduler buffers are removed.
- Input pixel bounds 28,224–802,816; vision buckets
  `256,384,512,640,768,1408,1920,2048,2944,4096`; text buckets
  `128,256,512,1024,1152`.
- Weight-padded vision attention, linear patch projection, setup GC freeze and
  the anchor's noncompact decode-control updates. No interruption-cap policy.

The endpoint accepts table crops only. Operational controls do not change
resolution, vocabulary, token limits or model math. Text weights now require NZ;
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
