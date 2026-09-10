# Experiment 19: table OCR serving

**First model simplification applied; CPU checks pass. This edit has not been
run or compiled on an NPU.** The step-2 results below describe the earlier
validated source commit, not a new measurement of the simplified code.

This self-contained runtime is copied from experiment 09 at
`be691de190ae099d1a9b0ba80865006b122ecc00`. It does not import experiment 09.
The original is preserved unchanged. The same B8 / 6 QPS 1,000-request benchmark
has now been repeated before any cleanup.

## Structure

- `serve.py`: HTTP API, operational CLI, process lifecycle and readiness.
- `serving_runtime.py`: preparation, prefill and continuous-decode coordination.
- `paddle_ocr_vl_1_6_modeling.py`: checkpoint loading and model composition.
- `vision_prefill.py`: vision encoder/projector and its existing runtime.
- `text_prefill_and_decode.py`: both original text implementations, joined without
  changing computation; the prefill-only `_linear_tokenwise` helper is renamed
  `_prefill_linear_tokenwise` to avoid changing either implementation.
- `crop_processing.py`: original image/prompt preprocessing.
- `_support/`: temporary original dependencies, including the decode scheduler,
  configuration compatibility, timing and table-output formatting. Further
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

**Follow-up after cleanup and validation:** revisit cross-crop vision/text
prefill packing as a performance experiment. `be691de1` explicitly served with
`vision_packing="off"` and `text_packing="off"`; removing those unused alternatives
does not remove an optimization behind the chart. Packing could improve prefill
efficiency, but its end-to-end benefit and decode-interruption cost still need
measurement. Do not reintroduce or benchmark it during this cleanup.

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
resolution, vocabulary, token limits or model math. Native/NZ compatibility and
compiler machinery remain deferred; readiness still describes actual formats.

Image processing, vocabulary mapping, scheduling, startup warmup and GC remain
unchanged. Stage timings and queue/slot metrics remain; obsolete preset metadata
is removed, and prefill grouping metadata now describes independent crops.
Compiler source hashing and the active graph-wrapper/cache-compile block are
unchanged. Cache-name metadata uses the same resolved literals as before;
tests check the vision key with a fixed source hash. Deleted inactive files use the hasher's existing `nohash`
handling. No cache identity is spoofed and no compilation has been run.
Source changes may cause a later NPU launch to compile: discuss that explicitly
before performing any such run. Text and vision cleanup are grouped before that
validation, avoiding a separate compile cycle for each edit. This does not
guarantee that a discovered NPU failure could never require a further change.

CPU-only checks:

```sh
python3 -m unittest discover -s 19_table_ocr_serving/tests -p 'test_*simplification.py' -v
python3 19_table_ocr_serving/serve.py --help
```

The sixteen tests use small CPU tensors and simulated NPU operations. For the
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
- Native versus NZ weight handling and exact versus bucket padding remain for
  separate review. Do not remove
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
