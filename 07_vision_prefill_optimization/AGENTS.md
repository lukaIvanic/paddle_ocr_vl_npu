# Experiment 07 Agent Notes

## Scope

This experiment is only for PaddleOCR-VL vision-prefill measurement and optimization.

The reference target is:

1. native-resolution visual encoder output: `visual_features`
2. adaptive MLP projector output: `image_embeds`
3. text prefill next-token logits: `prefill_logits`

The authoritative baseline should be generated with fp16, eager/non-compiled execution, and
`PADDLE_OCR_VL_VISION_ATTENTION=prompt_flash_attention` on NPU unless the run is explicitly
marked as a CUDA/manual smoke. Decode, hot-swapping, EOS handling, and page-level layout
evaluation are out of scope here.

## Isolation Rule

Experiment 07 must not import code from earlier experiment folders. Keep needed model/config
code inside this folder. CUDA is acceptable for script development and orchestration checks, but
CUDA results are not NPU speed or correctness evidence.

## Development Rule

Work methodically from the system goal backward. Before adding a workaround, mode, or benchmark,
ask what future task it enables, what invariant it is supposed to preserve, and whether it moves the
project toward the real target: fast, accurate, static/compiled PaddleOCR-VL vision prefill on real
OCR crops. Local fixes that make one run look better but make batching, padding, compilation, or
accuracy review harder are not progress.

Prefer one general implementation path that handles both the full feature and the simplest case.
For example, the static visual path must support masked padding as the normal path, while any
no-padding/no-mask run is only a diagnostic control for isolating effects. Do not keep separate
padded and non-padded model paths just because the non-padded path worked first. During active
development, code is provisional unless it is explicitly identified as the reference contract or has
passed the relevant correctness, timing, and anti-cheat review.

Use tests as the safety mechanism instead of preserving stale fallback paths. If a cleaner path is
the one needed for the future system, implement it, run the equivalence and timing checks, and fix or
revert if it fails. If two paths are introduced for diagnosis, write the equivalence test and remove
the losing or redundant path after the test answers the question.

## Timing Rule

Use the default `--timing-mode standard` for candidate comparisons. It records two separate
measurements in the same run:

- `visual_tower_e2e_s`: pixel tensor already on the target device, static inputs such as
  `cu_seqlens` prepared outside the timed region, synchronize, call the visual tower, synchronize.
  This is the headline speed metric for visual-encoder optimization.
- `full_prefill_e2e_s`: a separate wrapper around visual tower plus adaptive MLP projector plus text
  prefill. This is secondary context, not the visual-tower throughput denominator.

Use `--timing-mode phase_sync` only when debugging phase breakdowns, because it synchronizes around
every named phase and is not representative pipeline latency.

## Compile Boundary

All candidate comparisons use the compile-shaped static visual boundary. Use
`--vision-compile-backend none` for the single noncompiled candidate path, and use
`--vision-compile-backend torchair` for the TorchAir candidate.

`static_visual` has two compile APIs over the same callable:

- plain `torch.compile(fullgraph=True, dynamic=False)` for diagnostics such as run-eagerly, graph
  dumps, and MSIT GE-vs-FX localization;
- `torchair.inference.cache_compile(..., dynamic=False, ge_cache=True)` for the warm production
  path, enabled with `--vision-use-torchair-cache-compile`.

Do not mix these up. Cache compile is a cold-start optimization, not a different model path. When it
is enabled, record `compile_api`, `uses_torchair_cache_compile`, `torchair_ge_cache`,
`torchair_cache_dir`, first-call timing, effective visual tok/s, and physical padded tok/s.

The NPU equivalence gate has passed: `static_visual` with `--vision-compile-backend none` matched
the stored eager PromptFA baseline with 0.0 diffs across the 64-crop truth bundle and the eager
self-check also had 0.0 diffs. Do not reintroduce a second noncompiled candidate path unless there
is a new diagnostic reason and a planned removal gate.

When checking whether compilation changed numerics, compare `--vision-compile-backend none` against
the same static visual path with the real compile backend. Those two should differ only by the
compile wrapper and backend lowering. The candidate path always uses the same padding-capable static
visual encoder. The normal policy is recorded as `static_visual_pad_policy` and always adds masked
dummy rows. On Atlas inference cards, masked PromptFA has a physical sequence alignment contract:
tiny shapes use 16 alignment, while normal crop shapes with `S > 128` use 128 alignment. The compiled
visual tower returns the physical padded output; the benchmark synchronizes and stops
`visual_tower_e2e_s` before slicing back to real rows for projector/logit correctness.

The first correctness split is padded eager versus the stored baseline, then padded compiled versus
padded eager/baseline. The `--debug-static-visual-no-padding`,
`--debug-static-visual-min-pad-tokens`, and `--debug-static-visual-pad-to-multiple` flags are
diagnostic only. Use no-padding only as a no-mask control, not as the normal path.

For masked PromptFA on the current 310P3/CANN 8.2.RC1 work box, use
`--vision-prompt-fa-mask-sparse-mode 1` unless a specific diagnostic is testing sparse-mode behavior.
The synthetic probe found that mode 1 honors the full custom padding mask while mode 0 ignores it on
this hardware/software stack. The padding mask is a full custom block mask, not a causal/default
mask. Sparse modes are not padding-at-start/end controls: modes 2/3/4 are causal/band patterns with
stricter mask constraints, and `actual_seq_lengths` is not supported for the 310P/Atlas-inference
PromptFA path this experiment targets. Running padded eager with mode 0 can invalidate the
padding-vs-compile split.
Before debating OCR-level padded drift, run `probe-promptfa-mask` on NPU. It uses synthetic Q/K/V
where the mask must visibly change the output, and checks whether the recommended mode with a
non-null mask matches the full-mask manual reference instead of the unmasked reference. The key
summary fields are `recommended_mask_sparse_mode` and `recommended_full_mask_semantics_passed`.

If padded eager matches the stored baseline but padded TorchAir diverges, run
`probe-promptfa-compile` before making model-level claims. This probe does not load the OCR model:
it compiles a tiny PromptFA-only module with `fullgraph=True, dynamic=False` and checks eager versus
compiled for `no_mask`, `all_false_mask`, and `block_mask` at 640/768 tokens. Its key summary field
is `compiled_second_matches_eager_all`. The probe output explicitly records that Experiment 07 uses
the diagnostic plain `torch.compile` API, not `torchair.inference.cache_compile`, so stale explicit
GE cache is not the default explanation for this probe. Pass
`--output outputs/promptfa_compile_probe.json` so the full JSON is saved without writing an inline
parser.

When a compiled static visual compare diverges, add `--validate-compiled-against-static-eager` to
the compare command before changing model code. This records a direct compiled-first-call versus
static-eager diff for the same candidate wrapper, both on the physical padded output and the sliced
real rows. If `real_rows` already diverges, the error is inside compilation/lowering of the candidate
path, not in the disk-baseline comparison or padded-row slicing.

For TorchAir visual diagnostics on the current 310P work box, stay on the default/max-autotune
executor. Do not add or request `reduce-overhead` runs for this hardware. To localize compiled
numeric failures, use `--torchair-run-eagerly` first: if run-eagerly matches static eager, the traced
FX graph is semantically fine and the failure is in GE/CANN graph execution; if it still diverges,
the traced graph or compiled boundary is already wrong. Use `--torchair-graph-dump-type pbtxt` with
an explicit `--torchair-graph-dump-dir` when we need to inspect whether masks, constants, and
operator attributes were captured as expected.

If run-eagerly matches but normal TorchAir diverges, use the native MSIT TorchAir dump path before
inventing custom tensor dumps. The MSIT docs describe `get_ge_dump_config()` for GE execution dumps,
`get_fx_dump_config()` for FX reference dumps, and `msit llm compare --my-path GE --golden-path FX`
for localization. Experiment 07 exposes this through
`--torchair-msit-dump-kind {ge,fx}` and `--torchair-msit-dump-dir`. Always use different base
directories for GE and FX; mixed dump directories invalidate the comparison. Start with one crop and
GE `--torchair-msit-dump-mode output` to limit dump size. If compare says inputs are needed, rerun
with `--torchair-msit-dump-mode all`. The committed runner is:

```sh
ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_msit_ge_fx_compare.sh
```

The runner executes the same static visual compare twice, once with GE dump and once with FX dump,
then runs `msit llm compare` if the `msit` CLI is on PATH. It is a diagnostic path, not a speed
benchmark. Do not ask the work agent to write inline dump parsers or extra scripts for this first
pass; inspect the generated JSON summaries and the MSIT compare output directory.

`msit_llm` is the optional MSIT LLM Python component, not part of TorchAir or torch-npu. The
benchmark prefers the official `msit_llm.dump.torchair_dump` helper when available, but it has a
local compatibility fallback that sets the same `CompilerConfig` dump fields directly. Missing
`msit_llm` should no longer block GE/FX dump collection. The official comparison report still needs
the `msit` CLI; install it in the active NPU env with
`python -m pip install msit && msit install llm`, then check with `msit check llm`. The runner now
also looks for `msit` next to `PYTHON_BIN`, so a non-activated conda environment can still work.
If valid dumps already exist and only the official compare failed, use
`OUT_ROOT=outputs/msit_ge_fx_... bash run_npu_msit_compare_existing.sh` instead of regenerating GE
and FX dumps. The compare scripts set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` by default; if
protobuf descriptor errors persist, pin protobuf with `python -m pip install 'protobuf==3.20.2'`.

Padding exists to make static fullgraph compilation and later batching possible while preserving
real-token math. Treat padded rows as implementation detail, not as a second model. The invariant is:
real tokens must not attend to padded tokens, padded tokens must not attend to real tokens, padded
rows must be excluded before downstream real-token consumers, and real-row
`visual_features`/`image_embeds`/`prefill_logits` must match the reference.

Current NPU finding: TorchAir-compiled PromptFA at PaddleOCR-VL's native vision head dimension
`D=72` is numerically unsafe. Attention-only repros showed deterministic dense drift and, when RoPE
is inside the compiled graph, deterministic NaNs in head 15. Padding only the PromptFA Q/K/V call
dimension to `D=80` or `D=96`, keeping `scale_value=1/sqrt(72)`, and slicing the output back to
`D=72` made the attention-only compiled path match eager PromptFA and removed the NaNs. Treat this
as a promising operator-contract workaround, not as a completed optimization, until the inline
single-layer repro and the full static visual compare also pass real-row feature/logit checks.
Report both the real head dimension and the PromptFA call head dimension in every candidate output.

The representative fixed-S fullgraph bucket compare is:

```sh
MAX_ITEMS=4 STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN=1024 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_static_visual_512_fullgraph.sh
```

This is the actual static prefill direction, not a control. It must report effective and physical
vision tok/s plus `summary.bucket_filter`. Crops over the fixed physical visual length are excluded
from that bucket, never resized, clipped, or truncated. The runner uses cache compile by default for
the TorchAir case; set `VISION_USE_TORCHAIR_CACHE_COMPILE=0` only when doing diagnostics.

The corresponding OCR generation check is:

```sh
MAX_ITEMS=4 MAX_NEW_TOKENS=128 STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN=1024 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_static_visual_generation.sh
```

It compares actual generated token sequences and decoded text from two paths: stored eager baseline
`visual_features` versus candidate static-visual `visual_features`, both flowing through the same
projector, text prefill, and static decode loop. Passing first-token argmax is not enough.

The batched transformer-layer check is:

```sh
BATCH_SIZE=4 MAX_ITEMS=8 STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN=1024 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_static_visual_batched_encoder.sh
```

This is the real batching direction. It does not batch raw crops, patch embedding, or absolute
position interpolation. It builds each crop prefix sequentially, stacks the fixed-S prefix tensors,
and batches only `encoder layers + post LayerNorm` over `[B, S_fixed, hidden]`. The headline speed
fields are `encoder_effective_tokens_per_s` and `encoder_physical_tokens_per_s`. The
`prefix_plus_encoder_*` fields are context, not the batched-transformer headline. If this fails,
report the first mismatching item and batch JSON; do not go back to same-pixel-shape bucket audits.

After that smoke passes, run the real batch-size speed sweep:

```sh
SWEEP_BATCH_SIZES="1 2 4 8" MAX_ITEMS=32 STATIC_VISUAL_FIXED_PHYSICAL_SEQ_LEN=1024 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_static_visual_batched_encoder_sweep.sh
```

This defaults to `SKIP_GENERATION=1` because the target metric is vision-transformer prefill speed,
not decode. `MAX_ITEMS=32` keeps the selected crop set identical for B=1,2,4,8. Report
`encoder_physical_tokens_per_s` first, then `encoder_effective_tokens_per_s`,
`prefix_plus_encoder_physical_tokens_per_s`, selected count, correctness counts, and the
`TORCHAIR_PHYSICAL_SPEEDUP` table. If `MAX_ITEMS` is changed, keep it divisible by the largest batch
size or explicitly report that different batch sizes used different selected item counts.

## Anti-Cheat Ledger

Add short notes here whenever we catch a mistake that could make future results misleading. Phrase
new notes as general research rules first, with project-specific examples only when they help.

- Do not compare against regenerated "truth" during candidate benchmarks. Generate the reference
  bundle once, store it on disk, and compare candidates to that stored bundle.
- For real-world speed metrics such as tokens/sec, define the timed boundary before running. State
  whether inputs are already on device, where synchronization starts and ends, and whether output is
  synchronized before stopping the timer.
- Keep critical compiled-section metrics focused on the compiled section. For the vision encoder,
  report the device-loaded visual-tower timing as the primary metric, and keep broader wrapper
  timings such as adapter/text-prefill timing as secondary context unless that wrapper is the target.
- For padding/batching work, judge the path by whether it enables the concrete system goal while
  preserving real-token outputs. Avoid adding ad-hoc cleanup of padded tokens merely because their
  values look odd; padded rows are implementation detail unless they can influence real rows or leak
  into downstream consumers.
- Avoid mode multiplication. If a future deployment needs a capability, make it part of the main
  contract and make the simplest no-op case flow through that same contract. Extra modes are
  acceptable only for a bounded diagnostic question and should come with a removal gate.
- Treat hardware semantics as empirical contracts. Vendor docs and examples are starting points,
  but for NPU-only operators the decisive check is a small probe that proves the exact behavior on
  the target card, CANN version, torch-npu version, dtype, layout, mask shape, and compile/eager mode.
- Think through downstream consequences before coding: how this affects later batching, fixed-shape
  compilation, cache reuse, truth-bundle comparisons, NPU-only operators, and the metrics we will
  use to decide whether the optimization is real.
- Replicate the realistic data path for the metric being claimed. Use real crops, real preprocessing
  outputs, real grid/token shapes, and the same dtype/attention implementation expected in the
  deployment path.
- Always report the conditions that change throughput interpretation: batch size, real token counts,
  physical padded token counts, max/context token length, padding or bucketing policy, filtered
  labels/items, warmup/repeat counts, compile cold/warm state, and cache hit/miss state if caching is
  involved.
- Do not claim compile compatibility from an uncompiled or differently shaped path. A compile claim
  must include the actual compiled boundary, backend, fullgraph/dynamic settings, and first-call or
  cache behavior.
- Do not use CUDA/manual smoke results as proof that NPU PromptFA is correct or fast. They only
  prove script wiring.
- Do not silently resize, crop, clip, or filter for speed beyond the normal PaddleOCR-VL
  preprocessor and the explicit crop-selection policy written into the manifest.
- Do not report padded-token throughput as effective throughput unless the output also reports
  real/effective token accounting.
- Do not let a missing model path trigger network setup. Experiment 07 requires a local model
  directory with `config.json`, `model.safetensors`, and `tokenizer.json`; HF download is disabled.
