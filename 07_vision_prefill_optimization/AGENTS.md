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

Do not preserve parallel implementation paths just because one path existed first. During active
development, code is provisional unless it is explicitly identified as the reference contract or has
passed the relevant correctness, timing, and anti-cheat review. If two paths are introduced for
diagnosis, write the equivalence test and remove the losing or redundant path after the test passes.

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

As of this experiment version, `static_visual` uses `torch.compile(fullgraph=True, dynamic=False)`.
It does not use TorchAir `cache_compile` / GE cache loading yet. If compile caching is added later,
record the cache directory, cache hit/miss state, and cold versus warm first-call timing explicitly.

The NPU equivalence gate has passed: `static_visual` with `--vision-compile-backend none` matched
the stored eager PromptFA baseline with 0.0 diffs across the 64-crop truth bundle and the eager
self-check also had 0.0 diffs. Do not reintroduce a second noncompiled candidate path unless there
is a new diagnostic reason and a planned removal gate.

When checking whether compilation changed numerics, compare `--vision-compile-backend none` against
the same static visual path with the real compile backend. Those two should differ only by the
compile wrapper and backend lowering. If `--static-visual-pad-mode` is not `none`, the candidate path
also adds masked dummy visual tokens and slices them away before returning `visual_features`.

Current NPU finding: TorchAir-compiled PromptFA is not numerically usable yet. On the 4-crop smoke,
physical throughput rose to about 11.7k tok/s versus about 7.5k tok/s for backend none, but
`argmax_match_count` was only 2/4, visual max-abs drift reached fp16-max scale, and compiled outputs
contained NaN/Inf. Do not report compiled PromptFA speed as a valid optimization until correctness is
fixed.

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
