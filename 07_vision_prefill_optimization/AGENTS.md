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

## Timing Rule

Use `--timing-mode e2e` for candidate speed claims. It synchronizes once before and once after the
whole prefill call. Use `--timing-mode phase_sync` only when debugging phase breakdowns, because it
synchronizes around every named phase and is not representative pipeline latency.

## Compile Boundary

Vision candidates that need TorchAir should use `--candidate-vision-path static_visual` plus
`--vision-compile-backend torchair`. The normal `eager_visual` path is the reference-style runtime
path and is not a static fullgraph vision boundary.

## Anti-Cheat Ledger

Add short notes here whenever we catch a mistake that could make future results misleading.

- Do not compare against regenerated "truth" during candidate benchmarks. Generate the reference
  bundle once, store it on disk, and compare candidates to that stored bundle.
- Do not report `phase_sync` timings as serving throughput. They are diagnostic sync-heavy phase
  timings.
- Do not use CUDA/manual smoke results as proof that NPU PromptFA is correct or fast. They only
  prove script wiring.
- Do not silently resize, crop, clip, or filter for speed beyond the normal PaddleOCR-VL
  preprocessor and the explicit crop-selection policy written into the manifest.
- Do not report padded-token throughput as effective throughput unless the output also reports
  real/effective token accounting.
