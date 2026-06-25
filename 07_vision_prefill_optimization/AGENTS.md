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

Use `--timing-mode vision_tower` for candidate visual-encoder speed claims. It moves the crop's
pixel tensor to the target device, prepares static inputs such as `cu_seqlens` outside the timed
region, synchronizes, calls the visual tower, and synchronizes again. Report both
`visual_tower_effective_tokens_per_s` and `visual_tower_physical_tokens_per_s`.

Use `--timing-mode full_prefill_e2e` only when intentionally measuring visual tower plus adaptive
MLP projector plus text prefill. Use `--timing-mode phase_sync` only when debugging phase
breakdowns, because it synchronizes around every named phase and is not representative pipeline
latency. Legacy `--timing-mode e2e` is an alias for full-prefill timing and should not be used for
vision-tower speed claims.

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
- Do not call full-prefill timing "vision prefill" speed. The headline experiment 07 speed metric is
  the device-resident visual tower call only; adapter and text prefill are correctness checks here,
  not the main optimization target.
- Do not claim TorchAir vision readiness from a normal eager visual path. A candidate must use
  `--candidate-vision-path static_visual` with an actual compile backend and must record
  `compiled=true` plus per-item `vision_compile` metadata.
- Do not hide visual padding cost. Report effective real-token throughput and physical padded-token
  throughput together.
- Do not use CUDA/manual smoke results as proof that NPU PromptFA is correct or fast. They only
  prove script wiring.
- Do not silently resize, crop, clip, or filter for speed beyond the normal PaddleOCR-VL
  preprocessor and the explicit crop-selection policy written into the manifest.
- Do not report padded-token throughput as effective throughput unless the output also reports
  real/effective token accounting.
