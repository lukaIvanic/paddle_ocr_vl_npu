# 910B full MinerU E2E repeat and lower crop-cap comparison — 2026-10-09

Status: baseline submitted; no new performance or accuracy result yet.

User request: reproduce the near-1-page/s full E2E result, then reduce the
maximum crop pixel count and check accuracy. Both lanes use all 1,651 original
OmniDocBench page images, live PP-DocLayoutV3, fresh MinerU recognition, unchanged
predictions, and the frozen full evaluator.

The first lower cap is an author-selected experiment: 401,408 pixels / 2,048 raw
vision tokens, compared with 602,112 pixels / 3,072 tokens. Only that maximum
changes in the inference command. Minimum pixels stay 25,088. This reduces the
maximum by one third; actual total token savings must be measured.

Inference uses the clean exact d4e7fdd0efd90bf6408c21efab320cc3292f07a4 checkout
for both runs, original vision precision, FP16 MinerU, FP32 eager layout,
compiled vision/text prefill, B32/KV4096 decode, and the same existing cache roots.
The coordinator gains an optional --processor-max-pixels argument and validates
the selected cap; its default remains the original 602,112. No model or processor
implementation changes are part of this experiment. Baseline uses the coordinator
at aef97137; the lower-cap coordinator revision will be recorded before launch.

Verified environment: torch 2.10.0+cpu, torch-npu 2.10.0, Transformers 5.5.4,
mineru-vl-utils 1.0.5, kornia-rs 0.1.14, shapely 2.1.2. Physical 910B2 card 2 was
healthy and free at preflight; the prior reference used card 3. The host has 192
CPUs, all available to this process; observed initial load was about 27.

Each lane runs a two-page smoke then a fresh full run with no resumed pages or
warmup-page subtraction. Record setup separately, host/device state before and
after, exact commands, durable exits, real/padded vision tokens, generated-token
counts, stage times, and output length caps. Run evaluation after inference, not
concurrently with the next timed lane. Compare text accuracy, Page TEDS, Page CDM,
overall, available category scores and page-level regressions; an unchanged
aggregate alone does not establish that no pages degraded.

Baseline run root:
`/workspace/repos/paddle_ocr_vl_npu_mineru_lengths/tmp/11_mineru_2_5_pro_inference/d4_cap3072_repeat_910b_20261009T110942Z`

All new measurements in this report are 910B-only. The 310P handoff is unchanged.
