# Experiment 07: Vision Prefill Optimization

This experiment isolates PaddleOCR-VL recognition prefill for real OmniDocBench
crops. It intentionally excludes decode, hot-swapping, EOS handling, and page
layout detection.

## Reference Bundle

Create the authoritative NPU baseline:

```sh
cd 07_vision_prefill_optimization
ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_make_reference_baseline.sh
```

The reference contract is eager/non-compiled fp16 PromptFA. It stores one `.pt`
file per selected crop under `baselines/promptfa_fp16_eager_64/tensors/` plus a
`reference_manifest.json`.

Each tensor file stores:

- `visual_features`: native-resolution visual encoder output
- `image_embeds`: adaptive MLP projector output
- `prefill_logits`: next-token logits after text prefill

## Candidate Compare

Compare a candidate path to the stored baseline:

```sh
python vision_prefill_bench.py compare \
  --baseline baselines/promptfa_fp16_eager_64 \
  --candidate-name candidate_name \
  --device npu:0 \
  --dtype fp16 \
  --vision-attention prompt_flash_attention \
  --output outputs/candidate_name.json
```

The compare command re-extracts the exact manifest crops, verifies tensor file
SHA256s, computes the same three intermediate targets, and reports per-item plus
aggregate diffs.

Candidate timing defaults to `--timing-mode e2e`, which uses one device
synchronize before and after the whole prefill call. Use `--timing-mode
phase_sync` only for diagnostic phase breakdowns because it synchronizes around
every named phase.

To test a compile-compatible visual boundary, use the static visual candidate:

```sh
python vision_prefill_bench.py compare \
  --baseline baselines/promptfa_fp16_eager_64 \
  --candidate-name static_visual_torchair \
  --device npu:0 \
  --dtype fp16 \
  --vision-attention prompt_flash_attention \
  --candidate-vision-path static_visual \
  --vision-compile-backend torchair \
  --static-visual-pad-mode none \
  --output outputs/static_visual_torchair.json
```

The static visual path hoists `cu_seqlens`, absolute position embeddings, and
vision RoPE out of the forward graph and compiles with `fullgraph=True,
dynamic=False`.

## CUDA Smoke

CUDA smoke uses manual attention only and is not authoritative NPU evidence:

```sh
bash run_cuda_manual_smoke.sh
```
