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

Candidate timing defaults to `--timing-mode standard`. This records the visual
tower metric and the full-prefill wrapper metric separately in the same run.

The headline visual-prefill speed metric is `visual_tower_e2e_s`: the crop's
pixel tensor is already on the target device, static inputs such as `cu_seqlens`
are prepared, the script synchronizes, calls the visual tower, and synchronizes
again. The JSON reports both `visual_tower_effective_tokens_per_s` and
`visual_tower_physical_tokens_per_s`.

The secondary wrapper metric is `full_prefill_e2e_s`, which measures visual
tower plus adaptive MLP projector plus text prefill. Use `--timing-mode
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
dynamic=False`. This path currently uses plain `torch.compile`; it does not yet
load or write a TorchAir GE compile cache.

## CUDA Smoke

CUDA smoke uses manual attention only and is not authoritative NPU evidence:

```sh
bash run_cuda_manual_smoke.sh
```
