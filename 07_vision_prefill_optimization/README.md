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
again. When static padding is enabled, this timed visual tower returns physical
padded rows; the script slices back to real rows only after the sync/timer stop,
before projector/logit correctness checks. The JSON reports both
`visual_tower_effective_tokens_per_s` and `visual_tower_physical_tokens_per_s`.

The secondary wrapper metric is `full_prefill_e2e_s`, which measures visual
tower plus adaptive MLP projector plus text prefill. Use `--timing-mode
phase_sync` only for diagnostic phase breakdowns because it synchronizes around
every named phase.

Candidate compare always uses the compile-shaped static visual boundary. Use
`--vision-compile-backend none` for the noncompiled path, and use the TorchAir
backend to test compilation:

```sh
python vision_prefill_bench.py compare \
  --baseline baselines/promptfa_fp16_eager_64 \
  --candidate-name static_visual_torchair \
  --device npu:0 \
  --dtype fp16 \
  --vision-attention prompt_flash_attention \
  --vision-compile-backend torchair \
  --output outputs/static_visual_torchair.json
```

The static visual path hoists `cu_seqlens`, absolute position embeddings, and
vision RoPE out of the forward graph and compiles with `fullgraph=True,
dynamic=False`. This path currently uses plain `torch.compile`; it does not yet
load or write a TorchAir GE compile cache.

The static visual candidate has one padding-capable encoder path. The automatic
padding policy is reported as `static_visual_pad_policy`, and zero padding is
the no-padding case inside the same path rather than a separate mode. The visual
tower timing stops after the synchronized physical padded output; real rows are
sliced only afterward for projector/logit correctness checks.

The NPU equivalence gate has already passed: backend `none` matched the stored
eager PromptFA truth bundle with 0.0 diffs before the padding path was unified.
After changes to the static visual path, rerun backend `none` first and only
interpret TorchAir results if the noncompiled static path still matches the
stored baseline.

## CUDA Smoke

CUDA smoke uses manual attention only and is not authoritative NPU evidence:

```sh
bash run_cuda_manual_smoke.sh
```
