# 910B eager PromptFlashAttention comparison

Observed on 2026-07-16 from source commit `2c457c1`, using physical NPU 5 as
logical `npu:0`. Four five-page runs used the same process configuration and
TorchAir B=4 decode cache: manual versus BNSD PromptFlashAttention at the model
default `min_pixels=112896`, followed by the same pair at `min_pixels=56448`.

PromptFA was selected with:

```sh
PADDLE_OCR_VL_VISION_ATTENTION=prompt_flash_attention \
PADDLE_OCR_VL_VISION_PROMPT_FA_LAYOUT=bnsd \
python3 08_offline_e2e_b1/run_offline_e2e.py ...
```

This changes only each eager vision layer's attention implementation. Crop
shapes remain dynamic B=1 with no image-token padding. `run.json` records the
effective attention mode and PromptFA layout.

## Aggregate result

| `min_pixels` | Attention | Vision device | Encoder device | Projected tok/s | Run wall | Exact outputs |
|---:|---|---:|---:|---:|---:|---:|
| 112896 | Manual | 6.139986 s | 5.889214 s | 4,915.973 | 22.941522 s | reference |
| 112896 | PromptFA | 7.016048 s | 6.751744 s | 4,302.137 | 23.921233 s | 158/160 |
| 56448 | Manual | 6.521661 s | 6.261624 s | 3,248.099 | 22.964748 s | reference |
| 56448 | PromptFA | 6.742868 s | 6.475002 s | 3,141.542 | 22.984844 s | 158/160 |

Global PromptFA made the default-resolution vision stage 14.27% slower and the
half-area stage 3.39% slower. It changed the same two outputs in both regimes:
one 276-token mixed text/formula crop and one 150-token formula crop. Every
stop reason remained unchanged.

## Shape-dependent crossover

PromptFA loses on the many small crops and wins strongly on large crops:

| Projected tokens | Default PromptFA change | Half-area PromptFA change |
|---:|---:|---:|
| below 100 | n/a | +16.71% |
| 100-149 | +30.07% | +15.40% |
| 150-199 | +30.27% | +13.43% |
| 200-299 | +30.55% | +13.29% |
| 300-499 | -2.66% | -6.45% |
| 540 | -50.29% | -49.51% |
| 1,200-1,260 | -76.13% | -75.98% |

The two table crops fell from approximately 397/343 ms with manual attention
to 93/83 ms with PromptFA. Those savings were outweighed by a 13-30% penalty on
the 150 small crops below 300 projected tokens.

A post-hoc per-crop selection using manual attention below 300 projected tokens
and PromptFA at or above 300 would have reduced summed vision-device time by
10.19% at the default floor and 9.76% at the half-area floor. The ten selected
large crops all retained exact generated output in these full PromptFA runs.
This is evidence for a hybrid threshold experiment, not a measured hybrid E2E
result.

## Artifacts

```text
tmp/08_offline_e2e_b1/promptfa_bnsd_smoke/run.json
tmp/08_offline_e2e_b1/five_pages_uniform/promptfa_pair/manual_default/run.json
tmp/08_offline_e2e_b1/five_pages_uniform/promptfa_pair/promptfa_default/run.json
tmp/08_offline_e2e_b1/five_pages_uniform/promptfa_pair/manual_half/run.json
tmp/08_offline_e2e_b1/five_pages_uniform/promptfa_pair/promptfa_half/run.json
```
