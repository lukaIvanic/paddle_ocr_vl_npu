# 910B2 dense-bucket target-768 full E2E control

## Classification

`PASS_REGRESSION`

The full 1,651-page run completed successfully, but the dense bucket ladder and
768-token packing target were slower than the otherwise identical target-1024
control on Ascend 910B2.

## Provenance

- Project commit: `c06a9cf`
- Device: one Ascend 910B2, physical NPU 5 exposed as logical `npu:0`
- Runner: `09_persistent_page_engine/scripts/run_omnidocbench.py`
- Dataset: full OmniDocBench v1.6, offset 0, limit 1,651
- Page frontend: all pages first, staged layout with four input workers and
  eight page-finalization workers
- Decode: B64, KV2048, TorchAir, `combined_apply_pse_sentinel`
- Vision: TorchAir PromptFA, 4352-wide zero-extended MLP, FRACTAL_NZ weights
- Pixels: global 28224..401408; text crop scale 0.5
- Text prefill: production-group packing, buckets 128/256/512/1024
- Timeline: disabled

The configuration diff against
`910b_full_b64_pse_layout8_def2260/full/output/run_summary.json` contained only:

```text
vision_pack_target: 1024 -> 768
vision_buckets:
  128,256,384,512,640,768,1024,1408,1920,2048
  ->
  128,256,384,512,640,768,896,1024,1152,1280,1408,1536,1664,1792,1920,2048
```

## Measured result

| Metric | target-1024 control | dense target-768 | signed delta |
|---|---:|---:|---:|
| setup | 42.489 s | 237.809 s | +195.319 s |
| pipeline E2E | 717.705 s | 727.988 s | +10.283 s |
| pages/s | 2.30039 | 2.26790 | -0.03249 |
| vision prefill | 178.088 s | 203.540 s | +25.452 s |
| text prefill | 74.936 s | 76.407 s | +1.471 s |
| decode | 124.087 s | 126.486 s | +2.399 s |
| effective decode tok/s | 12,668.8 | 12,437.1 | -231.6 |
| raw decode tok/s | 13,372.8 | 13,128.3 | -244.5 |

The 195-second setup increase is missing-graph materialization and is excluded
from pipeline E2E. The dense run's `compile_first_call` was 12.852 seconds.

## Packing result

| Metric | target-1024 control | dense target-768 | signed delta |
|---|---:|---:|---:|
| real vision tokens | 8,651,324 | 8,651,324 | 0 |
| physical vision tokens | 9,512,832 | 9,274,368 | -238,464 |
| vision fill | 90.416% | 92.352% | +1.936 pp |
| vision groups | 8,781 | 11,230 | +2,449 |
| real text tokens | 2,560,072 | 2,560,072 | 0 |
| physical text tokens | 3,997,056 | 3,214,080 | -782,976 |
| text fill | 64.049% | 79.652% | +15.603 pp |
| text groups | 8,781 | 11,230 | +2,449 |

Dense target-768 vision histogram:

```text
256:403, 384:363, 512:541, 640:2515, 768:5733, 896:145,
1024:148, 1152:132, 1280:76, 1408:102, 1536:74, 1664:90,
1792:68, 1920:295, 2048:545
```

Dense target-768 text histogram:

```text
128:910, 256:8870, 512:1285, 1024:165
```

## Conclusion

On 910B2, padding reduction did not repay the 27.9% increase in vision/text
graph calls. This is a device-specific control, not a conclusion for 310P: the
optimized 310P physical-throughput curve peaks at different shapes and falls
much more strongly above 768. Phase 55 therefore measures the same change
directly on 310P rather than transferring the 910B result.
