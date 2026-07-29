# Full 27-layer vision RoPE comparison

This comparison measures the complete PaddleOCR-VL 1.6 vision-transformer
stage, not an isolated RoPE operator. Every lane uses B1 x S2048, the
zero-extended 4352-wide MLP, D80 weight-padded attention, native ND Linear
weights, compiled PromptFlashAttention, and 10 device-event samples with five
complete 27-layer replays per sample.

All runs used physical NPU 5 on the same Ascend 910B2 environment
(`torch==2.10.0+cpu`, `torch_npu==2.10.0`). Setup and compile time are excluded
from replay latency.

| RoPE lane | Median full-stage replay | Physical tok/s | D80 raw parity |
| --- | ---: | ---: | --- |
| Separate manual FP32, historical control | 25.7301 ms | 79,595.5 | control |
| Separate manual FP32, warm repeat | 25.5523 ms | 80,149.2 | exact |
| Joint-QK manual FP32 | 24.3146 ms | 84,229.1 | exact |
| Joint-QK native in-place, interleaved | 187.1045 ms | 10,945.8 | mean abs 0.00293; finite |

Against the end-of-sequence warm control, joint manual lowers latency by
4.84% and raises physical throughput by 5.09%. Its independent first run was
24.2970 ms / 84,290.2 tok/s, so the result reproduced within 0.08%.

The native in-place lane is rejected. It compiled and replayed through an
explicit TorchAir converter, and its output remained finite, but replay was
7.32x slower than the warm control. This is sustained execution cost across
all ten samples, not first-call compilation.

## Full-graph profile

The profile covers three complete replays of each graph. Counts below are per
replay.

| Kernel family | Separate manual | Joint manual |
| --- | ---: | ---: |
| PromptFlashAttention | 27 | 27 |
| MatMulV3 | 54 | 54 |
| MatMulV2 | 108 | 108 |
| StridedSliceD | 108 | 54 |
| Mul | 108 | 54 |
| Add | 54 | 27 |
| Cast | 54 | 27 |
| Neg | 54 | 27 |
| RoPE/QKV ConcatV2D total | 81 | 54 |
| Transpose | 108 | 81 |
| SplitVD | 27 | 0 |

The summed kernel duration fell from 25.7465 to 24.3143 ms per replay.
PromptFA and all 162 Linear calls remained unchanged; the gain comes from
applying the existing FP32 half-RoPE formula to one contiguous QK tensor and
performing one combined Q/K layout conversion.

## Conclusion

The lab result supports the portable joint-QK manual path as the only current
winner. It uses ordinary PyTorch operations and is structurally portable to
310P, although 310P compile/replay remains to be executed there. This commit
does not change the production vision stage; a real-crop/E2E validation should
precede production integration.

Evidence:

- `../rope_baseline_repeat_e12cfe8/result/run_summary.json`
- `../rope_joint_manual_b1s2048_95bd2ca/result/run_summary.json`
- `../rope_joint_manual_profile_e12cfe8/result/run_summary.json`
- `../rope_joint_manual_profile_e12cfe8/result/parsed_profile_summary.json`
- `../rope_inplace_b1s2048_3a98867/result/run_summary.json`
