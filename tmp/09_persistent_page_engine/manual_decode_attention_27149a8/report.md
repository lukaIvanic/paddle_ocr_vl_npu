# Manual decode-attention profile on Ascend 910B2

## Result

Replacing IncreFA with decomposed manual GQA removes IncreFA's unusually
large AIC Scalar span, but it makes the complete compiled decoder slower.
Removing the numerical `1/sqrt(128)` score multiply does not provide a
reliable speedup.

## Contract

- Source commit: `27149a8`
- Device: Ascend 910B2
- Backend: TorchAir static graph
- Shape: B1, KV capacity 1024, initial cache position 900
- Stack: complete 18-layer text transformer, LM head, and argmax
- Dtype: FP16
- Linear format: FRACTAL_NZ, 145/145 requested weights converted
- Non-attention implementation: `combined_apply`
- Throughput controls: 5 warmups and 50 measured graph calls
- Pipe profiles: 3 graph calls per lane

## Full compiled step

| Attention | Mean ms | Median ms | tok/s | vs IncreFA |
|---|---:|---:|---:|---:|
| IncreFA | 1.2293 | 1.2197 | 813.5 | baseline |
| Manual, scaled | 1.3573 | 1.3474 | 736.7 | -9.4% |
| Manual, unscaled | 1.3438 | 1.3343 | 744.2 | -8.5% |

The unscaled control is 1.0% faster than scaled manual attention in the
throughput control. The profiler does not confirm that as score-scale work:
TorchAir fuses the scale and mask in the scaled graph, and removing the scale
changes it to a standalone `MaskedFill`. The profiled kernel-duration sum is
6.3 us per graph higher, not lower, in the unscaled lane. Treat the 1.0%
control difference as graph-level noise or secondary scheduling variation.

## Attention kernels

Pipe fields overlap and must not be added to derive kernel duration.

| Kernel | Calls/layer | Duration us/call | AIC Scalar us/call | AIC MAC us/call | AIC MTE2 us/call | Cube utilization |
|---|---:|---:|---:|---:|---:|---:|
| IncreFA | 1 | 18.159 | 13.182 | 1.076 | 7.034 | 8.1% |
| Manual BatchMatMul | 2 | 4.362 | 0.600 | 0.448 | 1.935 | 61.2% |

Manual QK and PV together take 8.724 us/layer. Their combined Scalar span is
1.200 us/layer. Therefore, manual BatchMatMul does **not** inherit IncreFA's
Scalar bottleneck.

The full manual attention decomposition is slower because every layer also
executes these materialized operations:

| Manual-attention work | Calls/layer | Kernel duration us/layer |
|---|---:|---:|
| Expand two KV heads to sixteen heads (`BroadcastTo`) | 2 | 9.892 |
| QK and PV (`BatchMatMul`) | 2 | 8.724 |
| FP16/FP32 softmax casts | 2 | 3.351 |
| Scale plus mask, fused | 1 | 2.911 |
| Softmax | 1 | 2.115 |
| **Total listed attention work** | 8 | **26.993** |

IncreFA performs the corresponding fused work in 18.159 us/layer. The manual
path saves about 9.4 us/layer inside QK/PV compared with IncreFA, then loses
about 18.3 us/layer to KV expansion and separate vector kernels.

## Numerical check

Two real decode steps were compared with the IncreFA reference:

| Lane | Mean absolute logit difference | Maximum | Argmax |
|---|---:|---:|---:|
| Manual, scaled | 0.008401 | 0.061035 | 2/2 |
| Manual, unscaled | 2.043838 | 16.375 | 0/2 |

Both lanes completed with finite values. The unscaled lane is intentionally
numerically wrong.

## Interpretation

The `scale_value` multiply is a Vector operation. It is not the source of
IncreFA's AIC Scalar time. In the scaled manual graph, TorchAir fuses it with
masking into `MulsMaskedFill`; deleting the multiply does not even eliminate
an NPU kernel.

The useful finding is narrower: an explicit BatchMatMul implementation can
execute QK and PV with low Scalar time and much better Cube utilization. A
potential faster replacement would need to retain those matmuls while avoiding
physical GQA KV expansion and fusing scale, mask, softmax, and PV. The current
stock manual decomposition does not do that.

