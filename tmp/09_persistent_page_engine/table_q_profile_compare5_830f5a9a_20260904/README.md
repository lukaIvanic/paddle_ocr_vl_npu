# Sequential versus mixed M16 profile comparison

Three independent processes ran on an otherwise idle Ascend 910B2, physical
NPU 6. Each process used 10 real-input warmups, 50 measured calls, one profiler
warmup, and five profiled calls. Profile values below are per-call averages.

All lanes used the production-locked compact vocabulary, FP16, NZ weights, and
optimization presets. The mixed graph matched all 16 isolated native token IDs.

## Normal timing

| Lane | Median | P95 |
|---|---:|---:|
| B8Q1 | 1.122 ms | 1.131 ms |
| B1Q8 | 1.328 ms | 1.339 ms |
| Sequential sum | 2.449 ms | 2.470 ms |
| Mixed M16 | 2.468 ms | 2.475 ms |

The mixed graph is 18 microseconds, or 0.75%, slower by ordinary median
latency. This is practical parity.

## Device profile

| Kernel group | Sequential | Mixed M16 | Mixed saving |
|---|---:|---:|---:|
| Transformer matmuls, including LM head | 860.9 us | 510.7 us | **350.2 us** |
| Layout operations | 182.9 us | 439.4 us | **-256.5 us** |
| Q1 IncreFA | 304.8 us | 384.6 us | **-79.8 us** |
| Q8 QK and PV batch matmuls | 327.4 us | 299.9 us | 27.5 us |
| Q8 scaled masked softmax | 115.7 us | 173.6 us | **-57.9 us** |
| MLP activation | 106.1 us | 144.3 us | **-38.2 us** |
| RMSNorm | 183.2 us | 148.2 us | 35.0 us |
| RoPE | 124.0 us | 101.4 us | 22.7 us |
| KV scatter | 125.2 us | 131.0 us | -5.8 us |
| Remaining groups | 165.0 us | 137.9 us | 27.1 us |
| **Total device kernels** | **2,495.2 us** | **2,470.8 us** | **24.4 us** |

The shared M16 matmuls work. They remove 350 microseconds. The mixed graph then
spends almost all of that saving on layout and slower contextual execution of
the same attention and activation kernels.

## Matmul detail

| Matmul | Sequential | Mixed M16 | Saving |
|---|---:|---:|---:|
| QKV | 191.2 us | 120.0 us | 71.2 us |
| Output projection | 144.6 us | 78.1 us | 66.5 us |
| MLP gate | 158.4 us | 93.5 us | 64.9 us |
| MLP up | 152.0 us | 80.2 us | 71.8 us |
| MLP down | 180.0 us | 121.8 us | 58.2 us |
| LM head | 34.7 us | 17.0 us | 17.7 us |

Every matmul group is faster in the packed M16 graph. Matmul performance is not
the failure.

## Layout detail

| Layout operation | Sequential | Mixed M16 | Mixed overhead |
|---|---:|---:|---:|
| Strided slice | 0.0 us | 96.6 us | 96.6 us |
| Concatenate | 34.7 us | 45.0 us | 10.2 us |
| Split | 50.8 us | 38.2 us | -12.6 us |
| Transpose | 52.2 us | 148.4 us | 96.2 us |
| Fused confusion-transpose | 45.2 us | 111.3 us | 66.1 us |

The mixed graph adds 108 strided-slice kernels per call, six per transformer
layer. It has the same transpose counts as B1Q8, but those transposes take 162
microseconds longer in the mixed execution context.

## Conclusion

The packed transformer body is valid and matmul sharing saves substantial time.
The two-attention mixed graph does not convert that saving into latency. Layout
cost consumes 73% of the matmul saving by itself. IncreFA, softmax, and MLP
activation consume the rest.

The profiler does not distinguish L2 interference from TorchAir scheduling,
but it proves that these kernels become slower inside the mixed graph even when
their logical shapes do not change. Further small edits to this two-attention
design are unlikely to produce a useful speedup.
