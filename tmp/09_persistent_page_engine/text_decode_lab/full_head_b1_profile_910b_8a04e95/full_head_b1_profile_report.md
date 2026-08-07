# PaddleOCR-VL full-head B1 decode profile

## Configuration

- Device: Ascend 910B2
- Batch size: 1 active slot
- KV capacity: 1024
- Initial cache position: 768
- Decode implementation: `combined_apply`
- Backend: cached TorchAir graph
- LM head: full 103,424 rows
- Dtype: FP16

The 100-step unprofiled run is the timing authority. The four-step NPU profile
is used only for operator attribution because profiling inflates execution.

## Production timing and host contribution

| Measurement | Per decode step |
|---|---:|
| Device step | 1.2381 ms |
| Host wall | 1.2475 ms |
| Non-device critical-path difference | 9.4 us |
| Host/non-device share of wall time | 0.75% |
| Device throughput | 807.7 tok/s |
| Host-wall throughput | 801.6 tok/s |
| Model graph plus argmax | 1.2239 ms |
| Post-graph device state update | 14.2 us |

The host is not the B1 bottleneck. The large host spans shown inside the
instrumented profiler are tracing overhead and are not production timings.

## Device operator attribution

The profiler reports 1.4007 ms of summed kernels per step, 13.1% above the
unprofiled 1.2381 ms step. The following percentages are therefore attribution
shares, not replacement throughput measurements.

| Device work | Profiled us/step | Share |
|---|---:|---:|
| Transformer MLP MatMuls | 379.5 | 27.1% |
| IncreFlashAttention | 322.9 | 23.1% |
| Attention projection MatMuls | 221.0 | 15.8% |
| Full LM-head MatMul | 163.7 | 11.7% |
| RMSNorm and AddRMSNorm | 69.3 | 4.9% |
| ApplyRotaryPosEmb | 46.5 | 3.3% |
| KV Scatter | 44.3 | 3.2% |
| Split operations | 27.6 | 2.0% |
| Automatic buffer fusion kernels | 27.4 | 2.0% |
| GatherV2 | 26.0 | 1.9% |
| Argmax | 13.9 | 1.0% |
| Other kernels | 58.6 | 4.2% |

MatMuls plus IncreFlashAttention account for 77.6% of profiled kernel time.

## Pipe behavior

The MatMuls are MTE2-heavy at B1. Pipe ratios overlap because hardware pipes
can execute concurrently.

| MatMul role | MTE2 ratio | MAC ratio | Cube utilization |
|---|---:|---:|---:|
| Packed QKV projections | 76.3% | 11.5% | 70.3% |
| Attention output projections | 77.9% | 12.2% | 66.1% |
| MLP gate/up projections | 79.2% | 10.7% | 75.2% |
| MLP down projections | 85.2% | 13.5% | 70.2% |
| LM head | 97.7% | 12.3% | 90.1% |

IncreFlashAttention is not cube-bound at this shape. Its average AIC pipe
ratios are 75.7% scalar, 40.7% MTE2, and 6.2% MAC; reported cube utilization
is 8.1%.

## Interpretation

The primary B1 decode cost is moving weights and executing the transformer
linears. Attention is the second large cost. The host, post-graph state update,
and argmax are small.

The synthetic-head experiment provides the less-intrusive estimate of the real
LM-head contribution. Reducing 103,424 rows to 16,384 rows improves the full
step by 8.1%, which implies that the full LM head accounts for approximately
9% of unprofiled decode time. Profiling attributes 11.7% to it because the
instrumented execution is slower.
