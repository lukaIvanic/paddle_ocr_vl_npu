# Real B2 serving-loop profile

Source `1e32b233`, one Ascend 910B2, physical NPU6. The diagnostic adapter
launches the unchanged crop API with its current B2 optimized decoder, KV4096,
native selected vocabulary, smaller prefill buckets and weight-padded vision.
Normal control and device timing remain enabled. No synthetic model inputs,
different attention backend, fixed-step decode substitute, or manual per-step
synchronization is introduced.

One complete warm request precedes two real random-seed1 requests
(`page_000287_table_box_id_8`, `page_001367_table_51`). The adapter waits for
32 actual two-active-slot decoder calls, then performs 5 profiler warmup steps
and 20 active captured steps. All 25 observations retain two active requests.
Captured positions span `[267,1010]` through `[286,1029]`. The ordinary loop is
restored after capture and both requests finish with EOS. These requests and
their timings are **diagnostic**, not a benchmark or validation gate.

## Results and boundary correction

The capture begins while a prior asynchronously submitted graph is finishing.
Naively dividing all CSV rows by 20 counts 375 IncreFA calls. `analyze.py`
removes that prior partial graph's 201 rows and checks 20 complete graphs,
each with exactly 18 IncreFA, 91 matmuls and one ArgMax. This is a profiler
boundary correction, not request/tail exclusion from a latency distribution.

| Per complete captured iteration | Mean |
|---|---:|
| Model kernel-duration sum | 1,215.508 µs |
| Model device envelope | 1,295.475 µs |
| Five ordinary control kernels | 11.455 µs |
| Instrumented graph-start cadence (19 intervals) | 1,718.026 µs |
| All matmuls, including LM head | 517.259 µs |
| IncreFA | 386.151 µs |
| InplaceAddRmsNorm | 68.576 µs |
| ApplyRotaryPosEmb | 59.294 µs |
| KV scatter | 56.932 µs |

Matmuls and attention account for ~74% of model kernel time. The five tiny
control kernels account for less than 1% of the model-kernel total. Their low
device cost is consistent with the absence of a meaningful full-serving gain
from simplifying them.

Host scopes are nested and **not additive**: `serving.decode_step` averages
873.651 µs, `cache_compiler inference` 254.180 µs, and
`TorchNpuGraphBase::Run` 134.850 µs. Eighty `Event::record` scopes total
350.457 µs per iteration across all four records. The profiler materially
inflates event/host overhead and observed cadence; these numbers cannot be
subtracted from unprofiled request latency or assumed to be recoverable gaps.
The prior unprofiled cache-loaded B2 device-event average was ~1.192 ms, not
the instrumented 1.718 ms start cadence. This profile alone does **not** explain
the remaining ~48 µs/call difference versus the earlier 1.144 ms control.

Two recorded AI-core frequency samples both report 1,800 MHz; this is evidence
for this capture only, not the frequency history of earlier benchmarks.

## Evidence and cleanup

`capture_1e32b233/analysis.json` contains the detailed kernel breakdown and
scope totals. CSVs and a losslessly gzip-compressed Chrome trace are retained
under `analysis_input`; the analyzer works with either raw or compressed trace
and records its SHA256. The original full binary/raw capture remains remotely
under the `profile` directory recorded in `command.txt`. No 100-table
performance result is claimed from profiling.

Source/commands, configuration, actual request outputs, profile observations
and server logs are saved. Host PID1812359 maps to container2485007 and its
owned API parent1812353. Direct-host snapshots show only this worker during
the capture; monitoring is sampled, not continuous kernel tracing. The
server exits zero after saving its summary. At 2026-09-06 03:32:18 CST, the
host confirms NPU6 free and the worker/parent gone. Owned monitor1812319 was
stopped. No other NPU or user's process was affected.
