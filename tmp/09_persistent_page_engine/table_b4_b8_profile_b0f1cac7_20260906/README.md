# Real B4/C5 versus B8/C5 serving profiles

Source `b0f1cac7`, one910B2, physical NPU6. These are diagnostic runs of the
ordinary crop API, not a decode lab and not qualifying latency measurements.
They use the same packed-MLP/complete-prefetch/RoPE-LUT/native16384/linear-patch/
setup-GC-freeze/no-decode-device-timing settings as the preceding real controls.
KV4096, pixels, greedy output policy and client admission C5 are unchanged.

`profile_api.py` wraps the actual arena step and leaves serving intact. Capture
begins after32 eligible real steps, then5 profiler warmups and20 measured host
iterations. Eligibility is4 active rows for physical B4 and5 for B8. Each run
has a separate model process and one full HTTP warm request before the same
frozen development100 seed1/C5 workload. No synthetic graph build is introduced.

| Mean per complete device graph | B4/C5 | B8/C5 | B8 minus B4 |
|---|---:|---:|---:|
| IncreFA |442.002us|611.693us|+169.691us|
| Matmuls incl. compact head |384.284us|390.794us|+6.510us|
| KV scatter |71.932us|116.330us|+44.398us|
| ApplyRotary |71.368us|92.165us|+20.797us|
| AddRMSNorm |76.850us|87.380us|+10.530us|
| SwiGLU |53.133us|65.347us|+12.213us|
| All model kernels (sum) |1192.914us|1462.897us|+269.983us|
| Model device envelope |1272.488us|1544.417us|+271.929us|

Matmul cost barely rises while attention/cache/elementwise work grows. This
supports testing exact B5 instead of padding five requests to B8. It does not
guarantee B5 selects a favorable tiling or meets the serving goal. No new
attention kernel or scheduler is needed: the engine already preserves exact
positive batch sizes. The subsequent B5 test is a separate unprofiled gate.

## Timing boundaries and interpretation

Both captures contain20 host `serving.decode_step` scopes, but21 COMPLETE
device graphs due to the asynchronous queue-depth-one boundary, plus a partial
earlier graph. `analyze.py` finds `UpdateModelParam_static_bin` boundaries,
verifies each complete graph has18 IncreFA,73 MatMul and one ArgMaxV2, and
divides kernel totals by21. Partial preceding rows are discarded explicitly
(21 rows B4,66 B8); they are not silently charged to20 graphs. Host scopes
still use their actual20-count denominator. Each capture has5 profiler warmup
observations and20 measured observations. Both report1800MHz AI Core frequency.

All20 measured B4 iterations have4 live slots; B8 has5. Representative cache
positions at the measured start are B4 `[271,1010,829,626]`, B8
`[263,1010,829,626,1049,null,null,null]`; each live row advances19 by the last
observation. Thus these are faithful serving states from the same request
sequence, NOT identical synthetic KV contents/positions between batches.
The raw capture metadata preserves every observation and source request ID.

Host nested scopes are not additive: B4 arena step~565us, cache compiler~238us,
graph Run~130us, dependency-event recording~128us; B8~608/240/124/126us.
The profiler itself increases CPU/control cost and synchronizes during export.
Its client wall/P95 cannot be treated as performance evidence or adjusted by
subtracting profiler overhead. Use the prior plain-API B4/C5 and B8/C5 runs for
throughput/latency comparisons. Each profiled client still completes100 requests.

The existing analyzer accepts either the historical `analysis_input` layout or
the direct `*/ASCEND_PROFILER_OUTPUT` export. Run it on `b4_capture` and
`b8_capture`; both `analysis.json` files retain raw-trace SHA256 values. No
generated text is encoded, and no request output is used to route inference.

## Commands and ownership

`b4_command.txt` and `b8_command.txt` record exact interpreter/argument arrays,
device, commit and profiler environment. Launch appends `--freeze-setup-gc
--no-decode-device-timing --service-summary-output <service.json>`.
Warm client: `--set warm --count 1 --max-in-flight 1`. Workload client:
`--set random --count 100 --shuffle-seed 1 --max-in-flight 5`, ordinary endpoint
`http://127.0.0.1:8767/v1/ocr`, distinct output dirs. Payload preparation is
before timing, as in the real benchmark.

B4 host parent/worker2029431/2029433 map to container2549168/2549170.
B8 host parent/worker2033440/2033442 map to2550330/2550332. Full commands,
NSpid mappings and physical NPU6 listings were manually checked. Shared monitor
2028986 samples only NPU6; `host_npu6_monitor.log` preserves its evidence.
Both full warm requests and clients exit0. Both owned server parents are
terminated gracefully, server exits0, and NPU6 is verified empty between runs
and before the subsequent exact-B5 launch. No other user process is touched.
