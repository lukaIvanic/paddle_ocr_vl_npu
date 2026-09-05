# B2 serving: exact normalization lookup and background image decode

Source `f0f9df06`, one Ascend 910B2 (physical NPU6). Ordinary B2 decode,
client concurrency two, same smaller vision/text buckets and non-blocking CPU
preparation as `table_cpu_poll_02584647_20260905/c2`. No oracle or speculative
route. The server accepts independent crop-image requests; source IDs do not
control preparation, routing or generation.

Two CPU changes are measured together, after an isolated CPU normalization probe:

1. A 256-entry arithmetic lookup reproduces FP32 normalization for uint8 pixels.
   Pixel values, Pillow resampling, normalization rounding and patchification
   remain bit-exact. Exhaustive channel-value tests and full resized-image tests
   pass. This caches arithmetic constants, not request image features or results.
2. The same `Image.open(...).convert('RGB')` operation moves from the inference
   thread to the existing CPU preparation worker. Its duration is recorded as
   `cpu_image_decode` and remains in each request's latency. Decode and prefill
   model execution are unchanged. Malformed images use the existing preparation
   failure callback; they are not silently dropped.

| Same frozen random-100 seed1 | Completed tables/s | P95 request latency |
|---|---:|---:|
| Previous improved B2 | 2.808024198737056 | 1.9147474726370988 s |
| Lookup + image-decode offload | 2.914567462409707 | 1.9137373317964366 s |

Measured wall: 34.31040841899812 s. This still misses the 3 tables/s goal, so
neither validation gate is authorized by this result. There is no removal of
initial fill, final drain, slow requests, CPU overlap or interruptions from timing.

All 100 requests complete with EOS and native tokens identical to the preceding
B2 run (which also matched the historical B2 outputs). Image dimensions and real
vision/text token counts are identical. `analyze.py` checks these assertions,
selection SHA256, client concurrency and ownership monitoring from saved records.

One complete `--set warm --count 1` request precedes measurement. Initialization
uses the existing runtime's graph cache/warm calls; there was no separate graph
builder. Server configuration, command, logs, per-request results and final
service statistics are preserved. Main-thread image work moved to the CPU worker
inside the same request—not to untimed client preparation.

PNG/RGB conversion totals 1.746516 s inside the background CPU spans. Total CPU
preparation service is 5.905566 s and includes that conversion; do not add the two.
Neither number is subtracted from request latency or claimed as an E2E saving.

Manual host evidence maps worker PID1775914 to container PID2470812 and its
owned API parent (`ownership_observation.txt`, retained from direct SSH output).
Every NPU6 process snapshot within measurement contains only
that worker. Approximately five-second sampling is not continuous kernel tracing.
The server exits zero after saving its summary. At 2026-09-06 02:21:24 +08:00,
direct host checks showed the server/worker gone and NPU6 free. The owned monitor
PID1775862 was stopped afterward. No benchmark is left running.
