# Compact decode control: no meaningful serving gain

Source `2b0fc9f2`, one Ascend 910B2, physical NPU6. Same weight-padded vision,
optimized ordinary B2 decoder, native selected vocabulary, KV4096, prefill
buckets and client C2 as `table_vision_3b222363_20260905/cache_reload`.
Decode device timing is **enabled**. This does not combine the preceding
unsuccessful `--no-decode-device-timing` experiment.

Opt-in `--compact-decode-control` replaces five per-step tensor operations with
two: copying sampled tokens into a persistent next-token buffer, and adding a
persistent active-slot increment vector to cache positions. The sampled output
remains independent storage while D2H may read it; slot retirement cannot
overwrite it through the next-token buffer. Inactive positions remain zero,
and admission/retirement update the increment vector. Inactive generated IDs
are ignored; fresh admission overwrites the slot's token and KV state. Active
generation math, stopping rules, model graphs and native IDs are unchanged.

Twelve CPU scheduler tests pass, including B1/B2/B4/B8 completion/refill parity,
separate output storage, stable buffer identities and position/reset checks.
These do not substitute for the actual NPU request run below.

| Fixed development-100 seed1 | Completed tables/s | P95 request wall latency |
|---|---:|---:|
| Cached vision control | 2.8926129443484783 | 1.9431582124438127 s |
| Compact control | 2.8746560721729524 | 1.957017857010941 s |

All 100 requests complete with EOS and identical native tokens to that control.
Input dimensions, real vision/text-token counts and frozen membership/order
match. Maximum client outstanding requests remains two over the entire
pipeline. No timing subtraction, work removal, queue exclusion or request
selection change was introduced. No validation gate was run: throughput fails.

The service lifetime (including one sacrificial warm request) executes 24,170
decoder calls, same as the control. Their device-event regions total
29.178924245953745 s, compared with 28.802045722484497 s for the control.
Thus the residual cadence difference was not fixed by removing the small
control operations. No claim of kernel-only profiling is made from these event
regions. The flag remains off by default. Further similar small-op changes
are not supported by this result; next work should profile the real serving
loop, separating model kernels and host/copy gaps.

Server setup took 35.257091 s with cached graphs. Commands, configurations,
warmup and per-request records are saved. `analyze.py` audits this run and the
preceding event-removal run, including honest null device metrics there.

Direct-host evidence maps worker PID1802256 to container2481718 and the owned
API parent1802247. Approximately five-second NPU6 process samples bracket the
measurement and contain only our worker during it. This is sampled monitoring,
not continuous tracing. Server/client exit zero. At 2026-09-06 03:16:02 CST,
the direct host reports no NPU6 process and neither worker nor parent remains.
Owned monitor1801981 was stopped. No other process or NPU was affected.
