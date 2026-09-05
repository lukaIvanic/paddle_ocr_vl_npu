# Observed-cadence admission oracle — no P95 improvement

Source HEAD `7ced563ad8f576a9548e2b7146ffb29999285490`; the relevant ordinary
runtime and diagnostic adapter are unchanged from `02fe5645`. Physical NPU6,
Ascend 910B2. This completes the previously unrun observed-cadence variant.
No qualifying validation gate was attempted.

Same frozen development-100 seed1, B4/C3, ordinary greedy decode, KV4096,
output limit4096, compact native vocabulary, unchanged real image/token counts.
Separate cache-loaded server lifetimes and one full-request warmup per run.
Setup durations were35.612573s and36.427334s; no new graph compilation needed.

| Mode | Returned tables/s | P95 request seconds |
|---|---:|---:|
| Ungated matched control | 3.556697157335719 | 2.284986938780639 |
| Observed-cadence oracle deferral | 3.530285588678326 | 2.3866927362920247 |

Both runs:100 responses, all EOS, zero errors/unsent requests;100/100 native
streams identical. The full-pipeline client outstanding cap is three. All
frozen manifests and input/token/configuration checks pass `analyze.py`.

The policy deferred two newcomers for0.510304s and0.246334s. P95 did not
improve. This evidence does not support implementing an image estimator to
drive this particular deferral rule. Do not treat oracle lengths as deployable
inputs or subtract observed pauses to claim a counterfactual result.

Every observed measurement-window NPU6 sample contained only the expected
owned worker (five samples per run). Exact direct-host namespace mappings are
recorded separately in `ownership_pid_mapping.txt`. Monitoring is sampled, not
continuous tracing. Both clients and both servers exited zero. At
2026-09-06T06:26:26+08:00 the owned parents/workers/monitor were absent and
NPU6 was free. No other process or card was affected.

User clarification after the capacity audit: KV4096 and the existing capacity
stop are intentional and remain unchanged. No larger-cache inference ran.
