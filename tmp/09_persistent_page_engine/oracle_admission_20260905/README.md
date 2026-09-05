# Diagnostic oracle prefill protection: one request rescued, P95 unchanged

This adapter is **not a deployable policy or a qualifying goal result**. It
looks up saved native output lengths by benchmark ID, explicitly forbidden in
the qualifying serving path. It only answers a scheduling question around the
unchanged ordinary crop API. No speculative decoding is used.

## Hypothesis fixed before measurement

Use ordinary B4 decode with client C3, same frozen random100 seed1. The
newcomer can be CPU-prepared in the background and still counts toward C3.
Before its NPU prefill, defer only if:

- Its saved length is below256 native tokens, including EOS.
- A running request's remaining saved length is positive.
- The running request would finish below2s without the prefill, but at/above2s
  if one more prefill is inserted.
- Waiting for that remaining decode still leaves the newcomer below2s.

The cost model is deliberately simple:1.35ms per remaining token and100ms for
prefill. These constants were selected before testing, from the recent B4
cadence (~1.313ms device average) and conservative prefill cost. They are not
per-table recorded latency lookups. Reconsider every iteration; missing or
exhausted length estimates fall back to immediate prefill. This estimates wall
time, rather than being a perfect wall-time oracle. It does not protect
requests whose estimated finish is already beyond2s.

Source `37e42bc0`, Ascend910B2 physical NPU6. Setup-only compilation uses the
actual API, serves no requests, and exits before measurement. Control and
policy each use a separate cache-loaded server, one complete warm request
outside timing, then100 requests at C3. Same model/cache/vision settings in
both; loop-boundary prefetch remains disabled. All timing includes queueing,
CPU preparation, prefill, decode, policy waiting and response delivery.

## Result

| Metric | Ungated control | Oracle protection |
|---|---:|---:|
| Completed tables/s | 3.5950020166487704 | 3.586862129228667 |
| P95 seconds | 2.326323885156307 | 2.3355806675856 |
| Mean seconds | 0.8009133621118962 | 0.8025004016479943 |
| Requests over2s | 7 | 6 |

One newcomer was deferred for0.413930s:

| Request | Control seconds | Policy seconds |
|---|---:|---:|
| Protected page_000261_table_box_id_3 | 2.0560 | 1.9539 |
| Waiting page_000789_table_34 | 0.2222 | 0.7051 |

The protected request experiences three during-decode prefill interruptions,
down from five. The newcomer remains well below2s. All100 requests complete
with EOS;100/100 native token streams are identical. No failed or unsent
requests. The client peak is exactly three, including the waiting request.

This is a useful localized success, not an aggregate P95 win. The next
threshold-setting request, page_000626_table_box_id_1, is2.31984s in control
and2.3292s with the policy, so P95 stays high. Its first decode starts at
0.210697s, and it has1352 native output tokens. At the fixed1.35ms/token cost,
its projected finish is already over2s at decode start. Therefore the policy
deliberately fails to protect it. This exposes sensitivity to the remaining-
time model; it is not proof that no scheduling policy could help.

Do not subtract its0.278081s of recorded host prefill pauses from latency to
claim a counterfactual win. Those spans can overlap an in-flight graph, and
deferring one request changes later arrivals and contention.

## Evidence and cleanup

`run_37e42bc0/` keeps exact commands/configuration, setup/control/policy logs,
native outputs, client manifests, service summaries and ownership monitoring.
`analyze.py` checks the frozen manifest, all100 responses, full client C3 cap,
native parity, real vision-token equality, model cache identity and sampled
NPU ownership. `test_policy.py` exercises positive protection, exhausted or
missing estimates, insufficient newcomer slack and idle initial filling.

Setup host1833421/container2491707; control host1835350/container2492533;
policy host1838207/container2493704. All were owned children of the respective
API parents. Direct-host five-second samples showed only the expected worker
during each run. This is sampled monitoring, not continuous tracing. At
2026-09-06 04:35:29 CST NPU6 was free and the policy worker/parent were gone.
All servers exited zero; owned monitor1833136 was stopped. No other NPU or
user's process was used or affected. Neither validation gate was run.
