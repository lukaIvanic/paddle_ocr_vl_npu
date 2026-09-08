# INVALID TIMING: 8192-budget Poisson control

This run must not be used for a Pareto point or optimization comparison.
Other NPU processes overlapped the measured window on physical NPU6.

All 100 requests completed with zero errors and no length stops; the exact
saved table/arrival sequence matches the 4096 control. 99/100 native output
token streams match that control, and both runs generated 36,429 tokens.
This does not rescue contaminated performance measurements.

Our engine was host PID 1666113 (container EngineCore PID 4297); API container
PID 4068. Direct-host monitoring initially showed NPU6 idle and then only our
engine. During measurement it recorded additional processes:

- PID 1696007, `python3 -m runtime_test.run`, started 18:20:07 CST,
  observed on NPU6 from the 18:20:31 sample onward, 111 MB.
- PID 1702267 at 18:20:53, 147 MB.
- PID 1706744 at 18:21:14, 109 MB.
- PID 1711854 at 18:21:35, 637 MB.

The measured window overlapped those timestamps. Several additional processes
were transient and had exited by the final manual PID check. Their exact jobs
were not established; they were not started by this benchmark.

Raw, invalid performance observations retained for audit only:

| Metric | Invalid observation |
|---|---:|
| Requests | 100 |
| Mean latency | 3.765838 s |
| P95 latency | 11.040492 s |
| Maximum latency | 24.705364 s |
| Completed throughput | 0.915464 tables/s |
| Wall time | 109.234178 s |
| Preemptions | 0 |

The configured change was solely max_num_batched_tokens=8192 rather than
4096. Per-request context remained 4096, maximum sequences 16, async and
chunked prefill on, caches disabled, FULL_AND_PIECEWISE, FP16. See command.txt.

Our API/engine was stopped after collection; other users' processes were not
signaled. NPU6 remained occupied by their runtime test. No follow-on inference
was started. A clean rerun is required before interpreting this configuration.
