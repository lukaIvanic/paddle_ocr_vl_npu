# Decode profiling-event removal: no measured serving gain

Source `447176de`, one idle Ascend 910B2 (physical NPU6), same cached
weight-padded-vision B2/C2 configuration as
`table_vision_3b222363_20260905/cache_reload`, plus
`--no-decode-device-timing`. No model, image, native vocabulary, stopping,
cache-capacity, request-set, or scheduling-policy change.

The flag suppresses only the two profiling events per decoder call. Required
copy-stream dependency/completion events remain. Submission-to-response client
timing, all requests, interruption logging, outputs and prefill timing remain.
Unavailable decode device time and its derived rates serialize as JSON null,
not zero or estimates. Defaults remain instrumented.

Frozen development-100 seed1, one complete warm request outside timing:

- 100/100 successful EOS completions; 100/100 native-token parity with the
  cache-loaded vision control.
- Completed tables/s: **2.8578578996415858**.
- P95 complete request latency: **1.9706100761657572 s**.
- Control: 2.8926129443484783 tables/s, P95 1.9431582124438127 s.

This does not support profiling-event removal as a useful optimization. It
does not pass the throughput gate. Neither validation set was run.

The graph was cache-loaded (decode initialization 0.237276 s), and complete
runtime setup took 35.369109 s. Full commands/configuration/results are saved.
Host PID1798348 maps to container worker2480525, owned by the documented API
parent1798346. Approximately five-second direct-host snapshots show no foreign
NPU6 process during measurement; this is sampled monitoring, not continuous
tracing. Both sides of the measured interval are covered. Server exits zero;
host check at 2026-09-06 03:07:52 CST confirms the worker/parent gone and NPU6
free. Owned monitor1798147 was then stopped.
