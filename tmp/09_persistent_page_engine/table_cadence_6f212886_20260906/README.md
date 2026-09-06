# Real serving graph-submission cadence diagnostic

Source `6f212886`, one910B2, NPU6. The existing `profile_api.py` adapter used
`TABLE_SERVING_PROFILE_COMPONENT=cadence`, calling the real crop API. A full
warm request precedes each frozen seed1 random100/C2 run. `command.txt`
contains the API argument arrays. No synthetic prefill or model-only loop.

The wrapper records wall/thread-CPU timestamps around the existing arena
step and decode function. No extra device events or synchronizations are
introduced. The original profiling events are exported after serving stops.
These instrumented runs diagnose cadence; they are not goal-gate scores.

| Diagnostic | Tables/s (perturbed) | P95 s | Largest graph-call wall |
|---|---:|---:|---:|
| Unpacked MLP control | 2.8806 | 1.9346 |439.917986ms|
| Packed MLP | 2.8859 | 1.9329 |444.872491ms|

The corresponding thread CPU is438.83619ms and443.84265ms. Such long host
pauses lie inside an NPU-event interval and cannot be interpreted as pure
kernel duration. Ordinary calls are around292–298us host submission, and
two-active-slot event medians are1178us/1159us. The lightweight diagnostic
therefore motivated GC callbacks, rather than more attention-kernel tuning.
It does not alone prove the cause; see the next GC diagnostic.

Host parent/worker1972428/1972433 map to container2542016/2542018 (control).
Packed parent/worker1975242/1975244 map to2542509/2542511. Full commands and
NPU6 ownership were checked manually. All warm/client/server exits0; both
servers stopped and NPU6 was checked free between launches. The shared
ownership monitor is retained at
`../table_packed_noevents_23d5518c_20260906/host_npu6_monitor.log`.
Use the existing `table_serving_profile_20260905/analyze.py` on each diagnostic
directory to regenerate the cadence analysis.
