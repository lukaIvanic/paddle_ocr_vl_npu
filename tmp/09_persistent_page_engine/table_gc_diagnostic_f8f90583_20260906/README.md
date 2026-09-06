# A generation-2 GC pause inside real graph submission

Source `f8f90583`, packed-MLP ordinary B2/C2, one910B2/NPU6. This extends the
preceding cadence diagnostic with `gc.callbacks` only. GC policy is unchanged.
The original real API processes seed1 random100 after one complete warm request.

Iteration20113 has441.500144ms graph-call wall,440.66465ms thread CPU and
441.731445ms device-event interval. Generation2 collection overlaps it from
monotonic827727460789353ns through827727901806098ns: **441.016745ms**,
440.1882ms thread CPU,10 objects collected,0 uncollectable. The callback data
and per-call timestamp rows provide direct causal evidence for this outlier.

There are246 generation0 collections,23 generation1, and one generation2.
Other3–4ms host-call outliers have low thread CPU and no GC overlap. They are
not explained by this finding. This report does not attribute every earlier
timing regression to garbage collection.

The diagnostic completes100/100 EOS with2.9155 tables/s, P95~1.8833s; these
instrumented numbers are not a qualifying throughput run. The next experiment
freezes long-lived setup objects while leaving request garbage collection on.

Host parent1979526/worker1979528 map to container2543032/2543034. Ownership
was checked directly; only that worker occupied NPU6. The shared monitor is
retained beside `table_packed_noevents_23d5518c_20260906`. Warm/client/server
exit0; the owned parent was terminated gracefully and NPU6 verified free.
