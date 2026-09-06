# Setup GC freeze: identified pause, then real serving gates

Source `23d5518c`, one Ascend910B2, physical NPU6. The unchanged packed-MLP
B2/C2 configuration is the control; this candidate adds only
`--freeze-setup-gc` to the ordinary crop API. Inputs, model, native vocabulary,
greedy policy, KV4096, stopping rules and admission cap are unchanged.

## Causal diagnostic

The preceding lightweight cadence diagnostic is recorded in
`../table_gc_diagnostic_f8f90583_20260906/`. Its generation-2 GC callback spans
441.016745ms inside a441.500144ms graph-submission call. That call consumes
440.66465ms of thread CPU; the existing NPU event interval is441.731445ms.
Thus this particular device-event outlier includes a host GC stall, not just
NPU kernels. This does not explain every earlier timing fluctuation.

The new opt-in collects once after model/graph setup and before any request,
then freezes the persistent setup objects. GC stays enabled. A subprocess CPU
test verifies that request cycles allocated afterward remain collectible.
The measured setup collection took0.4332807529717684s and froze636303 objects
in the uninstrumented server. Frozen objects share the worker lifetime; this
does not cache outputs, visual features or request KV states.

In `diagnostic/`, the same100-request cadence run with callbacks now has no
generation-2 collection. Young collections continue:243 generation0 and22
generation1 collections. Maximum graph-submission wall falls from441.5ms to
4.498486ms. Host/cadence diagnostics are not goal measurements and do not alter
the scheduler or add device events/synchronization. They cannot be subtracted
from request latency.

## Uninstrumented serving gates

The plain `serve_crop_ocr_api.py` command is the saved base argument array plus
`--freeze-setup-gc --service-summary-output <service.json>`.
For each client set, one complete `--set warm --count 1 --max-in-flight 1`
request precedes timing. The same persistent server handles both sets.

```sh
python 09_persistent_page_engine/scripts/table_closed_loop_api_client.py \
  --api-url http://127.0.0.1:8767/v1/ocr \
  --set random --count 100 --shuffle-seed 1 --max-in-flight 2 \
  --output-dir <client>
```

The second gate changes only `--shuffle-seed 2` and the output directory.
Selection uses all665 source tables, sampling without replacement. The original
development manifest retains SHA256
`1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85`.

| Gate | Completed tables/s | P95 wall seconds | Outcome |
|---|---:|---:|---|
| Development100, seed1 | 3.061542938848451 | 1.7778192293655584 | Pass |
| Independent100, seed2 | 2.9165534209864914 | 1.9889938855485514 | Throughput fails |

Both runs finish100/100 EOS, no HTTP failures or unsent requests. The full
pipeline cap is2. The development output is100/100 native-token-identical to
the same packed model without freezing, with identical input/vision-token
counts. **No1000-table run is authorized by these results**, because seed2
misses3.000 tables/s. The change remains opt-in.

## Ownership and subsequent work

Frozen diagnostic: host worker1984437/container2543555, parent1984408/2543553.
Plain server: host worker1988078/container2544045, parent1988076/2544043.
Both were manually checked against `/proc` and NPU6 process listings. The
shared host monitor PID1972167 spans this work and the adjacent cadence tests.
Both servers were stopped gracefully with exit0 and NPU6 was verified empty
before the next experiment. The monitor is retained with the cadence audit.

The next separately recorded experiment uses the existing
`--no-decode-device-timing` switch. This disables optional profiling event
pairs/history only, retaining dependency events and complete request timing.
The hypothesis is that reduced packed-MLP kernel cost may expose host overhead
that was hidden by the older decoder; previous no-event results alone cannot
establish an effect for this new execution stack.
