# Second-milestone development: ordinary B4, C4 versus C5

One910B2, physical NPU6. Same execution stack as the passing B2 milestone:
packed MLP, complete-layer prefetch, RoPE lookup, native16384 vocabulary,
unchanged KV4096/stopping/pixels, linear patch projection, setup GC freeze,
optional decode-device timing disabled. Only decode batch changes to4.

| Client outstanding cap | Tables/s | P95 seconds | Approx. launched-slot use |
|---|---:|---:|---:|
| C4 (`client`) |4.454861004573734|2.3847790930594783|83.6452%|
| C5 (`c5`) |4.742527029517923|2.3707526877638876|89.7111%|

Both runs finish100/100 EOS, same frozen seed1 manifest/order. All100 native
outputs match between C4 and C5, with unchanged crop/input/vision-token counts.
Neither reaches the second milestone's5 tables/s, so no independent100 or
1000 validation was run for either candidate. This does not undo the separate
B2 first-milestone result.

The C4 record suggested idle-slot cost: many calls had only three active rows.
C5 permits one request to prepare/wait while four decode slots are occupied.
There is no server admission override, batch-filling wait, or knowledge of
future requests. The closed-loop client caps the entire pipeline at5 and
includes that request's waiting time. It improves occupancy, but does not close
the throughput gap. Summed ready-queue wait grows1.2391s→10.1337s across the100
requests; CPU-consumer blocking is only0.0836s→0.0427s. Stage sums overlap and
are never subtracted from end-to-end latency.

## Reproduction

`command.txt` records source `23d5518c`, model/interpreter/device, and the full
server argument array. Launch is that array plus `--freeze-setup-gc
--no-decode-device-timing --service-summary-output <service.json>`.

Client for both: `table_closed_loop_api_client.py --api-url
http://127.0.0.1:8767/v1/ocr --set random --count 100 --shuffle-seed 1
--max-in-flight <4 or 5> --output-dir <client or c5>`. One full warm request
precedes each measured run. Payload preparation remains before timing.

The original client rejected5 before submission (exit2), preserved in
`c5_cli_rejected.log`; its preceding warm request is in `c5_warm`.
Commit `7d7256fc` removes only the artificial allowed-values list, retains
positive-integer validation, and adds C5 response-driven refill tests. All16
client tests pass, including frozen-manifest hashes. The NPU server stays the
same already-loaded process; its inference code is unchanged by this client
commit. A fresh full warm request (`c5_run_warm`) precedes the successful C5 run.

`analyze.py` reconstructs actual outstanding counts, compares configuration,
native IDs, input counts, manifests, stage sums and sampled device ownership.
Slot-use estimates divide each request's launched-iteration count by that
iteration's active-row count; completion look-ahead is included, not useful
output token accounting. Full responses retain both kinds of counters.

Host parent2015278/worker2015280 map to container2547251/2547253, manually
checked from full commands and `/proc/.../status`. Every monitored in-window
NPU6 snapshot has only2015280. Warm/measured clients and the gracefully stopped
server exit0, except the explicitly recorded CLI rejection. Direct `npu-smi`
reported no process before the subsequent B8 experiment. All result records,
including the disappointing C4/C5 throughput, are retained.
