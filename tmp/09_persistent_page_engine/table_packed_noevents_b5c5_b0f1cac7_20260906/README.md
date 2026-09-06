# Exact B5/C5 ordinary serving: second-milestone candidate

One910B2, physical NPU6. Runtime/client source `b0f1cac7` (later commits during
the run save evidence/audits only). No speculative decoding or routing. The
candidate differs from the passing B2 stack only in static decode batch5 and
closed-loop client cap5. The engine already supports exact positive batches;
there is no padding to B8, new scheduler, corpus-derived rule or model change.

The direction comes from real B4/B8 profiles, not a blind batch sweep:
`../table_b4_b8_profile_b0f1cac7_20260906/` shows B8 adds~170us IncreFA and~44us
KV scatter per graph relative to B4, while matmuls grow only~7us. B4/C5 left
one admitted request waiting for a decode slot; B8/C5 paid for three idle rows.
Exact B5 avoids that tradeoff. Its actual serving result, not this reasoning,
determines whether it passes.

## Frozen gates (second attempt contaminated; clean replacement required)

| Gate | All response completions/s | EOS completions/s | P95, all submitted requests |
|---|---:|---:|---:|
| Development100, seed1 |5.01253755342034|5.01253755342034|2.592431990162004s|
| Independent100, seed2 |5.0177570969864185|5.0177570969864185|2.993809562240494s|
| Validation1000 A, seed3 |5.605123315729733|5.537861835940976|2.884945039480224s|
| Validation1000 B, seed3 — **INVALID: shared NPU** |5.415152604136896|5.350170772887253|2.9913312447781175s|

Both100 sets finish100/100 EOS. Validation A finishes988 EOS and12 KV-cap
responses, exactly the historical cap occurrences with prompt+generated-1
equal4096. No capacity/output limit/stop-rule change was made. Capped outputs
are not called EOS or full OCR successes; all1000 latencies remain in P95 and
the full run remains the throughput denominator. EOS-only throughput is the
conservative success numerator for the goal. No HTTP errors/unsent requests.
The second attempt is retained but cannot count toward the goal, despite its
numerical scores. Its measured window is2026-09-06 09:23:20.681914 through
09:26:25.348917 CST. The monitor catches non-owned PID2057079 at09:25:58 and
PID2059772 at09:26:16 and09:26:22 alongside our worker2037247. Another user's
`runtime_test.run_b2 --device 6` suite parent2056282 started at09:25:29; its
later child2069506 was first noticed manually after our client had finished.
The parent/start-time and saved in-window snapshots establish actual overlap,
not just a busy after-run snapshot. Maximum client latency grows to10.5648s.
Neither the apparent pass nor the changed maximum may be used as a clean
performance result. A clean replacement of validation B remains required.

## Exact serving command

Run only after manually verifying physical NPU6 is free; keep ownership
monitoring during measurement. Initialize the known container environment with
`source npu-setup`, then set `ASCEND_RT_VISIBLE_DEVICES=6` per the goal. Use:

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python -u \
  09_persistent_page_engine/scripts/serve_crop_ocr_api.py \
  --host 127.0.0.1 --port 8767 --request-timeout-s 3600 --queue-capacity 64 \
  --model /workspace/models/PaddleOCR-VL-1.6 --decode-backend torchair \
  --decode-optimization combined_apply_complete_layer_prefetch1_rope_lut_packed_mlp \
  --decode-vocab-token-ids 09_persistent_page_engine/presets/table_compact_vocab/b1_verifier_topfreq_16384.json \
  --token-selection greedy --cache-length 4096 --max-new-tokens 4096 \
  --min-pixels 28224 --max-pixels 802816 --request-scheduling-metrics \
  --decode-batch-size 5 \
  --vision-buckets 256,384,512,640,768,1408,1920,2048,2944,4096 \
  --text-buckets 128,256,512,1024,1152 \
  --vision-attention-weight-padding --vision-linear-patch-projection \
  --freeze-setup-gc --no-decode-device-timing \
  --service-summary-output <run-directory>/service.json
```

This uses the frozen selected native-ID map, not an arbitrary vocabulary
prefix. Pixels, resampling policy, real vision tokens, checkpoint, FP16,
greedy semantics and cache/stop policy match the established ordinary route.
Weight packing/linear patch computation are mathematically equivalent paths;
GC freezing and optional timing-event removal do not change model computation.
All mandatory dependency events and request timings remain.

After READY, warm with one complete request using the same client, `--set warm
--count 1 --max-in-flight 1`, outside measurement. Each subsequent gate gets
the same one-request warmup and separate output directory. Client:

```sh
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python -u \
  09_persistent_page_engine/scripts/table_closed_loop_api_client.py \
  --api-url http://127.0.0.1:8767/v1/ocr \
  --set random --count 100 --shuffle-seed 1 --max-in-flight 5 \
  --output-dir <run-directory>/client
```

Independent set changes seed to2. Both final runs use count1000, seed3.
Manifests were frozen before tuning: development SHA256
`1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85`,
seed2 `b6fb05f880146e500d11091ac204879a62027b574682c170fcbb7946f161b502`,
final `fcf572b443303fb449913a12d58989eaf59b5e563e45fa6dac6629e194b7fa62`.
The1000 sequence contains all665 once in shuffled order, then335 distinct
samples from the fresh cycle. Same candidate/server/configuration across gates.
No retiming, exclusions, interruptions subtracted or extra margin gate.

Payload preparation precedes timing under the agreed crop-in-RAM client
contract. Server image decode, CPU prep, all NPU work, queueing/control and
response completion remain inside request wall latency. Responses emit
independently, and each permits one replacement. The cap covers all outstanding
requests, including CPU prep/waiting. No future requests, output/image-feature
or prior-request KV cache is reused. Ordinary source IDs are bookkeeping only;
the benchmark-dependent speculative endpoint is not used.

## Output inspection

Both100 sets match the corresponding optimized B2 native IDs100/100, with
unchanged crop sizes and input/real-vision-token counts. Validation A matches
optimized B2 at996/1000. Direct raw-text/native-ID inspection of the differences:

- `page_001215_table_5`: `I_{41}/amd`→`I4_{1}/amd` and `I_{4}/mmm`→`I4/mmm`.
  The new symbolic placement matches the saved GT cells; not merely whitespace.
- `page_000766_table_2`: removes one space after the closing math delimiter.
- `page_001035_table_28`: changes the count of dot-leader characters; the numeric
  cells remain unchanged.
- `page_001227_table_10`: restores a `C` in the DNA sequence previously lost by
  the B2 linear-patch path. This is a real content difference/improvement.

Against the older pre-packed/pre-linear B2 run,985/1000 match. Its remaining
differences and regressions are NOT dismissed: see the detailed edit contexts
in `output_comparison.json` and the first milestone's inspection table. The
candidate does not claim byte parity with the historical model path or an
unchanged TEDS score. Only equivalent-execution numerical drift is permitted;
no quality-changing policy was introduced to obtain speed.
The two B5 validation attempts match native IDs1000/1000 despite the second
attempt's timing contamination; this does not rehabilitate its latency scores.

## Evidence audit and ownership

Use the existing shared audit, rather than a new benchmark script:

```sh
python3 tmp/09_persistent_page_engine/table_packed_noevents_23d5518c_20260906/audit.py \
  --root tmp/09_persistent_page_engine/table_packed_noevents_b5c5_b0f1cac7_20260906 \
  --batch-size 5 --max-in-flight 5 --worker-pid 2037247 \
  --target-qps 5 --target-p95 3
```

It checks frozen manifests/order/counts, reconstructs the outstanding cap,
compares full recorded configuration across gates, recomputes latency/QPS,
checks native EOS and exact cache exhaustion, and audits sampled ownership.
Post-run comparisons are not used by the inference engine.

Host parent2037245 / worker2037247 map to container2551493 /2551495. Full
command, NSpid mapping and physical NPU6 were checked manually. Shared host
monitor2028986 spans the profiles and this run. Development/seed2/validation A
have only our worker in every in-window snapshot; validation B does not.
`audit.json` explicitly sets `qualifying_timing=false` for that second attempt
and lists its unexpected ownership snapshots. None of its requests are deleted.

All4 warm requests and4 measured clients exit0. The owned server was stopped
gracefully after validation B and exited0. Its final service summary contains
exactly2204 requests (2200 measured plus4 full warmups), with the frozen
16384-row vocabulary map SHA256
`9c48e5c3b92776ba250f75359fccb407448c4da8419fe927f5ea381d345712c3`.
The host parent and worker PIDs are gone and NPU6 no longer has our process.
It still has the other user's test job. Only the read-only owned monitor is
left running while awaiting a safe replacement validation; no other process
has been signalled, stopped or modified.
