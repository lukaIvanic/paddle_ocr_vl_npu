# Final clean serving validation — one 910B2, physical NPU6

Both goal milestones pass their two100 development gates and two clean1000
validation runs. These are **closed-loop HTTP submission-to-response wall
latencies**, not stage-time estimates or offered-arrival QPS claims.

| Configuration / final1000 run | Response completions/s | EOS completions/s | P95 seconds |
|---|---:|---:|---:|
| Ordinary B2/C2, A |3.310780705472748|3.271051337007075|1.9443893248273525|
| Ordinary B2/C2, B |3.3119760036561687|3.272232291612295|1.9410503454506394|
| Ordinary B5/C5, A |5.605123315729733|5.537861835940976|2.884945039480224|
| Ordinary B5/C5, clean replacement B |5.821290025805274|5.75143454549561|2.8129160678014133|

B2 meets >=3/s and P95<2s. B5 meets >=5/s and P95<3s. The100 gates and
unrounded scores are retained in `final_audit.json` and the earlier run roots:

- `../table_packed_noevents_23d5518c_20260906/`: all four B2 gates.
- `../table_packed_noevents_b5c5_b0f1cac7_20260906/`: both B5 development100
  gates and valid validation A. Its original validation B is **invalid** due
  to overlapping external NPU work and remains saved, not replaced or hidden.
- This directory: only the missing clean replacement B. No retuning or new
  performance margin gate was introduced after the independent100 gate.

## What ran

The plain `serve_crop_ocr_api.py`, exact decode B5, client outstanding cap5.
Same optimized ordinary model path as the earlier B5 candidate: packed gate/up
MLP and native SwiGLU, complete-layer-ahead prefetch of executed weights,
RoPE lookup, equivalent linear patch projection, setup-GC freeze, and optional
decode profiling events disabled. Dependency events and request timing remain.
No speculative decoding, routing, token proposals, oracle or corpus-specific
heuristic is used. The rejected dataset-derived proposal idea is not included.

`command.txt` records interpreter, complete arguments, physical device6,
hostname and source commit `d958f186`. Runtime code is unchanged from the
original B5 source `b0f1cac7`; intervening commits only preserve reports/audits.
`client_commands.txt` records the full-request warmup and measured client.
The server used the same compiled graph/cache keys and inference settings.
Only compile/setup wall measurements and GC object counts differ between
process launches; `final_audit.py` excludes precisely those fields when checking
the full inference contract, not arbitrary fields or kernel settings.

One full warm request ran outside measurement. All665 crop payloads were
prepared in RAM before timing. Measured execution lasted171.78322941600345s;
no per-table server preparation was moved outside request timing. Requests
arrive independently and refill after individual responses, with at most5
outstanding over the entire pipeline. The run is not a synchronized cohort.

The frozen1000 seed3 manifest is SHA256
`fcf572b443303fb449913a12d58989eaf59b5e563e45fa6dac6629e194b7fa62`:
all665 in shuffled order, then335 distinct from a fresh shuffled cycle.
The audit reconstructs membership/order, admission count and percentiles from
individual records. Every result was flushed as received, including the caps.

## Stops, inputs and outputs

Each final1000 run has988 EOS responses and12 historical KV4096-cap stops.
No HTTP errors, dropped submissions or changed output limits. Capped outputs
are not full OCR successes: all1000 latencies remain in P95, but EOS-only QPS
uses988 as the success numerator over the entire run time. Both milestones
also pass that conservative throughput measure. We did not enlarge KV or
change stopping policy to make the results look better.

The clean B5 repeat matches the first valid B5 run **1000/1000 native IDs**,
including stop reasons, crop dimensions, input/projected tokens and real vision
tokens. `replacement_output_comparison.json` contains the direct comparison;
no generated text was re-encoded. The previously inspected numerical changes
against older B2 paths remain documented in their READMEs/output comparisons,
including real content/structure regressions. Repeat parity is NOT a claim of
unchanged quality against the historical baseline or a new TEDS measurement.

## Ownership and cleanup

At17:46 CST NPU6 was free and external suite parent2056282 and its children
were gone. We manually verified host parent3228766/worker3228768 mapping to
container2555499/2555501, full command and `ASCEND_RT_VISIBLE_DEVICES=6`.
No other card was used. `host_npu6_monitor.log` brackets the measured window
and contains69 in-window samples, all with only worker3228768. There are no
unexpected ownership snapshots. This is sampled observation, not a claim
to detect an arbitrarily short external kernel between samples.

The owned server exited0 and saved exactly1001 service requests (1000 measured
plus1 warmup). The owned monitor3227129 was then stopped. Direct host checks
show the server, worker and monitor gone and NPU6 with no process. See
`ownership_before.log`, `ownership_running.log`, `ownership_released.log`
and `final_cleanup.log`. No other user's process was signalled or modified.

## Reproduce the CPU evidence audit

From repository root:

```sh
python3 tmp/09_persistent_page_engine/table_b5_clean_repeat_d958f186_20260906/final_audit.py
```

This reuses the existing audit for all original and replacement gates,
preserves the contaminated attempt, compares configurations and outputs,
checks chronological gate order, warmup/exit codes and shutdown evidence,
and writes `final_audit.json`. It does not run inference or change routing.

Fresh local CPU checks (not NPU validation) also passed and are saved:
16 client/admission/manifest tests;5 GC/service-summary tests;2 packed-MLP
algebra/prefetch-storage tests;2 linear-patch algebra/geometry tests.

## Requirement-by-requirement completion audit

| Requirement | Current evidence |
|---|---|
| Both targets and all prescribed gates | `final_audit.json`; original and replacement results/manifests; both100 gates precede each pair of qualifying1000 runs. |
| Frozen images/resampling/vision tokens | Identical source crop manifest and client PNG construction; frozen min/max pixels28224/802816 and processor configuration; all1000 response input/crop/vision counts match. Linear patch code changes only the equivalent projection after pixel preparation. |
| Same checkpoint, vocabulary, greedy, KV and stops | Full configuration comparison including model/cache/source hashes and selected native-ID map `9c48e5c3b92776ba250f75359fccb407448c4da8419fe927f5ea381d345712c3`; ordinary argmax, KV4096/output4096;12 unchanged caps. |
| No hidden quality-for-speed change | Mathematically equivalent packed-MLP/linear-patch code and CPU tests; numerical changes directly inspected in earlier reports; no suppression, early stop or lossy input change. |
| Deployable request-time execution, no ID/GT dependence | Ordinary worker constructs requests from supplied image bytes and prompt. `_prepare_cpu` resolves each image and runs real preprocessing; source IDs are bookkeeping. No orientation lookup or oracle is imported by this route. The benchmark-dependent historical speculative endpoint is not a qualifying/deployed solution. |
| No reuse of prior outputs/features/KV | Open source submits `_prepare_cpu` for each request, stages real vision/text prefill and leases private KV; hot-swap copies that request's prefix and resets control. Only model/graph/storage resources and pre-timed input payload bytes are reused. |
| Independent serving; full-pipeline cap | Client response-driven worker loop and16 tests; reconstructed peak outstanding2/5. Server emits each completion and does not wait to fill a cohort. |
| Faithful timing; no exclusions | Raw dispatch/completion offsets reproduce each latency and P95; every submission retained; prep/prefill/decode/control/postprocessing and waiting are inside HTTP wall time. Warmup and payload loading are explicitly separate. |
| NPU6 only, no overlap | Manual PID/namespace/environment evidence and bracketed ownership logs for every qualifying run. Contaminated attempt explicitly fails audit. |
| No dataset-specific speed hack | Same general runtime operations for all requests; no output-length routing, token-pattern proposal, ID exception or future-arrival knowledge. The pre-existing native vocabulary map is frozen, not retuned. |
| Local authoring, main, remote pull-only | Implementation and commands committed on main; remote tracked runtime unchanged during this validation. This continuation adds only evidence and CPU audit files. |
| Preserve results, stop owned work | Invalid and regressing attempts retained in earlier roots; all new artifacts saved; server exit0, monitor stopped, direct-host NPU release verified. |

The source audit includes `serve_crop_ocr_api.py` (`_worker_main`,
`QueueRecognitionSource`, `emit_result`), `table_closed_loop_api_client.py`
(`run_closed_loop`, payload loading), `engine.py` (`_OpenPrefillSource`,
`_prepare_cpu`) and `continuous_decode.py` (refill/hot-swap/completion).
The ordinary deployment path bypasses the known experimental speculative
metadata dependency documented in
`../table_1000_matrix_02fe5645_20260905/serving_metadata_audit.md`.
