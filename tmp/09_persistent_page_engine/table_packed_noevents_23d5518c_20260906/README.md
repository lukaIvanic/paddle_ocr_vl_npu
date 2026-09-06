# Ordinary B2/C2: first serving milestone

One Ascend 910B2, physical NPU6. Runtime source `23d5518c`.
These are real HTTP request wall times, not composed stage estimates.

| Gate | Request completions/s | EOS completions/s | P95 seconds |
|---|---:|---:|---:|
| Development100, seed1 | 3.1716184673601306 | 3.1716184673601306 | 1.7593515512591686 |
| Independent100, seed2 | 3.0203698080505923 | 3.0203698080505923 | 1.9289231458795246 |
| Validation1000 A, seed3 | 3.310780705472748 | 3.271051337007075 | 1.9443893248273525 |
| Validation1000 B, seed3 | 3.3119760036561687 | 3.272232291612295 | 1.9410503454506394 |

Both100 gates finish100/100 EOS. Each1000 run has988 EOS and12 KV4096-cap
stops, with no HTTP errors or unsent requests. These are exactly the historical
B2 run's cap occurrences: capacity and stopping policy were NOT changed.
Capped requests remain in all latency statistics, and their responses are
not represented as EOS/full-quality OCR. Even EOS-only throughput exceeds3.
`audit.json` reports every cap occurrence and its actual latency.

## Execution and timing contract

The ordinary crop API uses the packed-MLP preset with complete-layer-ahead
prefetch and RoPE lookup, the unchanged frequency-selected16384-row native-ID
head, KV4096, greedy argmax and the original4096 output cap. The full argument
array is in `command.txt`. The launch expands that array as:

```sh
"$table_py" -u "${table_mlp_common[@]}" \
  --freeze-setup-gc --no-decode-device-timing \
  --service-summary-output "$table_noevents_dir/service.json"
```

The setup-GC change and its causal diagnostic are recorded in
`../table_gc_freeze_23d5518c_20260906/`. The new comparison changes only the
existing optional decode-device-timing switch: dependency events, token D2H,
prefill timing, all HTTP/request timestamps and scheduling counters remain.
It removes per-decode profiling event pairs and retained event history, not
model work. Disabling measurement overhead is not subtracting it afterward.

For each set, one full warm request (`--set warm --count 1 --max-in-flight 1`)
precedes measurement. One persistent server handles all four gates. Client:

```sh
python 09_persistent_page_engine/scripts/table_closed_loop_api_client.py \
  --api-url http://127.0.0.1:8767/v1/ocr \
  --set random --count 100 --shuffle-seed 1 --max-in-flight 2 \
  --output-dir <client>
```

The independent gate changes seed to2. Both validation runs use count1000,
seed3, C2. Sample manifests are frozen before tuning. They contain all665 in
shuffled order followed by335 distinct tables sampled from the fresh cycle.
Payload PNG construction is before timing, consistently with crop-in-RAM
semantics; image decode, CPU preparation, NPU work, queue/control and response
handling are timed. The client admits a replacement only after one response.
There is no batch-filling wait, output reuse, or table-specific routing.

## Audit and output differences

Run `python3 <this-directory>/audit.py`. It checks manifests/order/counts,
the development SHA256, reconstructed outstanding-request cap2, identical
recorded configuration, request percentiles and throughput, and sampled NPU
ownership. `output_comparison.json` retains exact text edit contexts alongside
native-token comparisons. No generated text was re-encoded.

The two1000 repeats match native IDs1000/1000. Relative to historical B2
(`table_1000_matrix_02fe5645_20260905/b2/measured`),985/1000 match; the15
different occurrences cover10 unique tables. Input counts, crop sizes, real
vision-token counts and stop reasons match all1000.

Direct inspection found the following differences; these are not dismissed
as harmless formatting:

| Table suffix | Observed change relative to historical B2 |
|---|---|
| 001417 / 2 | Chinese `初转`→`初赛`, `电源`→`电票`; the latter adds a character error against GT. |
| 000766 / 2 | One space after a closing math delimiter. |
| 000291 / box_id_0 | `15349`→`1539`; one extra trailing digit fits before the unchanged cache stop. |
| 001368 / box_id_263 | `So, Illinois`→`So. Illinois`, matching the GT punctuation. |
| 001375 / 4 | Nine minus signs disappear from temperature cells. This is a real numeric regression, not a LaTeX equivalence. |
| 000281 / box_id_1 | An extra closing parenthesis is generated after a formula. |
| 001138 / 1 | Email loses `f`: `dxsyf`→`dxsy`, a content regression against GT. |
| 001227 / 10 | DNA sequence loses one `C`; previously localized to linear patch execution. |
| 000918 / 0 | Adds Chinese `及` before `产地`. |
| 001398 / box-7mwax454 | Fifteen cell delimiters become dot leaders, merging label/page-reference content into a cell. Real table-structure divergence; GT has two columns. |

The source changes are mathematically equivalent projection/activation
implementations, not a quality-for-speed policy: same weights, pixels,
vocabulary and greedy/stopping semantics. The exact packed projection and
prefetch-storage CPU tests are retained with the implementation. Numerical
drift can cause autoregressive content/structure changes; this evidence does
not claim byte parity or unchanged TEDS. It also does not prove which changed
kernel caused each of these10 tables individually. GC/event changes alone
match the corresponding packed control100/100 on both development sets.

## Deployment and hardware

This winning path is `serve_crop_ocr_api.py`, not the experimental speculative
endpoint. The ordinary worker consumes image bytes and crop type; source IDs
are bookkeeping. It has no saved-target/orientation/GT lookup. The source audit
is in `../table_1000_matrix_02fe5645_20260905/serving_metadata_audit.md`.
The speculative endpoint's metadata dependencies remain unqualified and are
not used to achieve these results.

Host parent1994480 / worker1994482 mapped to container2544641 /2544643 via
`/proc/.../status`; full command and physical ownership were checked manually.
`host_npu6_monitor.log` spans all four measured windows, with only1994482 in
every in-window device snapshot. Monitoring is sampled, not a hardware-level
proof against arbitrarily brief outside activity. The parent was gracefully
terminated after validation B; server exit0, service summary saved, and direct
host `npu-smi` reported no process on NPU6 before any subsequent experiment.
Every warmup/measured client returned exit0.

Both first-milestone numerical targets pass the two100 and two1000 gates.
The overall goal remains active: its next milestone is5 tables/s with P95<3s,
with the same development/independent/final validation ladder.
