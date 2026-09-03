# MinerU serving refactor

## Validation sequence

1. Record lossless layout and recognition token IDs from the unchanged B32,
   KV4096, vision-lookahead-32, CPU-prefetch-64 pipeline on the first 384 pages.
   Keep the historical refill behavior in this anchor commit.
2. Replace finite batch ownership with a bounded page/request source, per-request
   completion and per-page collection. Keep model kernels and input policy fixed.
3. Test refill-window boundaries, immediate EOS, temporary starvation, EOF,
   duplicate identities, output ordering, and producer/writer error propagation.
4. Validate on 910B2 NPU4, compare tokens and page outputs, and report occupancy,
   pending-work holes, final drain and end-to-end wall separately.

The legacy stepping runner remains available for reproduction. No 310P result
is implied. The baseline stores unfiltered generated IDs including EOS, CPU
prompt IDs, raw text, crop-image hashes, page/block identity and geometry,
generation caps, and model/tokenizer/dataset hashes. Retokenized Markdown is
not the token reference.

## Boundaries

One inference thread owns the shared MinerU model, prefill and decode arena.
Layout completion creates recognition requests, because both stages use the
same autoregressive model. CPU preparation may run ahead in a bounded worker
queue. A temporarily empty request source is not EOF. A page can complete and
be written without waiting for unrelated pages or the complete input corpus.

Retain official MinerU layout parsing, crop policy, prompt routing and
postprocessing. Cross-page table merge is disabled in this benchmark. The
serving refactor must reject unsupported enabled cross-page merge explicitly.

## Status

Trace-only reference commit: `13061fc4`. Streaming runtime validated at
`cb8e36ab` on physical 910B2 NPU4. The suite now has 23 tests covering
vision-window refill, mixed layout/recognition arrivals, immediate EOS, length
caps, idle live input, empty pages, page-window bounds, writer errors, CLI
defaults and comparison gates. CPU tests also passed in the validation host's
Torch environment.

The first-384 streaming run completed in 363.509 s, or 1.05637 pages/s, including
token recording and the final page-writer drain. Setup and warmup are excluded.
The earlier untraced run took 540.304 s, or 0.71071 pages/s. The trace-only anchor
took 551.485 s, or 0.69630 pages/s. This is 48.6% higher throughput than the
earlier untraced run, or 51.7% higher than the trace-only anchor. These are single
runs on a shared host, not an isolated attribution of each refactor component.

Average decode-slot occupancy increased from 25.72% to 96.57%; device-time-
weighted occupancy is 96.66%. Decode device time fell from 227.081 s in the
anchor to 61.833 s. Empty slots despite prepared work: zero. Maximum live pages:
32. Maximum CPU preparation queue: 64. Maximum live generation requests: 63.
The first page was durable after 13.057 s. Final drain was 1.774 s.

All 5,486 request identities are present exactly once. All 384 layout sequences
are token-exact; 5,099 of 5,102 recognition sequences are token-exact. All final
Markdown files except one are byte-identical. That page changes the Chinese
character `度` to `座` in two occurrences of the same short text region. This is
an output difference, not a claim that either reading is more accurate.

The other two raw-sequence differences are random table-image labels generated
by the installed helper. They change two crop-image hashes. After bijective
placeholder normalization their raw table text is exact, and both final
Markdown and block JSON are byte-identical. The comparison requires explicit
`--allow-table-image-placeholders` for this narrow case; other input changes
still fail. See the archived result's `RESULTS.md` for source provenance.

Both runs contain the same 18 length-capped recognition requests with identical
output IDs. There are no new caps, missing pages, empty pages or lost requests.
This validates preservation of the existing output contract, not a new
OmniDocBench accuracy score or full-1651 benchmark.

Frozen evidence: `references/serving_anchor_384_13061fc4/` and
`references/serving_streaming_384_cb8e36ab/`.

The subsequent full-1,651 run completed at `ae4c947c`: 0.81331 hot pages/s,
99.77% decode-slot occupancy and 95.1131 overall OmniDocBench v1.6 accuracy.
All 384 prefix Markdown files match the earlier streaming run byte-for-byte.
See [the full result](references/serving_streaming_1651_ae4c947c/RESULTS.md) for
the KV4096 cap audit and checksum-protected predictions, tokens and scores.

## Serving API

`PageInbox` accepts live submissions with bounded backpressure. Its empty state
does not close the stream; `close_input()` ends submission and drains outstanding
pages. One model-owning caller runs `run_decode_stream(engine, page_source)`.
The CPU producer may submit pages while this call is running. All NPU operations
stay on the caller thread. A finite iterable is also accepted for benchmarks.

```python
inbox = PageInbox(capacity=32)
source = MinerUPageSource(client, inbox, on_page=writer.submit,
                         page_window=32, prepare_depth=64)
# A CPU ingress thread calls inbox.submit(page_id, image_loader), then
# inbox.close_input() when the service should drain and stop.
try:
    metrics = run_decode_stream(engine, source)
finally:
    source.close()
    writer.close()
```

The model, compiled kernels and decode arena can be reused for subsequent
streams. No HTTP transport is introduced. The source owns page collection and
preserves official reading order within each result. The single bounded writer
persists pages in completion-submission order, keyed by the original page name.

The benchmark entrypoint adds `--streaming-pages` and
`--streaming-page-window 32`. Streaming is now the default for the local
continuous backend; `--no-streaming-pages` selects the legacy orchestration.
`MODE=streaming LIMIT=384 bash
11_mineru_2_5_pro_inference/run_serving_validation.sh` records a new run on the
same free physical NPU4 and uses the existing graph-cache directories. Set
`MODE=stepping` only for the legacy orchestration with repaired refill logic.

Decode metrics include iteration-weighted and device-time-weighted active-slot
fractions, occupancy histograms, maximum live request states, empty rows despite
ready work, and final drain. Device events are resolved every 1024 iterations,
so an open service does not retain an unbounded event list.
