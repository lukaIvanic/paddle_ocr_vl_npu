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

Trace-only reference commit: `13061fc4`. The new source and scheduler have CPU
coverage for vision-window refill, mixed layout/recognition arrivals, immediate
EOS, length caps, idle live input, empty pages, page-window bounds and writer
errors. 910B comparison is pending.

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
`--streaming-page-window 32`. `MODE=streaming LIMIT=384 bash
11_mineru_2_5_pro_inference/run_serving_validation.sh` records a new run on the
same free physical NPU4 and uses the existing graph-cache directories. Set
`MODE=stepping` only for the legacy orchestration with repaired refill logic.

Decode metrics include iteration-weighted and device-time-weighted active-slot
fractions, occupancy histograms, maximum live request states, empty rows despite
ready work, and final drain. Device events are resolved every 1024 iterations,
so an open service does not retain an unbounded event list.
