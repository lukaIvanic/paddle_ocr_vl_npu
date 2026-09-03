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

Token recording implemented; NPU anchor and serving refactor pending.
