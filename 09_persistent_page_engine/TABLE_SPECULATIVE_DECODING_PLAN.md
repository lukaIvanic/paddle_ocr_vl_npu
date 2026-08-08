# Table speculative decoding plan

This file is the durable progress record for the table-latency project. Work on
one phase at a time. Do not start a later phase until the current phase has a
clear result and explicit approval to continue.

## Fixed decisions

- Development inputs are OmniDocBench v1.6 ground-truth table crops.
- The splitter receives only crop pixels. It must not use ground-truth HTML,
  row counts, or internal row annotations.
- Phase 1 first detects natural row boundaries. It does not impose a maximum
  row or band count.
- Initial splitting methods use CPU image processing. A learned detector is not
  part of the first implementation.
- Draft generation finishes before authoritative full-table decoding starts.
- Phase 3 receives already-generated draft token sequences. Draft production
  does no NPU work during target decoding.

## Phase 1 — Natural row splitting and visual validation

Status: **semi-complete**

Develop multiple CPU strategies for ruled, borderless, sparse, dense, merged,
and multi-line tables. Show representative table crops with proposed boundaries
and row crops. Record failure modes. The output is a natural-row description;
later scheduling policy is separate.

Visual review is complete enough to continue. Final validation moves into Phase
2: OCR every proposed row, stitch the outputs, and measure Page-TEDS. Phase 1
remains semi-complete until that experiment shows which splitting strategies
preserve useful table structure.

## Phase 2 — Precomputed row drafts

Status: **complete**

First run a small manually reviewed table set. Then run all OmniDocBench v1.6
tables for every Phase 1 split mode. Run all row/band vision and text prefills,
then generate draft outputs with concurrent decode. Measure per-table and corpus
wall time, input vision tokens, text-prefill tokens, output tokens, malformed or
degenerate outputs, draft-to-full-output similarity, and stitched Page-TEDS.

The full 665-table sweep is complete for all five split modes. Exact target and
draft token IDs are saved. Ruled rows dominate the other split modes in stitched
Page-TEDS, draft cost, and offline speculative acceptance. The optimized ruled
run uses one cross-table B8 schedule and keeps decode slots 98.9% active. See
`TABLE_SPECULATIVE_DECODING_RESULTS.md` for the complete measurements.

Exit gate: draft artifacts are complete, reproducible, and sufficiently useful
to justify target-side speculative verification work.

The exit gate passed. An offline Phase 4 feasibility simulation was also run by
explicit request before Phase 3. It does not replace the Phase 3 runtime gate.

## Phase 3 — Draft-input speculative target runtime

Status: **complete**

Implement a compiled target runtime that accepts precomputed draft tokens and
verifies a fixed token block against the authoritative full-table context.
Draft selection quality is not part of this phase. Validate KV state, rejection,
fallback, graph replay, timing, and baseline greedy-output parity.

Exit gate: the target runtime is correct and measurable with synthetic perfect,
partially correct, and incorrect drafts.

The real B1 D16/KV4096 runtime is implemented. It reuses the authoritative
full-table prefill cache, verifies fixed 16-token blocks, commits only accepted
KV positions logically, and falls back to ordinary B1 decode. Rejected physical
tail slots are masked and overwritten before use. A 16-table live comparison
was token-exact and covered full acceptance, partial rejection, and fallback.
The full 665-table run is 661/665 token exact. The four differences are bounded
PromptFA-versus-IncreFA numerical choices; full Page-TEDS changes by -0.000236.
See `TABLE_SPECULATIVE_RUNTIME_RESULTS.md`.

## Phase 4 — Draft matching and re-anchoring

Status: **first runtime candidate measured**

Develop the candidate index, suffix matching, row-order tracking, re-anchoring,
candidate ranking, and optional multi-candidate verification. Optimize accepted
tokens per verification step and final table latency while preserving the Phase
3 correctness contract.

Exit gate: full 665-table latency and Page-TEDS comparison against the committed
B1 baseline.

The column-aware exact matcher with reversible cursor, virtual width patch, and
table-start prior is integrated. The full comparison is complete. It reduces
target calls by 6.383x and improves P99 from 4.847 s to 3.720 s, but the draft
cost makes aggregate corpus time 4.7% slower. The next work is a legal bypass
for small tables and a confidence gate for pathological weak-anchor cases.
