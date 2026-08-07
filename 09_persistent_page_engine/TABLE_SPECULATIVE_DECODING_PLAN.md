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

Status: **in progress**

First run a small manually reviewed table set. Then run all OmniDocBench v1.6
tables for every Phase 1 split mode. Run all row/band vision and text prefills,
then generate draft outputs with concurrent decode. Measure per-table and corpus
wall time, input vision tokens, text-prefill tokens, output tokens, malformed or
degenerate outputs, draft-to-full-output similarity, and stitched Page-TEDS.

The first six-table B8 experiment is complete. All row requests ended at EOS.
The ruled strategy scored 0.7927 Page-TEDS, the image-only selected strategy
scored 0.7569, and the saved whole-table outputs scored 0.7706 on the same six
tables. The selector is strong on simple and borderless tables but selects too
many narrow bands on dense tables. Phase 2 therefore stays in progress. Improve
selection and stitching on the dense cases before the full 665-table sweep.

Exit gate: draft artifacts are complete, reproducible, and sufficiently useful
to justify target-side speculative verification work.

## Phase 3 — Draft-input speculative target runtime

Status: **pending**

Implement a compiled target runtime that accepts precomputed draft tokens and
verifies a fixed token block against the authoritative full-table context.
Draft selection quality is not part of this phase. Validate KV state, rejection,
fallback, graph replay, timing, and baseline greedy-output parity.

Exit gate: the target runtime is correct and measurable with synthetic perfect,
partially correct, and incorrect drafts.

## Phase 4 — Draft matching and re-anchoring

Status: **pending**

Develop the candidate index, suffix matching, row-order tracking, re-anchoring,
candidate ranking, and optional multi-candidate verification. Optimize accepted
tokens per verification step and final table latency while preserving the Phase
3 correctness contract.

Exit gate: full 665-table latency and Page-TEDS comparison against the committed
B1 baseline.
