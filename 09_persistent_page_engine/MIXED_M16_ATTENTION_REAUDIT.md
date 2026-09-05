# Mixed M16 attention: full-forward re-audit

2026-09-05, Ascend 910B2, physical NPU 6.

## What the earlier experiments could establish

The original mixed graph combined B1Q8/KV4096 verification and B8Q1/KV768
drafting into one M16 transformer. Matmuls benefited, but attention and layout
costs offset most of that saving. The best two-attention layout retained
separate manual verifier attention and draft IncreFA, with draft attention
first, per-lane ApplyRotary, and full prefetch.

The subsequent single-attention probe began with projected queries and
already-prepared caches. It excluded projection splitting, RoPE, cache updates,
prefetch interactions and output projection. It also synchronized after every
measured call, whereas the full-model anchor queues its repeated calls before
resolving device events. Its 352.7 / 388.4 / 397.2 microsecond operator-level
medians were not a full-model comparison and were insufficient grounds to
reject the full design.

The Qwen reference reinforces this distinction. Experiment 13's
`combined_bsnd` comparison reported almost unchanged attention operator time;
the improvement came from removing transposes in the surrounding full graph.
See `13_qwen3_reranker/README.md`, "combined_bsnd".

## Full-forward experiment

The existing `scripts/table_q_anchor_benchmark.py` now supports single-attention
layouts in `paddleocr_vl/model/text_mixed_q.py`. Every timed call includes token
embedding, all 18 transformer layers, RoPE, masks, persistent cache writes,
prefetch, all projections and MLPs, the selected 16,384-row LM head, argmax,
and mapping to native token IDs.

Each variant loads the model in its own process. It uses 10 ordinary warmups
(including first-call compilation), then 50 unprofiled measured calls. A
separate profiler window uses another 10 warmups and 50 captured calls.
All variants use FP16, NZ linear weights and the same fixed input IDs and
positions: verifier start 1249; draft positions
128, 155, 173, 189, 205, 225, 270, 382.

These are complete model-forward measurements on the original deterministic
anchor fixture, whose historical KV prefix is zero. They are not OCR request
latencies or a real-table generation/quality evaluation. Cache initialization
and eventual conversion between standalone and mixed serving arenas are not
in the measured forward; all repeated per-iteration cache writes are included.

## Results

| Full M16 variant | Attention Q shape | Persistent KV storage | Median forward |
|---|---|---:|---:|
| Best two-attention control | B1Q8 + B8Q1 | 180 MiB | 2.3725 ms |
| Same control, repeated later | B1Q8 + B8Q1 | 180 MiB | 2.3813 ms |
| Single packed BSND PromptFA | B1Q16 | 180 MiB | 3.5632 ms |
| Single two-row BSND PromptFA | B2Q8 | 216 MiB | 3.7694 ms |
| Single padded BSND PromptFA | B9Q8 | 648 MiB | 5.6870 ms |
| Single IncreFA, indexed cache writes | B16Q1 | 1,152 MiB | 3.1723 ms |
| Single IncreFA, dedicated KV Scatter | B16Q1 | 1,152 MiB | 2.6815 ms |

The B2Q8 arrangement was absent from the earlier probe. One batch row contains
the verifier queries; the other contains all eight draft queries. Both use a
6144-slot cache row. A block mask isolates each draft's 768-slot segment.
There are no dummy transformer tokens or dummy attention queries in this
variant. Despite fewer dense Q-by-KV pairs than packed B1Q16, it was slower.
The tested hardware kernel behavior cannot be inferred from arithmetic count.

In B9Q8, dummy queries are introduced only immediately around attention and
discarded before output projection. They receive a valid mask and never enter
the MLP, LM head, or cache writes. Each draft has its own physical KV4096 row.

In B16Q1, eight verifier queries use eight persistent copies of the same
logical verifier history. Each has its own causal endpoint. Only new Q8
blocks are replicated per iteration; full histories are not recopied each
step. The eight draft rows are padded to KV4096. All 16 transformer tokens
remain real.

## What the profiles explain

Microseconds below are sums of profiled kernel durations divided by 50 captured
full forwards. Profiler sums are diagnostic and are not interchangeable with
the unprofiled forward medians above.

| Kernel group | Control | Packed PromptFA | B2 PromptFA | B9 PromptFA | B16 IncreFA |
|---|---:|---:|---:|---:|---:|
| Attention, including manual QK/PV/softmax | 820.2 | 1987.6 | 2159.3 | 3993.4 | 1024.1 |
| Transformer matmuls + LM head | 474.2 | 560.9 | 531.5 | 577.1 | 498.8 |
| Transpose | 270.0 | 7.4 | 7.3 | 7.4 | 7.3 |
| KV writes | 156.5 | 312.1 | 346.1 | 281.7 | 685.8 |
| ApplyRotary | 154.9 | 117.9 | 128.1 | 107.4 | 108.6 |

The BSND implementation actually removes all 36 layer transposes; the single
remaining transpose is outside the layer stack. Thus the layout benefit was
real. PromptFA's added cost exceeded it. B9 also introduced 90 slice kernels
and more output assembly.

The IncreFA alternative isolates a different problem: attention is only about
204 microseconds more expensive than the control, while general indexed
`ScatterNdUpdate` costs about 529 microseconds more. Its cache replication also
adds about 201 microseconds of BroadcastTo work.

The dedicated KV Scatter follow-up gives all 16 cache rows a uniform Q8 update
block. Draft rows write one real K/V pair followed by seven zero future slots;
their Q1 causal masks hide those extra slots. Draft logical KV768 fits well
inside the physical KV4096 arena. This permits one stock `scatter_update_` per
K or V cache instead of general indexed updates.

| B16 IncreFA kernel group | Indexed writes | Dedicated KV Scatter |
|---|---:|---:|
| Attention | 1024.1 us | 1039.0 us |
| KV writes | 685.8 us | 155.6 us |
| Transpose | 7.3 us | 165.7 us |
| BroadcastTo | 201.2 us | 176.6 us |
| ConcatV2D | 12.7 us | 97.1 us |
| Full unprofiled forward | 3.1723 ms | 2.6815 ms |

This removes a substantial cache-write penalty, but preparation of the uniform
write blocks introduces transposes and concatenation. The full call improves
by 15.5% and still trails the mean of the two control medians by 12.8%.
Therefore, "one IncreFA call is slow" was also too broad: the indexed-write
implementation accounted for much of that candidate's initial regression.

The current result supports retaining the two-attention control. It does not
establish that every single-attention layout or mixed-M16 design is unviable.
The useful unresolved target is an attention/cache representation that preserves
independent histories efficiently without spending the shared-matmul saving on
cache replication, padding, or layout conversion. Any next candidate must be
judged through this complete forward and profiler boundary.

## Validation and evidence

All five completed single-attention candidates matched all 16 native output
IDs against the full eager two-attention reference on the same anchor fixture.
All inspected caches were finite. KV tensors were not bit-identical:
PromptFA variants had maximum absolute differences of 0.03515625 (verifier)
and 0.076171875 (draft); both IncreFA variants had 0.0576171875 (verifier)
and 0 (draft).
These are recorded numerical differences, not proof of real OCR output parity.

Per-run JSON retains exact argv, commit, physical device selection, warmup
times, timings, native output IDs, full-forward KV comparisons, and profiler
paths. Evidence is under
`tmp/09_persistent_page_engine/mixed_full_reaudit_20260905/`.
`profiles.tar.gz` retains the raw kernel and step-trace CSVs for all seven
captures, including both control runs. The NPU was free after the experiments;
the final snapshot is retained as `npu6_after.txt`.

The candidate layouts are benchmark options; no serving default is changed.

## Replicated IncreFA follow-up

Later on 2026-09-05, the full-model BSH implementation improved beyond the
two-attention control. Same device, FP16/NZ/compact head, separate processes,
10 warmups and 50 repeats, with a separate 10/50 profiler window for each
successful run.

| Full forward | Cache capacity | Median |
|---|---|---:|
| Ordinary B16Q1, saved earlier anchor | KV768 | 1.3567 ms |
| Ordinary B16Q1, current matched-position control | KV4096 | 2.0900 ms |
| Replicated IncreFA with dedicated BNSD Scatter, preceding audit | KV4096 | 2.6815 ms |
| Replicated IncreFA with direct gathered BSH writes | KV4096 | 2.3996 ms |
| BSH with hoisted starts and one complete RoPE lookup | KV4096 | 2.2314 ms |
| Same final BSH implementation, separate-process repeat | KV4096 | 2.2550 ms |

The current ordinary B16 control uses the same input IDs and positions as the
mixed candidate: first eight positions 1249 through 1256, remaining positions
128,155,173,189,205,225,270,382. It still performs ordinary independent Q1
writes, so it is a cost reference rather than a semantic verifier substitute.
The KV768 historical anchor uses shorter draft positions throughout; the two
ordinary rows are not a pure one-variable cache-capacity sweep.

The final implementation keeps persistent caches in BSH `[16,4096,256]`.
Gathering from the 16 projected K or V rows directly constructs each uniform
Q8 write block. Draft rows repeat their one new token into seven future slots;
the causal mask hides those slots and later iterations overwrite them before
use. Scatter writes along axis 1. IncreFA consumes BSH directly, so neither
cache-update preparation nor attention needs a transpose.

The cache-start vector is constructed once before all 18 layers. The profile
showed that relying on compiler common-subexpression elimination had instead
left 18 repeated gathers, broadcasts and concatenations. The final variant
also uses the existing scalar RoPE lookup for all 16 generated positions. The
earlier mixed implementation used lookup only on the draft side despite its
optimization preset specifying lookup. All generated-text position axes are
identical after applying the corresponding request's rope delta, permitting
this single lookup without changing the intended MRoPE semantics.

| Profile group | Ordinary B16 KV4096 | First gathered BSH | Final BSH |
|---|---:|---:|---:|
| IncreFA | 997.8 us | 1037.1 us | 1024.9 us |
| Matmuls + LM head | 507.8 us | 541.1 us | 506.0 us |
| KV Scatter | 162.7 us | 103.4 us | 114.5 us |
| GatherV2 | 21.5 us / 4 calls | 201.4 us / 70 calls | 149.7 us / 40 calls |
| Transpose | 0 | 8.2 us / 1 call | 0 |
| ConcatV2D | 0 | 38.5 us / 22 calls | 3.4 us / 2 calls |
| BroadcastTo | 0 | 25.4 us / 21 calls | 3.0 us / 2 calls |

The final candidate is about 16.4% faster than its preceding BNSD/Scatter
implementation and 5.6% faster than the mean of the two original best
two-attention control medians. It is about 7.3% slower than the matched-position
ordinary B16 control. Most of that remaining difference is the additional
per-layer gather for replicating verifier updates. KV storage remains
1,152 MiB, versus 180 MiB for the two-attention mixed representation.

Validation now covers three advancing teacher-forced steps. The final repeat
uses independent nonzero histories (`normal(0,0.02)`, seed 74), verifier starts
1249,1250,1254, and draft advances of one token. All 48 native output IDs match
the full eager two-attention reference. Valid-prefix cache maximum absolute
differences are 0.140625 for the verifier and zero for drafts; all caches are
finite. Future slots are reported separately because the physical filler
writes intentionally differ from the reference. These remain synthetic
full-forward and cache-contract tests, not a real-table serving evaluation.

The first BSH attempt failed compilation because TorchAir lacks the
`aten.repeat_interleave.self_int` converter. Reshape/expand replaced it; the
subsequent full graphs compiled normally. That failed attempt's log is retained
and has no reported timing.

Follow-up JSON, commands and raw kernel/step-trace CSV archive:
`tmp/09_persistent_page_engine/mixed_increfa_followup_20260905/`.
The final measured model source is commit `b6416dbf`; the stronger validation
is commit `5bc32272`. NPU 6 was released after the repeat.
