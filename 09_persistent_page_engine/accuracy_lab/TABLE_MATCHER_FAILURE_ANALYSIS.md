# Table draft matcher failure analysis

## Scope

This analysis uses the 143 OmniDocBench v1.6 tables whose measured sequential
B1 OCR latency is greater than 1.0 second. It uses the orientation-normalized,
boundary-snapped eight-band drafts and a 16-token verification block.

The practical matcher is legal: it sees only the generated target prefix and
the precomputed row drafts. The oracle can inspect future target tokens and is
only an upper bound.

## Current algorithm

The matcher removes the EOS token from each of the eight band outputs and
concatenates them into one token stream. It builds exact suffix indexes for
anchor lengths 1, 2, 4, 8, 16, 32, and 64.

For each target-model call, it:

1. takes the already-generated target suffix;
2. finds every matching suffix in the flat draft stream;
3. prefers the longest exact preceding match;
4. then prefers a candidate at or after the global cursor;
5. then prefers the candidate nearest the cursor;
6. proposes the following 16 draft tokens;
7. advances the cursor with `max(old_cursor, accepted_draft_position)`.

It has no explicit band identity, row order, table-structure weighting,
approximate alignment, or confidence score.

## Practical versus oracle

| Metric | Practical | Oracle |
|---|---:|---:|
| Accepted draft tokens per speculative call | 6.06 | 11.08 |
| Target decode coverage | 85.0% | 91.4% |
| Target-model calls | 25,630 | 14,617 |
| Ideal target-decode speedup | 6.09x | 10.65x |

The practical path makes 25,630 decisions. At 53.5% of them, another draft
location has a longer exact continuation than the selected location.

Call-local opportunity counts are diagnostic. They are not additive to the
oracle aggregate because practical and oracle acceptance produce different
future call positions.

## Failure classes

| Failure class | Calls | Share of gap calls |
|---|---:|---:|
| Wrong band selected | 5,555 | 40.5% |
| Oracle continuation has no exact prefix anchor | 5,307 | 38.7% |
| Wrong location inside the correct band | 1,395 | 10.2% |
| No suffix candidate exists | 1,444 | 10.5% |

Of the call-local lost-token opportunity:

- 55.3% has a legal exact prefix anchor and is recoverable by better ranking;
- 44.7% has no exact prefix anchor and needs approximate alignment or a
  position/structure prior;
- 72.2% of gap calls have more than one location sharing the indexed anchor;
- 72.6% of gap calls select a different band from the oracle.

Among the 6,950 anchored gap calls, the selected and oracle anchor lengths are
equal in 3,643 cases. In the other 3,307 cases, the current matcher selects a
longer historical anchor but a worse future continuation. Therefore, exact
anchor length must not be an absolute first-ranking rule.

## Cursor poisoning

The global cursor can move only forward. This creates a strong late-band bias.

| Cursor position relative to target progress | Lost opportunity tokens |
|---|---:|
| More than 25% ahead | 52,353 |
| 10-25% ahead | 18,164 |
| Within 10% | 22,033 |
| More than 10% behind | 31 |

Selected later-than-oracle bands account for 47,769 lost opportunity tokens.
Earlier-than-oracle bands account for 19,988. The state does not recover after
an accepted backward match because the update uses `max`.

A legal reversible-cursor probe changes only that update:

| Matcher | Accepted/call | Coverage | Target calls | Decode speedup |
|---|---:|---:|---:|---:|
| Current global cursor | 6.06 | 85.0% | 25,630 | 6.09x |
| Reversible global cursor | 6.57 | 85.9% | 24,011 | 6.50x |
| Oracle | 11.08 | 91.4% | 14,617 | 10.65x |

The reversible cursor removes 1,619 target calls, or 6.3%, without future-token
access. It is a useful corrected baseline, not the final matcher.

## What causes the first divergence

This classification describes the first target token that the selected
candidate fails to match. It does not claim that this token caused the wrong
location selection.

| Token class | Gap events | Call-local lost opportunity tokens |
|---|---:|---:|
| Numeric | 7,117 | 50,561 |
| Text | 2,078 | 12,335 |
| Table structure | 1,718 | 12,234 |
| CJK text | 953 | 8,795 |
| Math or LaTeX | 1,058 | 4,047 |
| Punctuation | 662 | 3,943 |
| Whitespace | 115 | 666 |

Numeric fields dominate because single digits, separators, and repeated values
produce many identical suffix anchors.

The selected proposal is usually not a one-token near miss. Among gap calls
with a candidate, 76.3% of 16-token proposal windows have at least eight token
edits from the target. Only 150 calls have edit distance one. The matcher is
usually at the wrong location.

## Anchor threshold and monotonic controls

Raising the minimum exact anchor improves proposal precision but loses too many
proposal opportunities.

| Minimum anchor | Accepted/call | Coverage | Target calls | Decode speedup |
|---:|---:|---:|---:|---:|
| 1 | 6.06 | 85.0% | 25,630 | 6.09x |
| 2 | 6.82 | 84.1% | 27,160 | 5.83x |
| 4 | 8.13 | 79.9% | 34,236 | 4.75x |
| 8 | 11.01 | 70.6% | 50,214 | 3.33x |

A strict monotonic cursor over the flat stream is much worse. Depending on the
minimum anchor, coverage falls to 30-34%. Each band output is a self-contained
mini-table. The target often needs to reuse structural scaffolding found near
the start of another band. Flat monotonic order forbids those useful moves.

## Manual cases

### Wrong band: `page_000178_table_box_id_1`

The whole-table target is a Chinese financial statement. At the operating
profit row, the correct continuation is in band 3:

`<fcel>3,683,038<fcel>5,900,209`

The matcher selects a repeated numeric-cell pattern in band 6:

`4,173,212<fcel>4,216,...`

The suffix anchor is short and appears in several bands. The forward-only
cursor favors the later occurrence.

### Wrong location inside one band: `page_000207_table_box_id_1`

Band 6 contains several Chinese AI-product rows. Both locations have the same
two-token preceding anchor. The oracle continues with:

`智能创作平台<fcel>百度在线网络技术有限公司...`

The practical matcher jumps farther into the same band and continues with:

`在线网络技术有限公司<fcel>智能写作会员...`

Cursor distance cannot distinguish repeated company and product fields.

### Unanchored continuation: `page_000199_table_box_id_0`

The target and band OCR differ in full-width punctuation, inserted line breaks,
and placement of postal-code text. The correct next address sequence exists in
the band draft, but the token immediately before it does not match the target
suffix. The exact suffix index cannot reach it.

### Structural discontinuity: `page_001276_table_2`

After a small LaTeX-format difference, the target reaches `<nl>` followed by a
row that exists exactly in band 0. No target suffix matches the draft context,
so the practical matcher falls back one token at a time even though a long exact
future continuation is available.

## Concentration

- 128 of the 143 tables have at least one gap.
- The worst 10 tables contain 34.0% of call-local lost opportunity tokens.
- The worst 20 contain 52.3%.
- The single worst table contains 5.1%.

Large tables deserve focused regression tests, but the failure is general and
cannot be fixed with per-table rules.

## Recommended matcher direction

The next matcher should preserve the eight draft streams separately and run an
online, prefix-only alignment beam over `(band, token_offset)` states.

The beam should:

1. allow matches, substitutions, insertions, and deletions so one formatting
   difference does not destroy location state;
2. use band order as a soft prior, not a strict monotonic constraint;
3. weight rare content tokens and structural boundaries more than `<fcel>`,
   digits, and whitespace;
4. let an accepted backward match repair the cursor;
5. rank candidates using alignment score, band position, target progress, and
   confidence margin instead of lexicographic anchor length;
6. keep the current target model authoritative and use only generated prefix
   tokens.

The first implementation comparison should use the reversible cursor as the
baseline. The primary metrics are target-model calls, accepted tokens per call,
coverage, zero-accept calls, and the top-20 manual regression cases.

## Artifacts

- Analyzer: `scripts/analyze_table_matcher_failures.py`
- Detailed output:
  `tmp/09_persistent_page_engine/table_matcher_analysis_gt1s_20260808/failure_analysis_v3`
- Anchor sweep:
  `tmp/09_persistent_page_engine/table_matcher_analysis_gt1s_20260808/global_anchor_sweep`
- Reversible cursor:
  `tmp/09_persistent_page_engine/table_matcher_analysis_gt1s_20260808/reversible_cursor`
