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

## OTSL grammar and column information

The recognizer emits a compact table grammar. Every marker is one model token.

- `<fcel>` starts a normal filled cell.
- `<ecel>` occupies an empty independent cell.
- `<lcel>` continues the cell to the left through a column span.
- `<ucel>` continues the cell above through a row span.
- `<xcel>` continues a merged region through both axes.
- `<nl>` ends the current table row.

Column width is the number of cell-slot markers between two `<nl>` markers,
including span continuations. On the 143-table cohort:

- the target first row matches the target modal width in 139 tables;
- all target rows have one consistent width in 132 tables;
- draft modal width matches target modal width in 117 tables;
- 2,916 of 3,761 draft logical rows have the exact target width;
- 3,518 are within one slot and 3,607 are within two slots.

Before target decode, draft consensus width is a useful soft hypothesis. Once
the target produces its first `<nl>`, its own first-row width is the stronger
constraint.

A bounded structural patch lattice is therefore practical. For a draft row of
width `C-1`, create the `C` possible empty-slot insertion paths. For `C+1`,
create bounded deletion paths. These are proposal candidates only. The target
model remains authoritative and rejects a wrong structural patch normally.

## Why exact oracle continuations become unanchored

For 41,369 call-local lost-token opportunities, the oracle continuation has no
matching token immediately before it:

| Prefix divergence | Lost opportunity tokens | Share |
|---|---:|---:|
| Numeric content | 8,791 | 21.3% |
| Structure/content boundary | 7,258 | 17.5% |
| Punctuation or whitespace | 7,065 | 17.1% |
| Unicode-equivalent formatting | 4,968 | 12.0% |
| Text content | 4,251 | 10.3% |
| Math or LaTeX form | 3,707 | 9.0% |
| Different OTSL structure token | 3,208 | 7.8% |
| CJK content | 1,090 | 2.6% |
| Start of a draft band | 1,031 | 2.5% |

At least 56.9% is clearly structural or formatting divergence. Including math
and LaTeX representation raises that share to 65.9%. Common exact-token
differences include ASCII versus full-width punctuation, alternate minus and
parenthesis characters, `<ucel>` versus `<nl>` or `<ecel>`, and equivalent
LaTeX spellings.

This is why exact contiguous suffix matching loses the correct location. An
online approximate aligner can preserve state across one substitution,
insertion, or deletion and then continue proposing the exact draft tokens that
follow.

## Concentration

- 128 of the 143 tables have at least one gap.
- The worst 10 tables contain 34.0% of call-local lost opportunity tokens.
- The worst 20 contain 52.3%.
- The single worst table contains 5.1%.

Large tables deserve focused regression tests, but the failure is general and
cannot be fixed with per-table rules.

## Matcher strategy experiment

All strategies were simulated from exact saved target and eight-band draft
tokens. They use generated target-prefix tokens only. The oracle can inspect
future target tokens and is an upper bound.

### Complete 665-table corpus, K=16

| Matcher | Accepted/call | Coverage | Target calls | Ideal target-decode speedup |
|---|---:|---:|---:|---:|
| Current exact suffix | 5.934 | 84.03% | 42,708 | 5.752x |
| Reversible cursor | 6.317 | 84.77% | 40,729 | 6.034x |
| Column-aware exact | 6.414 | 84.95% | 40,259 | 6.105x |
| Column + virtual width patch | **6.429** | **84.98%** | **40,185** | **6.117x** |
| Oracle | 10.210 | 90.42% | 25,616 | 9.556x |

The virtual width patch changes candidate scoring only. It does not rewrite the
draft output. Rows within two slots of the learned target width can align their
candidate column through a bounded insertion/deletion hypothesis.

### 143 tables with measured B1 latency above one second

| Matcher | Accepted/call | Coverage | Target calls | Ideal target-decode speedup |
|---|---:|---:|---:|---:|
| Current exact suffix | 6.056 | 84.99% | 25,630 | 6.089x |
| Reversible cursor | 6.567 | 85.93% | 24,011 | 6.503x |
| Column + virtual width patch | **6.778** | **86.29%** | **23,407** | **6.672x** |
| Hybrid exact/beam | 6.217 | 86.15% | 23,647 | 6.561x |
| Free beam | 5.041 | 83.45% | 28,251 | 5.492x |
| Oracle | 11.080 | 91.44% | 14,617 | 10.649x |

The column-patch matcher removes 2,223 target calls versus the current matcher
and 604 versus the reversible cursor. Against the reversible cursor, it is
better on 57 tables, equal on 46, and worse on 40. Improvements save 973 calls;
regressions add 369. Two page-279 tables provide 522 of the saved calls, so the
aggregate win is real but partly concentrated.

### CPU cost

The lab times only each matcher simulation. It excludes tokenizer loading,
JSON parsing, and report writing.

| Cohort | Current | Reversible | Column + width patch |
|---|---:|---:|---:|
| All 665, CPU ms/table | 5.162 | 4.835 | 6.029 |
| All 665, CPU us/target call | 80.372 | 78.935 | 99.778 |
| Latency above 1 s, CPU ms/table | 15.841 | 14.666 | 18.173 |
| Latency above 1 s, CPU us/target call | 88.384 | 87.343 | 111.026 |

The selected matcher adds 0.867 ms per table over the current matcher on the
complete corpus and 2.332 ms per table on the slow cohort. At 750 target tokens
per second, its added 19-23 us per target call is approximately 1.5-1.7% of
one model iteration. This CPU cost does not erase the reduction in target calls.

### Table-start prior

The full-table model can begin with `<ecel>` while the first row-band draft
begins with `<fcel>`. An exact suffix lookup then jumps to an unrelated later
`<ecel>`. A grammar-scoped start prior proposes draft position zero only for
this exact `(target <ecel>, draft <fcel>)` transition.

Across all 665 tables, this changes reversible-matcher calls from 40,729 to
40,667. It improves 48 tables, leaves 617 unchanged, and regresses none. On
`page_000263_table_box_id_7`, it accepts the correct first 16 draft tokens and
changes total calls from 803 to 801. The net gain is small because the original
matcher recovers at target position three, but the fix is legal and safe.

The free-running fuzzy beam is not a good replacement. It can recover weak
anchors, but it also overrides reliable exact locations and makes too many
short proposals. The hybrid limits that damage, but still loses to the simpler
column-aware exact matcher.

## Recommended matcher direction

Use the column-aware exact matcher with reversible cursor repair as the next
runtime candidate. Keep exact-prefix length authoritative. Use learned OTSL
column position and the bounded width hypothesis only to break candidate ties.
Measure its CPU time inside the real decode scheduler before enabling it by
default.

Do not put the free beam in the runtime. Retain it as a research tool for the
remaining oracle gap. A future beam should activate only when no reliable exact
anchor exists and should have a confidence gate that can decline to propose.

## Artifacts

- Analyzer: `scripts/analyze_table_matcher_failures.py`
- Detailed output:
  `tmp/09_persistent_page_engine/table_matcher_analysis_gt1s_20260808/failure_analysis_v3`
- Anchor sweep:
  `tmp/09_persistent_page_engine/table_matcher_analysis_gt1s_20260808/global_anchor_sweep`
- Reversible cursor:
  `tmp/09_persistent_page_engine/table_matcher_analysis_gt1s_20260808/reversible_cursor`
- Strategy lab:
  `scripts/table_matcher_strategy_lab.py`
- Complete exact/column comparison:
  `tmp/09_persistent_page_engine/table_matcher_strategy_lab_20260808/full665_exact`
- Slow-table beam comparison:
  `tmp/09_persistent_page_engine/table_matcher_strategy_lab_20260808/full143`
