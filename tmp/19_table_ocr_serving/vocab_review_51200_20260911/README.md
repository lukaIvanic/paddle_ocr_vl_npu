# 51,200-token vocabulary review — 2026-09-11

Status: CPU-only inspection of a proposed selection. No serving preset changed,
no expanded-head inference run, and no claim of OCR parity. The automated audit
covers the inventory and saved records; the manual review examines representative
tokens, actual output contexts, formula commands and multilingual examples—not
every omitted vocabulary entry individually.

## Candidate examined

Keep the existing 16,384 selected IDs, add every non-special Han-containing token,
then protect single-codepoint and single-grapheme tokens (including leading-space
variants). Fill remaining positions using ascending existing non-special IDs.

| Component | Cumulative IDs |
|---|---:|
| Existing selection | 16,384 |
| Plus every Han-containing token | 35,995 |
| Plus character/grapheme core | 39,556 |
| Plus 11,644 lower-ID fillers | 51,200 |

The last filler ID is 23,338. Lower ID is a heuristic, not a measured general-use
frequency ranking. Protecting characters moves this cutoff down, so the previous
Han-only proposal's lexical coverage does not carry over automatically.

## Findings and manual interpretation

### Saved table generation

The full-vocabulary vLLM run has 1,000 responses covering all 665 unique tables.
Audit retains the last response for each repeated table ID. Six of these 665
streams contain excluded tokens: nine unique IDs, twelve occurrences.

| Excluded piece | ID | Actual generated context |
|---|---:|---|
| `TEMP` | 29785 | Header `TEMP MINUS`, `TEMP FOR` |
| `▁MIN` | 23464 | Part of `TEMP MINUS` |
| `▁Tac` | 84170 | `Tacit knowledge transfer` |
| `ldr` | 49098 | Fund abbreviation `CptIncBldrA` |
| `ISM` | 67113 | Fund abbreviation `TtISMIdx...` |
| `Convention` | 50413 | `Conventionally treated high-grade gliomas` |
| `▁SHOW` | 63572 | Weather legend `SHOWERS` |
| `STOR` | 37946 | Weather legend `THUNDERSTORMS` |
| `▁(+` | 33914 | Chinese financial growth percentage |

These are not all obscure vocabulary: weather, clinical descriptions, finance
and table headings are ordinary OCR use cases. Context records are generated
outputs, not claims that every output agrees with ground truth. In particular
`▁MIN` is a fragment of `MINUS` here, not proof of a minimum-temperature heading.

Both local-runtime full-head random-100 runs (B2 and B8) expose the same excluded
`TEMP`/`▁MIN` pieces in one table. Those comparisons are useful corroboration,
but are not independent coverage of all 665 tables.

### Table reference text, not HTML syntax

Parsed the 665 HTML references into 40,010 nonempty cell strings before tokenizing
each cell. 367 cells contain excluded IDs: 226 unique IDs, 447 occurrences.
Examples include `ETF`, `Compression`, `knowledge`, `reaction`, `macro`, math
commands `geqslant`/`leqslant`, and punctuation merges such as `%(` and `$^{`.

The separate raw-HTML audit has many more missing occurrences, largely markup
merges. It is not representative of the model's OTSL output format. Cell-token
boundaries also differ from complete generated OTSL tokenization; neither is an
accuracy score. All six inspected OTSL markers are retained.

### Formula and technical notation

For 2,066 isolated-equation references, 1,265 normal tokenizations use at least
one excluded piece (332 distinct IDs, 5,274 occurrences). Common examples:

- Relations: `Rightarrow`, `subseteq`, `parallel`, `perp`, `leqslant`, `geqslant`.
- Operators/construction: `mathop`, `binom`, `substack`, `underset`.
- Notation: `vdots`, `therefore`, `lfloor`, `rfloor`, `forall`, `notin`.
- Fonts/layout: `mathscr`, `displaystyle`, `scriptstyle`, brace/space merges.

The backslash and letters survive, but their usual merged command pieces do not.
Standalone command inspection finds 79 distinct missing pieces. That extraction
also catches code escapes/custom strings such as `\\ncolormap`, so it must not
be promoted blindly into a mathematical-command whitelist.

Representative formulas using `frac`, `sqrt`, `alpha`, `beta`, `gamma`, `Delta`,
`sigma`, `epsilon`, `varepsilon`, `sum`, `prod`, `int`, `infty`, `partial` and
`nabla` have no missing pieces in the tested strings. Neither do the tested
aligned environment and OTSL example. This is not an exhaustive LaTeX guarantee.

Units/notation probes have no missing pieces: µg/mL, μmol/L, mmol/L, mg/kg,
kg·m⁻³, m², s⁻¹, °C, °F, ±, ≤, ≥, ≠, ≈, ×, ÷, ¾, ½, ‰; chemical subscripts
and charges such as H₂O, Ca²⁺, SO₄²⁻; scientific exponents and common currencies.

Judgment: explicitly protect mathematical command pieces and common technical
notation before allocating generic lexical filler. Do not equate symbol coverage
with formula-generation coverage.

### Chinese and other writing systems

Every non-special Han-containing vocabulary entry is protected, including both
individual characters and merged strings. Han is also used in Japanese; this is
a Unicode inventory criterion, not a Chinese-language classifier. Mixed Chinese
documents still use Latin abbreviations, math and punctuation tokens, and the
financial `▁(+` example demonstrates that the Han rule alone does not protect
their complete token streams.

The character/grapheme rule protects existing standalone script building blocks.
The original selection also retains all 256 byte-fallback entries. This is strong
representational coverage, not evidence of unchanged generation for every script.

Manual phrase probes:

- Japanese `平均 温度 標準偏差 カタカナ ひらがな`: no missing pieces.
- Korean `평균 온도 표준편차`: no missing pieces in this example.
- Russian: missing word fragments including `тура`, `▁сред`, `ное`, `лон`.
- Arabic: missing fragments including `جة`, `ارة`, `▁مت`, `وسط`.
- Hindi/Thai: missing `मान`, `▁मान`, `ี่ย` in the tested phrases.
- Accented Latin names: missing merged pieces in Ångström, Müller, François,
  İstanbul, Straße and café. The basic accented characters remain covered.

Judgment: this is a Han-prioritized multilingual selection, not equal multilingual
coverage. Prefer retaining common word pieces where space permits; long merged
pieces can be important for names, technical language and output length.

## What can fit within 51,200?

The union of the existing selection, all Han tokens, the character/grapheme core,
all canonical table-cell and isolated-equation reference IDs, and all inspected
native-output IDs contains **40,399 IDs**. That leaves **10,801** rows for broader
coverage within 51,200. Relative to the initially filled candidate, this union
requires 541 additional IDs; protected IDs would displace filler, not increase
the row count. This is a feasibility result, not a newly deployed selection.

However, preserving canonical tokenizations of **all** inspected text/LaTeX/HTML
annotation fields across 1,651 pages, as well as the original/Han/core protection,
requires **60,153 IDs**. This includes captions, prose, references, code, markup
and ignored regions. It proves that “every OmniDocBench token sequence unchanged”
is a stronger requirement than the current 51,200 budget allows under these
protection rules. It does not prove that 51,200 cannot reproduce the same text.

Dataset-derived protection should remain explicit. It is useful as a regression
floor, but cannot justify a general OCR-quality guarantee or replace broader
formula/script review. No per-request routing or benchmark-ID exception is used.

## Recommendation

Do not finalize the lower-ID-filled proposal unchanged. Protect the original
selection, all Han and core characters; strengthen common formula and technical
pieces; ensure known table/equation coverage; then use the remaining budget for
general lexical coverage. Keep a distinct count for empirically protected corpus
tokens versus general rules. Do not silently loosen either the budget or scope.

Omitting a merged token does not make its spelling impossible, but greedy decoding
does not automatically replace it with the correct shorter sequence. Different
token histories can change later output, and extra tokens can cost time or meet
the unchanged KV/output cap earlier. Only full-head comparisons on actual crops
can measure that effect. No inference/accuracy conclusion is drawn here.

## Evidence

- `review.py`: reproducible CPU inventory, reference and native-ID audit.
- `audit.json`: counts, exact IDs, source hashes, generated contexts and probes.
- Tokenizer SHA256: `c8a215a59183d0d0781adc33bacd3ce6162716f7fd568fb30234a74d69803a7d`.
- Temporary input locations are recorded in the script; tokenizer and full
  annotation inputs were copied read-only from the existing checkpoint/dataset.
- `repair_counts.protected_plus_all_reference_ids` is the full-union count.
  The older `protect_reference_tokenizations_total` field counts the protected
  base plus only IDs missing from the initial candidate; it is **not** the
  full reference-union size and must not be presented as such.
