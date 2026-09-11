# Follow-up character review of the native-ID union

2026-09-11. Review only; no active selection or inference configuration changed.

Starting point: the 59,537-ID union described in NATIVE_RESULTS.md. This combines
actual generated IDs with the existing selection, all non-special Han-containing
tokens, and single-character/grapheme protections. No reference retokenization is
used to establish generated-token coverage here.

## Inventory checks

- All **26,063** non-special Han-containing token entries are retained.
- All **6,360** single-codepoint entries are retained (SentencePiece space marker
  interpreted as a space).
- All **8,377** entries meeting the single-grapheme/character protection rule
  are retained, including leading-space variants. Counts overlap.
- All **256** byte-fallback entries are retained.
- All six entries containing Unicode variation selectors are retained.

Manual representative character review includes simplified/traditional Chinese,
financial numerals, Chinese quotation/bracket/punctuation styles, accented Latin,
Greek, mathematical signs and invisible characters. No existing direct token
for an inspected character is excluded. This is not a claim that every Unicode
character has its own token in the original checkpoint.

Examples such as 髮, 贰, 貳, 𠮷, and several mathematical signs do not have a
standalone exact-character entry in the original tokenizer. Byte fallback remains
available, and any Han-containing merged entries are retained. These are not
holes introduced by trimming, and inventing extra LM-head rows would not fix them.
There is no claim that byte representability guarantees accurate generation.

## Inexpensive additional protections

| Addition, in this order | Newly added IDs | Cumulative size |
|---|---:|---:|
| All whitespace-only entries | 14 | 59,551 |
| Han Script_Extensions-containing entries | 41 | 59,592 |
| All entries containing a combining mark | 713 | 60,305 |
| All entries containing a Unicode format-control character | 28 | 60,333 |
| Remaining tokenizer-declared special tokens | 19 | 60,352 |

The 41 Script_Extensions additions are punctuation/mixed sequences, not missing
Han ideographs: for example `】【`, `】，`, `】。`, `、「`, `）、《`, `……。`.
Some are code/markup mixtures; retaining the entire small category is simpler
than hand-ranking these forty-one entries.

Whitespace additions include repeated ASCII, ideographic and nonbreaking spaces.
Combining-mark additions are mostly multi-character multilingual fragments, not
missing standalone marks: Devanagari, Bengali, Tamil and Thai pieces, along with
accented/romanized pieces such as `▁chi̍t`. This is a conservative optional
coverage expansion, not evidence that all 713 are required for OCR. The joining-
control additions include Persian/Arabic-script word fragments. Three of the
31 initially excluded format-control entries overlap the combining-mark category,
so that step adds only 28 new IDs.

The final 19 special entries are protocol/multimodal/mask markers, not ordinary
characters. Existing BOS/EOS/unknown IDs are already retained. Including the
remaining markers preserves more of the full head's possibilities at negligible
row cost, but does not by itself add support for their associated tasks or make
them alternative stopping tokens. No stopping rules are changed.

## Recommendation and limits

Use **60,416 rows (59 × 1,024)** as the next candidate's physical size. The reviewed
extension has 60,352 unique IDs, leaving 64 real vocabulary slots unassigned;
rounding alone does not select those IDs. Final filler selection and deployment
have not been performed.

This removes no observed generated IDs and adds 815 protection IDs to the 59,537
base. No missing standalone Chinese character entry was found. It remains a
restricted head: omitted multi-character words, names and notation can affect
greedy predictions even where all characters are spellable. Validate actual
generation against the full head before claiming equal OCR quality.

Reproduce with `review_native_core.py`; exact added IDs and all findings are in
`native_core_review.json`.
