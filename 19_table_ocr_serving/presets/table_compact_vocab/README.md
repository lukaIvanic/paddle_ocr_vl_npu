# Decode vocabulary

`native_han_core_60416.json` is the only selected vocabulary shipped with this
runtime. It contains all 60,416 native token IDs in LM-head row order and their
SHA256: `c730b5388f9871ead92e2cb484f8df81ba69518f1e8baeccc5a37f44c1514637`.
The runtime constructs selected weight rows and the reverse native-ID mapping
from this file. No tokenizer, corpus, remote path or audit script is required to
load the selection. The checkpoint/tokenizer itself must be provided locally,
as for the rest of the engine.

The default is this selected head. `--full-decode-lm-head` uses all 103,424 native
checkpoint rows directly, so it does not load a selected-ID file. First-token
selection after text prefill uses the full head in both modes. Neither mode
changes stopping rules, input preprocessing, or KV capacity.

The selected vocabulary protects actual saved full-head generation IDs, all
Han-containing token entries, standalone character/grapheme entries, whitespace,
Han-associated punctuation sequences, combining-mark and format-control pieces,
and tokenizer-declared special tokens. The reviewed protected union has 60,352
IDs; the remaining 64 are deterministic lower-ID fillers. The file also retains
the original row-order digest and raw-trace hashes as provenance; those source
paths are metadata, not runtime dependencies.

All inspected generation IDs are covered, but this is still a restricted head.
Omitted merged tokens can affect generation on other inputs. Byte/character
coverage alone does not establish general OCR accuracy. The 100-table B2/B8
comparison matched the full head exactly; wider validation is separate.

The old 16,384-row mapping is removed from experiment 19. Its research copy and
past benchmark evidence remain in experiment 09 and Git history, not in the
product's supported vocabulary choices.
