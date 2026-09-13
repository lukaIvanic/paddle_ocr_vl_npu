# Decode vocabulary

This runtime uses **our own reduced vocabulary of 60,416 tokens by default**,
instead of PaddleOCR-VL's original 103,424-token vocabulary. The purpose is to
reduce latency. It is our optimization, not an official Paddle vocabulary.

We kept the actual token IDs generated in our full-vocabulary OmniDocBench
runs, plus all Han-containing tokens and reviewed basic characters, language,
math/LaTeX and syntax tokens. In our OmniDocBench v1.6 tests, outputs matched
the full head exactly; mean latency was about 2–3% lower and P95 about 3–6% lower.
The smaller vocabulary doesn't reduce quality of generation, but there is an 
option available to use the original full vocabulary instead. 

To use Paddle's original full vocabulary instead, start the server with
`--full-decode-lm-head`.

## Mapping used by the runtime

`native_han_core_60416.json` is the only selected vocabulary shipped with this
runtime. It contains all 60,416 native token IDs in LM-head row order and their
SHA256: `c730b5388f9871ead92e2cb484f8df81ba69518f1e8baeccc5a37f44c1514637`.
The runtime constructs selected weight rows and the reverse native-ID mapping
from this file. No tokenizer, corpus, remote path or audit script is required to
load the selection. The checkpoint/tokenizer itself must be provided locally,
as for the rest of the engine.
