# Unchanged custom MinerU token anchor

Trace-only source commit: `13061fc4`. Physical Ascend 910B2 NPU4.

`anchor.tar.gz` contains the complete 384-page output directory, command,
source commit, exit code and run log. Extract into an empty directory. The
`output/generation_trace.jsonl` file has 5,486 records: 384 layouts and 5,102
recognition requests. Generated token IDs include EOS and are not reconstructed
from decoded text. Each record includes prompt IDs, raw text, crop-image hash,
page/block identity, geometry, effective generation cap and stop reason.

The run manifest records model-weight, tokenizer/configuration and dataset-JSON
hashes. All 384 Markdown predictions were byte-identical to the earlier run at
`95fb6d8c`. Pipeline wall including tracing was 551.485 s; active decode-slot
fraction was 25.7234%. This anchor intentionally retains the old refill bug.

Verify `SHA256SUMS` before using the archive. Do not overwrite this reference
when updating the serving pipeline. New runs belong in separate directories.
