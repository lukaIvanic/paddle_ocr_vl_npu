# Exact MinerU config used on 910B

`config.json` is a byte-for-byte reference copy retrieved on 2026-09-05 from:

```text
/workspace/models/MinerU2.5-Pro-2605-1.2B/config.json
```

SHA-256 (matches the saved production/profile run asset hash):

```text
22097df08750242647a513043636a8dff16820a09757e9271e220bdea378df28
```

For the 310P agent: compare this file with your actual model's `config.json`
and report the changed JSON fields and values directly to Luka in plain text.
Distinguish whitespace/key-order differences from changed values. Do not
overwrite your model config, edit this reference, clear caches, or bypass the
profiling preflight hash gate merely to make the hashes match.

The reference declares `bfloat16` at top level and in both `text_config` and
`vision_config`; the validated custom runs explicitly load **float16**. Do not
change the runtime to BF16 based on this file. The top-level position limit is
32768 while `text_config.max_position_embeddings` is 8192. These are preserved
as they exist on the 910B model, not normalized or corrected.

This copy is comparison evidence, not a replacement model asset or a new
runtime configuration. Neither the 910B model nor its caches were modified.
