# Receipt note — 2026-10-08

The actual smoke and full-page inference used the clean, detached source
`d4e7fdd0efd90bf6408c21efab320cc3292f07a4`. The commands, environment and
before/after snapshots are preserved in `original_raw_evidence.tar.gz`.
The full-run receipt is also copied here without editing.

The initial coordinator (`d0bdb860`) stopped **after successful smoke inference**:
it incorrectly compared the generated two-page smoke dataset's hash with the
full 1,651-page dataset's hash. This was a coordinator check failure, not an
inference failure. The authoring change `d9ec5ed0` skips only that dataset-hash
comparison for the explicit two-page smoke. Checkpoint hashes remain required;
the full run still requires the full dataset hash. That corrected coordinator
validated the existing smoke and launched the untouched full stage with
`--continue-after-smoke`; no smoke inference was repeated and no receipt was
rewritten.

The coordinator's derived `gate.json` used the wrong summary key for its
`source_commit` field, producing `null`. The original summary's `git_commit`,
the inference receipt's source commit and `source_commit.txt` all identify
`d4e7fdd0`. The cosmetic key was fixed in the later harness, but these original
derived gate files are retained unchanged.

The 910B reproduction changes the hardware from the proposed 310P run. It
establishes no 310P performance, compatibility or quality result. All 910B
figures include the production pipeline's first-use cached-graph loads and
launch gaps; setup is separately reported. No isolated-attention result is
substituted for the full page pipeline or full vision encoder.

Archive SHA-256:
`bd8a4a1cd9bc3411eeeaef9cdce77e5a4b9a4f7177bdf2fa5cb2721a0b9b6dc7`.
