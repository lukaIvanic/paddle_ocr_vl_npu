# Receipt provenance correction — 2026-10-07

This note is additive. Existing `command.txt`, result JSONs, lane command files
and `raw_evidence.tar.gz` have not been rewritten as part of this correction.
The note also applies to the edited `command.txt` inside that archive.

The top-level `command.txt` was assembled after inference had begun from the
orchestration commands; it was not an immutable contemporaneous launch receipt.
Its initial test command incorrectly named:

```text
/usr/local/python3.12.13/bin/python3 -m unittest discover -s 11_mineru_2_5_pro_inference -p test_attention_mask_segments.py
```

The filename was subsequently replaced with `test_production_vision_attention.py`
after inspecting the actual test file and `tests.log` (three tests passed). The
initial filename and replacement are recoverable from the authoring session's
tool calls. No saved pre-edit copy of the whole file was retained. The padding
matrix command was also appended after that matrix was launched. These edits
were intended to correct/document the commands, but erased the distinction
between reconstructed documentation and a launch receipt. The edited copy is
retained only with this provenance note; it must not be represented as an
untouched execution receipt.

The per-lane `matrix/*.command.sh` and `padding_matrix/*.command.sh` files were
written by the matrix driver before its subprocess launches. Use those for the
replay invocation, together with lane logs/results and source commit. The root
capture/test command reconstruction is weaker evidence and is labelled as such.

Host load/CPU count and before/after device/job snapshots were not collected
for each historical lane. Existing run-boundary occupancy snapshots are
preserved; they cannot supply missing per-lane host measurements. See
`HISTORICAL_HOST_CONTEXT.md` for explicit availability by lane. New diagnostics
write exclusive-create command receipts and per-lane snapshots before launch
and after exit; no receipt is repaired in place.
