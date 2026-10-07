#!/usr/bin/env python3
"""Recheck published token accounting and compressed profiler receipts, stdlib only."""
from pathlib import Path
import csv, collections, gzip, json
root = Path(__file__).resolve().parent
d = json.loads((root / "summary.json").read_text())
assert d["status"] == "completed"
assert (root / "exit_code.txt").read_text().strip() == "0"
eos = d["batches"][0]["reference"]["token_ids"][0][-1]
samples = 0
for batch in d["batches"]:
    for variant, state in batch["variants"].items():
        assert state["status"] == "passed"
        for trial in state["samples"]:
            samples += 1
            assert trial["no_compile_in_timing"] and trial["format_match"]
            assert trial["new_graphs_during_generation"] == trial["recompile_warning_count"] == 0
            assert trial["matches_initial_variant_sequence"]
            rows = trial["token_ids"]
            assert trial["decode_tokens_excluding_prefill_and_eos"] == sum(
                len(row) - (row[-1] == eos) - (row[0] != eos) for row in rows)
            assert trial["forward_calls"] == max(map(len, rows)) - 1
            assert trial["raw_batch_slots"] == len(rows) * trial["forward_calls"]
            assert set(trial["cache_formats_after"]) == ({29} if variant.endswith("_nz") else {2})
            if not variant.endswith("_nz"):
                assert trial["token_match"] and trial["eos_count"] == 4 and trial["length_cap_hit_count"] == 0
assert samples == 50
profiles = 0
for path in root.rglob("kernel_details.csv.gz"):
    with gzip.open(path, "rt") as stream:
        rows = list(csv.DictReader(stream))
    counts = collections.Counter(row["Type"] for row in rows)
    attention = [row for row in rows if "Attention" in row["Type"]]
    conversions = [row for row in rows if row["Type"] == "TransData"]
    assert len(attention) == 192 and len({row["Name"] for row in attention}) == 24
    assert sum(n for name, n in counts.items() if name.startswith("MatMul")) == 776
    assert sum(n for name, n in counts.items() if name.startswith("ArgMax")) == 8
    assert sum(n for name, n in counts.items() if name.startswith("Scatter")) == 384
    assert len(conversions) == (384 if "_nz_" in str(path) else 0)
    assert all(row["Input Formats"] == "FRACTAL_NZ" and row["Output Formats"] == "ND" for row in conversions)
    profiles += 1
assert profiles == 10
print(json.dumps({"timed_samples_checked": samples, "full_model_profiles_checked": profiles,
                  "token_accounting_and_runtime_gates": "passed", "candidate_drift_not_hidden": True}))
