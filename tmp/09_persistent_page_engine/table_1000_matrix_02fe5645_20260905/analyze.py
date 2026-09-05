"""Audit the matched, ordinary-decoding 1,000-request comparison (CPU only)."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
frozen = json.loads((ROOT.parents[2] / "09_persistent_page_engine/presets/table_serving_goal_validation.json").read_text())
frozen1000 = next(g for g in frozen["gates"] if g["count"] == 1000 and g["seed"] == 3)
LANES = {"b1": (1, 1), "b2": (2, 2), "b4c3": (4, 3),
         "b4c4": (4, 4), "b8": (8, 8), "b16": (16, 16)}
OWNED_HOST_WORKERS = {"b1": 1850259, "b2": 1865538,
                      "b4c3": 1873266, "b4c4": 1873266, "b8": 1901755,
                      "b16": 1912954}
reports, outputs, reference_manifest, reference_config = {}, {}, None, None
FROZEN_FIELDS = ("recognizer_model", "device", "dtype", "decode_backend",
                 "decode_optimization", "decode_vocab", "token_selection",
                 "decode_attention", "decode_cache_update", "cache_length",
                 "max_new_tokens", "vision_mlp", "vision_linear_weight_format",
                 "vision_backend", "vision_attention", "decode_device_timing",
                 "compact_decode_control", "vision_attention_weight_padding",
                 "vision_promptfa_align_128", "vision_sequence_alignment",
                 "vision_packing", "vision_prompt_fa_layout", "text_backend",
                 "text_packing", "preprocessor", "linear_weight_format",
                 "request_scheduling_metrics", "max_prefill_interruptions")
monitor = ROOT / "host_npu6_monitor.log"
samples = []
if monitor.exists():
    for block in re.split(r"(?=^2026-\d\d-\d\dT)", monitor.read_text(), flags=re.M):
        if "Chip Count" not in block:
            continue
        samples.append((datetime.fromisoformat(block.splitlines()[0]).timestamp(),
                        sorted(set(map(int, re.findall(r"Process id:(\d+)", block))))))

for lane, (batch, concurrency) in LANES.items():
    directory = ROOT / lane / "measured"
    if not (directory / "summary.json").exists():
        continue
    summary = json.loads((directory / "summary.json").read_text())
    records = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    records.sort(key=lambda row: row["sequence"])
    manifest = (directory / "tables.jsonl").read_bytes()
    reference_manifest = manifest if reference_manifest is None else reference_manifest
    assert manifest == reference_manifest, f"{lane}: mismatched input sequence"
    ids = summary["dispatch_request_ids"]
    assert ids == frozen1000["dispatch_request_ids"]
    assert hashlib.sha256(manifest).hexdigest() == frozen1000["tables_jsonl_sha256"]
    assert len(ids) == len(records) == 1000
    assert len(set(ids[:665])) == 665 and len(set(ids[665:])) == 335
    assert set(ids[665:]) <= set(ids[:665])
    assert [r["sequence"] for r in records] == list(range(1, 1001))
    assert [r["request_id"] for r in records] == ids
    assert summary["failed_request_count"] == summary["unsent_request_count"] == 0
    config = summary["api_configuration"]
    stable_config = {key: config[key] for key in FROZEN_FIELDS}
    reference_config = stable_config if reference_config is None else reference_config
    assert stable_config == reference_config, f"{lane}: non-batch model/settings changed"
    assert config["batch_size"] == batch
    assert config["max_prefill_interruptions"] is None
    active = peak = 0
    for _, delta in sorted([(r["dispatch_offset_s"], 1) for r in records]
                           + [(r["completion_offset_s"], -1) for r in records]):
        active += delta
        peak = max(peak, active)
        assert 0 <= active <= concurrency
    assert active == 0 and peak == concurrency
    outputs[lane] = [r["service_result"]["response"] for r in records]
    stops = Counter(out["stop_reason"] for out in outputs[lane])
    first_seen, repeated_differences = {}, []
    for sequence, (rid, out) in enumerate(zip(ids, outputs[lane]), 1):
        if rid in first_seen and first_seen[rid] != out["token_ids"]:
            repeated_differences.append({"sequence": sequence, "request_id": rid})
        first_seen.setdefault(rid, out["token_ids"])
    begin = summary["actual_start_epoch_s"]
    end = begin + summary["run_wall_s"]
    during = [(t, p) for t, p in samples if begin <= t <= end]
    bracketed = bool(samples and samples[0][0] < begin and samples[-1][0] > end)
    expected_worker = OWNED_HOST_WORKERS.get(lane)
    uncontaminated = bool(during and expected_worker is not None
                         and all(p == [expected_worker] for _, p in during))
    reports[lane] = {
        "decode_batch": batch, "max_in_flight": concurrency,
        "requests": len(records), "completed_tables_per_s": summary["successful_completion_qps"],
        "frozen_model_settings_match_b1": True,
        "latency_s": summary["latency_s"], "wall_s": summary["run_wall_s"],
        "errors": summary["failed_request_count"], "stop_reasons": dict(stops),
        "eos_completed_tables_per_s": stops["eos"] / summary["run_wall_s"],
        "all_requests_ended_at_eos": stops["eos"] == len(records),
        "qualifying_goal_validation": False,
        "repeated_stream_differences": repeated_differences,
        "peak_outstanding": peak, "tables_sha256": hashlib.sha256(manifest).hexdigest(),
        "npu_pids_observed_during_measurement": sorted({p for _, ps in during for p in ps}),
        "ownership_sample_count": len(during),
        "ownership_brackets_measurement": bracketed,
        "only_expected_worker_in_observed_samples": uncontaminated,
        "ownership_max_sample_gap_s": max((b[0] - a[0] for a, b in zip(during, during[1:])), default=None),
        "measurement_begin_epoch_s": begin, "measurement_end_epoch_s": end,
    }
    if "b1" in outputs:
        different = []
        for sequence, (base, current) in enumerate(zip(outputs["b1"], outputs[lane]), 1):
            assert base["crop_size"] == current["crop_size"]
            assert base["input_tokens"] == current["input_tokens"]
            assert base["vision"]["real_vision_tokens"] == current["vision"]["real_vision_tokens"]
            if base["token_ids"] != current["token_ids"]:
                different.append({"sequence": sequence, "request_id": ids[sequence - 1],
                                  "b1_tokens": len(base["token_ids"]),
                                  "lane_tokens": len(current["token_ids"]),
                                  "b1_text": base.get("raw_text", base.get("text")),
                                  "lane_text": current.get("raw_text", current.get("text"))})
        reports[lane]["same_input_shapes_and_real_tokens"] = True
        reports[lane]["native_token_differences_vs_b1"] = different

(ROOT / "analysis.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({name: {k: v for k, v in report.items()
                         if k != "native_token_differences_vs_b1"}
                  for name, report in reports.items()}, indent=2))
