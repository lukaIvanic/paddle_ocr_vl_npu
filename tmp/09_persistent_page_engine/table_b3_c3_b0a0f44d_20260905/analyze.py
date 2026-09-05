"""Audit actual B3/C3 serving against the matched B4/C3 development run."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "table_b4_c3_c4_1e32b233_20260905"
load = lambda path: json.loads(path.read_text())
summary = load(ROOT / "development/summary.json")
service = load(ROOT / "service.json")
rows = list(map(json.loads, (ROOT / "development/results.jsonl").read_text().splitlines()))
old = {r["request_id"]: r for r in map(json.loads, (BASE / "c3/results.jsonl").read_text().splitlines())}
assert hashlib.sha256((ROOT / "development/tables.jsonl").read_bytes()).hexdigest() == "1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85"
assert len(rows) == len({r["request_id"] for r in rows}) == 100
assert summary["failed_request_count"] == summary["unsent_request_count"] == 0
cfg = summary["api_configuration"]
assert cfg == service["configuration"] and cfg["batch_size"] == 3
control = load(BASE / "service.json")["configuration"]
for key in ("recognizer_model", "dtype", "decode_backend", "decode_optimization", "decode_vocab", "token_selection", "decode_attention", "decode_cache_update", "cache_length", "max_new_tokens", "preprocessor", "max_prefill_interruptions", "vision_attention_weight_padding", "vision_backend", "text_backend", "text_packing", "vision_packing", "decode_device_timing", "compact_decode_control"):
    assert cfg[key] == control[key], key
assert cfg["vision_prefill"]["buckets"] == control["vision_prefill"]["buckets"]
assert cfg["text_prefill"]["buckets"] == control["text_prefill"]["buckets"]
active = peak = 0
for _, delta in sorted([(r["dispatch_offset_s"], 1) for r in rows] + [(r["completion_offset_s"], -1) for r in rows]):
    active += delta
    peak = max(peak, active)
    assert 0 <= active <= 3
assert active == 0 and peak == 3
differences = []
for row in rows:
    a = old[row["request_id"]]["service_result"]["response"]
    b = row["service_result"]["response"]
    assert row["status"] == "ok" and b["stop_reason"] == "eos"
    assert a["crop_size"] == b["crop_size"] and a["input_tokens"] == b["input_tokens"]
    assert a["vision"]["real_vision_tokens"] == b["vision"]["real_vision_tokens"]
    if a["token_ids"] != b["token_ids"]:
        differences.append(row["request_id"])
log = (ROOT / "host_npu6_monitor.log").read_text()
mapping = re.search(r"NSpid:\s+(\d+)\s+" + str(service["worker_pid"]) + r"\b", log)
assert mapping
host_pid = int(mapping[1])
samples = [(datetime.fromisoformat(b.splitlines()[0]).timestamp(), set(map(int, re.findall(r"Process id:(\d+)", b)))) for b in re.split(r"(?=^2026-\d\d-\d\dT)", log, flags=re.M) if "Chip Count" in b]
begin = summary["actual_start_epoch_s"]
end = begin + summary["run_wall_s"]
during = [p for t, p in samples if begin <= t <= end]
assert samples[0][0] < begin and samples[-1][0] > end
assert during and all(p == {host_pid} for p in during)
report = {"tables_per_s": summary["successful_completion_qps"], "latency_s": summary["latency_s"], "wall_s": summary["run_wall_s"], "all_100_eos": True, "peak_outstanding": peak, "native_identical_to_b4_c3": 100-len(differences), "changed_requests": differences, "ownership_samples": len(during), "host_worker_pid": host_pid}
(ROOT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
