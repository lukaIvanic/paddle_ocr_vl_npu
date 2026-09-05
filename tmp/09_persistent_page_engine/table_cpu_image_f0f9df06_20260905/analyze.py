"""Audit the full-runtime B2 normalization/image-decode candidate."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "table_cpu_poll_02584647_20260905/c2"
data = [json.loads(line) for line in (ROOT / "development/results.jsonl").read_text().splitlines()]
reference = {r["request_id"]: r for r in map(json.loads, (BASE / "results.jsonl").read_text().splitlines())}
summary = json.loads((ROOT / "development/summary.json").read_text())
assert len(data) == len({r["request_id"] for r in data}) == 100
assert {r["request_id"] for r in data} == reference.keys()
assert hashlib.sha256((ROOT / "development/tables.jsonl").read_bytes()).hexdigest() == "1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85"
assert summary["requested_request_count"] == 100
assert summary["failed_request_count"] == summary["unsent_request_count"] == 0
active = peak = 0
for _, change in sorted([(r["dispatch_offset_s"], 1) for r in data]
                        + [(r["completion_offset_s"], -1) for r in data]):
    active += change
    peak = max(peak, active)
    assert 0 <= active <= 2
assert active == 0 and peak == 2
different, timing, device = [], Counter(), Counter()
for r in data:
    a, b = reference[r["request_id"]]["service_result"]["response"], r["service_result"]["response"]
    assert r["status"] == "ok" and b["stop_reason"] == "eos"
    assert a["crop_size"] == b["crop_size"] and a["input_tokens"] == b["input_tokens"]
    assert a["vision"]["real_vision_tokens"] == b["vision"]["real_vision_tokens"]
    assert a["text_prefill"]["real_text_tokens"] == b["text_prefill"]["real_text_tokens"]
    if a["token_ids"] != b["token_ids"]:
        different.append(r["request_id"])
    assert "cpu_image_decode" in b["timing_s"]
    timing.update(b["timing_s"])
    device.update(b["device_stage_s"])
log = (ROOT / "host_npu6_monitor.log").read_text()
assert re.search(r"NSpid:\s+1775914\s+2470812", (ROOT / "ownership_observation.txt").read_text())
samples = []
for block in re.split(r"(?=^2026-\d\d-\d\dT)", log, flags=re.M):
    if "Chip Count" in block:
        samples.append((datetime.fromisoformat(block.splitlines()[0]).timestamp(),
                        set(map(int, re.findall(r"Process id:(\d+)", block)))))
begin = summary["actual_start_epoch_s"]
end = begin + summary["run_wall_s"]
assert samples[0][0] < begin and samples[-1][0] > end
during = [p for t, p in samples if begin <= t <= end]
assert during and all(p == {1775914} for p in during)
report = {"completed_tables_per_s": summary["successful_completion_qps"],
          "latency_s": summary["latency_s"], "run_wall_s": summary["run_wall_s"],
          "target_pass": summary["successful_completion_qps"] >= 3 and summary["latency_s"]["p95"] < 2,
          "all_100_eos": True, "native_token_identical": 100 - len(different),
          "different_request_ids": different, "input_sizes_and_real_tokens_match": True,
          "peak_client_outstanding": peak, "during_measurement_ownership_samples": len(during),
          "only_owned_pid_in_samples": True, "timing_totals_s": dict(timing), "device_totals_s": dict(device)}
(ROOT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
