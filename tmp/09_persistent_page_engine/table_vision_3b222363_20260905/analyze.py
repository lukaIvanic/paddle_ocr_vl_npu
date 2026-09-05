"""Audit actual serving results, retaining numerical differences as evidence."""
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sys

EXPERIMENT = Path(__file__).resolve().parent
ROOT = EXPERIMENT / (sys.argv[1] if len(sys.argv) > 1 else "")
BASE = EXPERIMENT.parent / "table_cpu_image_f0f9df06_20260905/development"
records = [json.loads(line) for line in (ROOT / "development/results.jsonl").read_text().splitlines()]
reference = {r["request_id"]: r for r in map(json.loads, (BASE / "results.jsonl").read_text().splitlines())}
summary = json.loads((ROOT / "development/summary.json").read_text())
assert len(records) == len({r["request_id"] for r in records}) == 100
assert {r["request_id"] for r in records} == reference.keys()
assert hashlib.sha256((ROOT / "development/tables.jsonl").read_bytes()).hexdigest() == "1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85"
assert summary["requested_request_count"] == 100
assert summary["failed_request_count"] == summary["unsent_request_count"] == 0
active = peak = 0
for _, change in sorted([(r["dispatch_offset_s"], 1) for r in records]
                        + [(r["completion_offset_s"], -1) for r in records]):
    active += change
    peak = max(peak, active)
    assert 0 <= active <= 2
assert active == 0 and peak == 2
different, timing, device = [], Counter(), Counter()
for r in records:
    a, b = reference[r["request_id"]]["service_result"]["response"], r["service_result"]["response"]
    assert r["status"] == "ok" and b["stop_reason"] == "eos"
    assert a["crop_size"] == b["crop_size"] and a["input_tokens"] == b["input_tokens"]
    for stage, field in (("vision", "real_vision_tokens"), ("text_prefill", "real_text_tokens")):
        assert a[stage][field] == b[stage][field]
    if a["token_ids"] != b["token_ids"]:
        different.append({"request_id": r["request_id"],
                          "baseline_text": a.get("text"), "candidate_text": b.get("text"),
                          "id_edits": [{"kind": tag, "baseline_offset": i, "candidate_offset": j,
                                        "baseline_ids": a["token_ids"][i:i2], "candidate_ids": b["token_ids"][j:j2]}
                                       for tag, i, i2, j, j2 in SequenceMatcher(None, a["token_ids"], b["token_ids"], autojunk=False).get_opcodes()
                                       if tag != "equal"]})
    timing.update(b["timing_s"])
    device.update(b["device_stage_s"])
log = (ROOT / "host_npu6_monitor.log").read_text()
host_pid, container_pid = map(int, re.search(r"NSpid:\s+(\d+)\s+(\d+)", log).groups())
assert container_pid == json.loads((ROOT / "service.json").read_text())["worker_pid"]
samples = []
for block in re.split(r"(?=^2026-\d\d-\d\dT)", log, flags=re.M):
    if "Chip Count" in block:
        samples.append((datetime.fromisoformat(block.splitlines()[0]).timestamp(),
                        set(map(int, re.findall(r"Process id:(\d+)", block)))))
begin = summary["actual_start_epoch_s"]
end = begin + summary["run_wall_s"]
assert samples[0][0] < begin and samples[-1][0] > end
during = [p for t, p in samples if begin <= t <= end]
assert during and all(p == {host_pid} for p in during)
report = {"completed_tables_per_s": summary["successful_completion_qps"],
          "latency_s": summary["latency_s"], "run_wall_s": summary["run_wall_s"],
          "performance_target_pass": summary["successful_completion_qps"] >= 3 and summary["latency_s"]["p95"] < 2,
          "all_100_eos": True, "native_token_identical": 100 - len(different),
          "differences": different, "input_sizes_and_real_tokens_match": True,
          "peak_client_outstanding": peak, "during_measurement_ownership_samples": len(during),
          "only_owned_pid_in_samples": True, "timing_totals_s": dict(timing), "device_totals_s": dict(device)}
(ROOT / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(report, indent=2, ensure_ascii=False))
