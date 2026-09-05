"""Matched B4 serving, varying only client concurrency three versus four."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
service = json.loads((ROOT / "service.json").read_text())
assert service["worker_pid"] == 2483736
log = (ROOT / "host_npu6_monitor.log").read_text()
assert re.search(r"NSpid:\s+1808655\s+2483736", log)
samples = [(datetime.fromisoformat(b.splitlines()[0]).timestamp(),
            set(map(int, re.findall(r"Process id:(\d+)", b))))
           for b in re.split(r"(?=^2026-\d\d-\d\dT)", log, flags=re.M) if "Chip Count" in b]
reports, sets = {}, {}
for concurrency in (3, 4):
    directory = ROOT / f"c{concurrency}"
    summary = json.loads((directory / "summary.json").read_text())
    records = list(map(json.loads, (directory / "results.jsonl").read_text().splitlines()))
    assert hashlib.sha256((directory / "tables.jsonl").read_bytes()).hexdigest() == "1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85"
    assert summary["failed_request_count"] == summary["unsent_request_count"] == 0
    assert len(records) == len({r["request_id"] for r in records}) == 100
    assert summary["api_configuration"] == service["configuration"]
    assert summary["api_configuration"]["batch_size"] == 4
    active = peak = 0
    for _, delta in sorted([(r["dispatch_offset_s"], 1) for r in records]
                           + [(r["completion_offset_s"], -1) for r in records]):
        active += delta
        peak = max(peak, active)
        assert 0 <= active <= concurrency
    assert active == 0 and peak == concurrency
    sets[concurrency] = {}
    for r in records:
        output = r["service_result"]["response"]
        assert r["status"] == "ok" and output["stop_reason"] == "eos"
        sets[concurrency][r["request_id"]] = output
    begin = summary["actual_start_epoch_s"]
    end = begin + summary["run_wall_s"]
    during = [p for t, p in samples if begin <= t <= end]
    assert samples[0][0] < begin and samples[-1][0] > end
    assert during and all(p == {1808655} for p in during)
    reports[f"c{concurrency}"] = {
        "completed_tables_per_s": summary["successful_completion_qps"],
        "latency_s": summary["latency_s"], "wall_s": summary["run_wall_s"],
        "all_100_eos": True, "peak_outstanding": peak,
        "ownership_samples_during_measurement": len(during),
        "only_owned_npu_pid_in_samples": True,
    }
assert sets[3].keys() == sets[4].keys()
for key in sets[3]:
    a, b = sets[3][key], sets[4][key]
    assert a["token_ids"] == b["token_ids"]
    assert a["crop_size"] == b["crop_size"] and a["input_tokens"] == b["input_tokens"]
    assert a["vision"]["real_vision_tokens"] == b["vision"]["real_vision_tokens"]
reports["native_token_identical"] = 100
reports["same_inputs"] = True
(ROOT / "comparison.json").write_text(json.dumps(reports, indent=2) + "\n")
print(json.dumps(reports, indent=2))
