"""Audit the full-runtime prefill-bucket comparison, without projecting E2E."""
import collections
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
EXPECTED = "1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


samples = []
for block in re.split(r"(?=^2026-\d\d-\d\dT)", (ROOT / "host_npu6_monitor.log").read_text(), flags=re.M):
    if "Chip Count" in block:
        samples.append((datetime.fromisoformat(block.splitlines()[0]).timestamp(),
                        set(map(int, re.findall(r"Process id:(\d+)", block)))))

result = {}
records = {}
for name, pid in (("control", 1748367), ("bucketed", 1751056)):
    directory = ROOT / name
    summary = json.loads((directory / "summary.json").read_text())
    rows = read_rows(directory / "results.jsonl")
    assert hashlib.sha256((directory / "tables.jsonl").read_bytes()).hexdigest() == EXPECTED
    assert len(rows) == len({r["request_id"] for r in rows}) == 100
    assert summary["requested_request_count"] == summary["request_count"] == 100
    assert summary["failed_request_count"] == summary["unsent_request_count"] == 0
    active = peak = 0
    for _, delta in sorted([(r["dispatch_offset_s"], 1) for r in rows]
                           + [(r["completion_offset_s"], -1) for r in rows]):
        active += delta
        peak = max(peak, active)
        assert 0 <= active <= 3
    assert active == 0 and peak == 3
    begin = summary["actual_start_epoch_s"]
    end = begin + summary["run_wall_s"]
    during = [p for t, p in samples if begin <= t <= end]
    assert samples[0][0] <= begin and samples[-1][0] >= end
    assert during and all(p == {pid} for p in during)
    devices = collections.Counter()
    vision = collections.Counter()
    text = collections.Counter()
    real_vision = real_text = 0
    for r in rows:
        response = r["service_result"]["response"]
        assert r["status"] == "ok" and response["stop_reason"] == "eos"
        devices.update(response["device_stage_s"])
        vision[response["vision"]["physical_vision_tokens"]] += 1
        text[response["text_prefill"]["physical_text_tokens"]] += 1
        real_vision += response["vision"]["real_vision_tokens"]
        real_text += response["text_prefill"]["real_text_tokens"]
    records[name] = {r["request_id"]: r for r in rows}
    result[name] = {
        "completed_tables_per_s": summary["successful_completion_qps"],
        "latency_s": summary["latency_s"], "request_count": len(rows),
        "all_eos": True, "peak_client_outstanding": peak,
        "device_stage_totals_s": dict(devices),
        "real_vision_tokens": real_vision, "real_text_tokens": real_text,
        "physical_vision_tokens": sum(k * v for k, v in vision.items()),
        "physical_text_tokens": sum(k * v for k, v in text.items()),
        "vision_bucket_counts": dict(vision), "text_bucket_counts": dict(text),
        "npu6_owned_host_pid": pid, "during_measurement_monitor_samples": len(during),
        "only_owned_pid_in_monitor_samples": True,
        "target_pass": summary["successful_completion_qps"] >= 3 and summary["latency_s"]["p95"] < 2,
    }

assert records["control"].keys() == records["bucketed"].keys()
differences = []
tail = []
for key, before in records["control"].items():
    after = records["bucketed"][key]
    a, b = before["service_result"]["response"], after["service_result"]["response"]
    for field in ("crop_size", "input_tokens", "projected_image_tokens"):
        assert a[field] == b[field], (key, field)
    assert a["vision"]["real_vision_tokens"] == b["vision"]["real_vision_tokens"]
    assert a["text_prefill"]["real_text_tokens"] == b["text_prefill"]["real_text_tokens"]
    if a["token_ids"] != b["token_ids"]:
        differences.append(key)
    tail.append({"request_id": key, "control_s": before["latency_s"],
                 "bucketed_s": after["latency_s"], "tokens": len(b["token_ids"])})
result["audit"] = {"selection_sha256": EXPECTED, "same_input_sizes_and_real_tokens": True,
                   "native_token_identical": 100 - len(differences), "different_request_ids": differences,
                   "sampling_limitation": "host process snapshots every approximately five seconds; not continuous kernel tracing",
                   "tail": sorted(tail, key=lambda r: -r["bucketed_s"])[:12]}
(ROOT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({name: {key: value for key, value in run.items()
                         if key in ("completed_tables_per_s", "latency_s", "target_pass")}
                  for name, run in result.items() if name != "audit"}, indent=2))
print("AUDIT", json.dumps(result["audit"]))
