"""Audit the matched diagnostic only. No oracle result can qualify the goal."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent / "run_37e42bc0"
load = lambda p: json.loads(p.read_text())
log = (ROOT / "host_npu6_monitor.log").read_text()
mapping_file = ROOT / "ownership_pid_mapping.txt"
mapping = log + (mapping_file.read_text() if mapping_file.exists() else "")
samples = [(datetime.fromisoformat(b.splitlines()[0]).timestamp(), set(map(int, re.findall(r"Process id:(\d+)", b)))) for b in re.split(r"(?=^2026-\d\d-\d\dT)", log, flags=re.M) if "Chip Count" in b]
report, records, configs = {"non_qualifying_oracle": True}, {}, {}
for mode in ("control", "policy"):
    directory = ROOT / mode
    s = load(directory / "summary.json")
    service = load(ROOT / (mode + "_service.json"))
    configs[mode] = s["api_configuration"]
    assert configs[mode] == service["configuration"]
    oracle = configs[mode]["diagnostic_admission_oracle"]
    assert oracle["non_qualifying"] and oracle["enabled"] == (mode == "policy")
    assert configs[mode]["batch_size"] == 4
    assert hashlib.sha256((directory / "tables.jsonl").read_bytes()).hexdigest() == "1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85"
    rows = list(map(json.loads, (directory / "results.jsonl").read_text().splitlines()))
    records[mode] = {r["request_id"]: r for r in rows}
    assert len(rows) == len(records[mode]) == 100
    assert s["failed_request_count"] == s["unsent_request_count"] == 0
    active = peak = 0
    for _, change in sorted([(r["dispatch_offset_s"], 1) for r in rows] + [(r["completion_offset_s"], -1) for r in rows]):
        active += change
        peak = max(peak, active)
        assert 0 <= active <= 3
    assert active == 0 and peak == 3
    for r in rows:
        assert r["status"] == "ok" and r["service_result"]["response"]["stop_reason"] == "eos"
    pid = int(re.search(r"NSpid:\s+(\d+)\s+" + str(service["worker_pid"]) + r"\b", mapping)[1])
    begin = s["actual_start_epoch_s"]
    end = begin + s["run_wall_s"]
    during = [p for t, p in samples if begin <= t <= end]
    assert samples[0][0] < begin and samples[-1][0] > end
    assert during and all(p == {pid} for p in during)
    report[mode] = {"tables_per_s": s["successful_completion_qps"], "latency_s": s["latency_s"], "all_100_eos": True, "peak_outstanding": peak, "host_pid": pid, "ownership_samples": len(during)}
for key in ("decode_vocab", "token_selection", "preprocessor", "batch_size", "decode_optimization", "max_new_tokens", "cache_length", "vision_attention_weight_padding", "max_prefill_interruptions"):
    assert configs["control"][key] == configs["policy"][key], key
assert configs["control"]["text_decode"]["cache_key_fields"] == configs["policy"]["text_decode"]["cache_key_fields"]
differences, deltas = [], []
for key, old in records["control"].items():
    new = records["policy"][key]
    a, b = old["service_result"]["response"], new["service_result"]["response"]
    assert a["crop_size"] == b["crop_size"] and a["input_tokens"] == b["input_tokens"]
    assert a["vision"]["real_vision_tokens"] == b["vision"]["real_vision_tokens"]
    if a["token_ids"] != b["token_ids"]:
        differences.append(key)
    deltas.append({"id": key, "control_s": old["latency_s"], "policy_s": new["latency_s"], "change_s": new["latency_s"]-old["latency_s"]})
report["native_identical"] = 100-len(differences)
report["changed_token_streams"] = differences
report["latency_deltas"] = sorted(deltas, key=lambda r: r["change_s"])
report["deferrals"] = [json.loads(line.split("ORACLE_RELEASE ",1)[1]) for line in (ROOT / "policy_server.log").read_text().splitlines() if line.startswith("ORACLE_RELEASE ")]
(ROOT / "comparison.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k != "latency_deltas"}, indent=2))
