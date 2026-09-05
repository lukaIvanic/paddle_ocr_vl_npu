"""Audit both unsuccessful control-overhead experiments against saved serving."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "table_vision_3b222363_20260905/cache_reload/development"
reference = {r["request_id"]: r["service_result"]["response"]
             for r in map(json.loads, (BASE / "results.jsonl").read_text().splitlines())}
for name, instrumented, compact in (
    ("table_events_447176de_20260905", False, False),
    ("table_compact_2b0fc9f2_20260905", True, True),
):
    run = ROOT / name
    summary = json.loads((run / "development/summary.json").read_text())
    records = list(map(json.loads, (run / "development/results.jsonl").read_text().splitlines()))
    service = json.loads((run / "service.json").read_text())
    assert hashlib.sha256((run / "development/tables.jsonl").read_bytes()).hexdigest() == "1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85"
    assert len(records) == len({r["request_id"] for r in records}) == 100
    assert set(reference) == {r["request_id"] for r in records}
    assert summary["failed_request_count"] == summary["unsent_request_count"] == 0
    assert summary["api_configuration"]["decode_device_timing"] == instrumented
    assert summary["api_configuration"].get("compact_decode_control", False) == compact
    for row in records:
        a, b = reference[row["request_id"]], row["service_result"]["response"]
        assert row["status"] == "ok" and b["stop_reason"] == "eos"
        assert a["token_ids"] == b["token_ids"]
        assert a["crop_size"] == b["crop_size"] and a["input_tokens"] == b["input_tokens"]
        assert a["vision"]["real_vision_tokens"] == b["vision"]["real_vision_tokens"]
        assert a["text_prefill"]["real_text_tokens"] == b["text_prefill"]["real_text_tokens"]
    active = peak = 0
    for _, delta in sorted([(r["dispatch_offset_s"], 1) for r in records]
                           + [(r["completion_offset_s"], -1) for r in records]):
        active += delta
        peak = max(peak, active)
        assert 0 <= active <= 2
    assert active == 0 and peak == 2
    log = (run / "host_npu6_monitor.log").read_text()
    host_pid, container_pid = map(int, re.search(r"NSpid:\s+(\d+)\s+(\d+)", log).groups())
    assert container_pid == service["worker_pid"]
    samples = [(datetime.fromisoformat(b.splitlines()[0]).timestamp(),
                set(map(int, re.findall(r"Process id:(\d+)", b))))
               for b in re.split(r"(?=^2026-\d\d-\d\dT)", log, flags=re.M) if "Chip Count" in b]
    begin = summary["actual_start_epoch_s"]
    end = begin + summary["run_wall_s"]
    during = [p for t, p in samples if begin <= t <= end]
    assert samples[0][0] < begin and samples[-1][0] > end
    assert during and all(p == {host_pid} for p in during)
    decode_time = service["summary"]["timing_s"]["decode_model_and_argmax_device"]
    assert (decode_time is not None) == instrumented
    if not instrumented:
        assert service["summary"]["rates"]["effective_device_tok_per_s"] is None
    report = {"completed_tables_per_s": summary["successful_completion_qps"],
              "latency_s": summary["latency_s"], "run_wall_s": summary["run_wall_s"],
              "performance_target_pass": summary["successful_completion_qps"] >= 3 and summary["latency_s"]["p95"] < 2,
              "all_100_eos": True, "native_token_identical": 100,
              "inputs_and_real_token_counts_match": True, "peak_client_outstanding": peak,
              "ownership_samples_during_measurement": len(during),
              "only_own_pid_in_samples": True, "decode_device_s_including_warmup": decode_time,
              "graph_calls_including_warmup": service["summary"]["graph_calls"]}
    (run / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(name, json.dumps(report))
