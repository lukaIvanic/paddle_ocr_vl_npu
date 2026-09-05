"""Audit real C3/C2 runs of non-blocking CPU preparation."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
EXPECTED = "1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


samples = []
for block in re.split(r"(?=^2026-\d\d-\d\dT)", (ROOT / "host_npu6_monitor.log").read_text(), flags=re.M):
    if "Chip Count" in block:
        samples.append((datetime.fromisoformat(block.splitlines()[0]).timestamp(),
                        set(map(int, re.findall(r"Process id:(\d+)", block)))))
report = {}
for name, count, pid, baseline in [
    ("c3", 3, 1760664, ROOT.parent / "table_prefill_buckets_aecebe0d_20260905/bucketed"),
    ("c2", 2, 1770509, ROOT.parent / "table_closed_loop_random100_seed1_3a745ba_20260903/b2"),
]:
    directory = ROOT / name
    data = rows(directory / "results.jsonl")
    reference = {r["request_id"]: r for r in rows(baseline / "results.jsonl")}
    summary = json.loads((directory / "summary.json").read_text())
    assert hashlib.sha256((directory / "tables.jsonl").read_bytes()).hexdigest() == EXPECTED
    assert len(data) == len({r["request_id"] for r in data}) == 100
    assert {r["request_id"] for r in data} == reference.keys()
    assert summary["requested_request_count"] == 100
    assert summary["unsent_request_count"] == summary["failed_request_count"] == 0
    active = peak = 0
    for _, delta in sorted([(r["dispatch_offset_s"], 1) for r in data]
                           + [(r["completion_offset_s"], -1) for r in data]):
        active += delta
        peak = max(peak, active)
        assert 0 <= active <= count
    assert peak == count and active == 0
    begin = summary["actual_start_epoch_s"]
    end = begin + summary["run_wall_s"]
    during = [p for t, p in samples if begin <= t <= end]
    assert samples[0][0] < begin and samples[-1][0] > end
    assert during and all(p == {pid} for p in during)
    timing, device = Counter(), Counter()
    different = []
    for r in data:
        a = reference[r["request_id"]]["service_result"]["response"]
        b = r["service_result"]["response"]
        assert r["status"] == "ok" and b["stop_reason"] == "eos"
        assert a["crop_size"] == b["crop_size"] and a["input_tokens"] == b["input_tokens"]
        assert a["vision"]["real_vision_tokens"] == b["vision"]["real_vision_tokens"]
        assert a["text_prefill"]["real_text_tokens"] == b["text_prefill"]["real_text_tokens"]
        if a["token_ids"] != b["token_ids"]:
            different.append(r["request_id"])
        timing.update(b["timing_s"])
        device.update(b["device_stage_s"])
    report[name] = {
        "baseline": str(baseline.relative_to(ROOT.parent)),
        "completed_tables_per_s": summary["successful_completion_qps"],
        "latency_s": summary["latency_s"], "run_wall_s": summary["run_wall_s"],
        "peak_client_in_flight": peak, "all_eos": True,
        "native_token_identical": 100 - len(different), "different_request_ids": different,
        "input_sizes_and_real_tokens_match": True,
        "timing_totals_s": dict(timing), "device_totals_s": dict(device),
        "during_measurement_ownership_samples": len(during), "only_owned_pid_in_samples": True,
        "target_pass": summary["successful_completion_qps"] >= 3 and summary["latency_s"]["p95"] < 2,
        "tail": [{"id": r["request_id"], "latency_s": r["latency_s"],
                  "interruptions": r["service_result"]["response"]["scheduling_metrics"]["decode_other_prefill_count"]}
                 for r in sorted(data, key=lambda r: -r["latency_s"])[:10]],
    }
(ROOT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
for name, result in report.items():
    print(name, result["completed_tables_per_s"], result["latency_s"], "identical", result["native_token_identical"])
