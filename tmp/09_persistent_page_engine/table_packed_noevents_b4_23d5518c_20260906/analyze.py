"""Post-run B4 admission comparison; never imported by serving code."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
control = None
report = {}
monitor = ROOT / "host_npu6_monitor.log"
snapshots = []
if monitor.exists():
    for block in re.split(r"(?=^2026-\d\d-\d\dT)", monitor.read_text(), flags=re.M):
        if "Chip Count" in block:
            snapshots.append((datetime.fromisoformat(block.splitlines()[0]).timestamp(),
                              set(map(int,re.findall(r"Process id:(\d+)",block)))))
for name, cap in (("client",4), ("c5",5)):
    path = ROOT / name
    if not (path/"summary.json").exists():
        continue
    summary = json.loads((path/"summary.json").read_text())
    rows = sorted(map(json.loads,(path/"results.jsonl").read_text().splitlines()),key=lambda r:r["sequence"])
    assert len(rows) == 100 and len({r["sequence"] for r in rows}) == 100
    assert hashlib.sha256((path/"tables.jsonl").read_bytes()).hexdigest() == "1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85"
    assert summary["failed_request_count"] == summary["unsent_request_count"] == 0
    active = peak = 0
    for _, delta in sorted([(r["dispatch_offset_s"],1) for r in rows]+[(r["completion_offset_s"],-1) for r in rows]):
        active += delta
        peak = max(peak,active)
        assert 0 <= active <= cap
    assert active == 0 and peak == cap
    responses = [r["service_result"]["response"] for r in rows]
    stops, hist, stages = Counter(), Counter(), Counter()
    for response in responses:
        stops[response["stop_reason"]] += 1
        hist.update(response["scheduling_metrics"]["launched_decode_iterations_by_active_slots"])
        stages.update(response["timing_s"])
    if control is None:
        control = (summary,rows,responses)
    assert summary["api_configuration"] == control[0]["api_configuration"]
    assert summary["api_configuration"]["batch_size"] == 4
    for old, new in zip(control[2],responses):
        assert (old["crop_size"],old["input_tokens"],old["vision"]["real_vision_tokens"]) == (new["crop_size"],new["input_tokens"],new["vision"]["real_vision_tokens"])
    assert [r["request_id"] for r in rows] == [r["request_id"] for r in control[1]]
    begin, end = summary["actual_start_epoch_s"],summary["actual_start_epoch_s"]+summary["run_wall_s"]
    during = [pids for stamp,pids in snapshots if begin <= stamp <= end]
    ownership = bool(during and snapshots[0][0] < begin and snapshots[-1][0] > end and all(pids == {2015280} for pids in during))
    calls = sum(value/int(slots) for slots,value in hist.items())
    report[name] = {"client_cap":cap, "physical_decode_batch":4,
        "request_completion_qps":summary["completion_qps"], "p95_s":summary["latency_s"]["p95"],
        "run_wall_s":summary["run_wall_s"], "stop_reasons":dict(stops),
        "native_ids_identical_to_c4":sum(a["token_ids"]==b["token_ids"] for a,b in zip(control[2],responses)),
        "launched_request_iteration_histogram":dict(hist),
        "estimated_launched_slot_utilization":sum(hist.values())/(4*calls),
        "timing_sums_not_additive":dict(stages), "errors":0, "unsent":0,
        "sampled_ownership_ok":ownership, "ownership_samples":len(during),
        "second_milestone_development_pass":summary["completion_qps"] >= 5 and summary["latency_s"]["p95"] < 3}
(ROOT/"analysis.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
