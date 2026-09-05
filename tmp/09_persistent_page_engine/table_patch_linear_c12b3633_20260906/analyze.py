"""Matched real-serving audit for the content-independent patch projection."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from difflib import SequenceMatcher
from pathlib import Path
import re
from statistics import mean
import sys

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
records, configurations, report = {}, {}, {}
monitor = (ROOT / "host_npu6_monitor.log").read_text()
samples = [(datetime.fromisoformat(block.splitlines()[0]).timestamp(),
            set(map(int, re.findall(r"Process id:(\d+)", block))))
           for block in re.split(r"(?=^2026-\d\d-\d\dT)", monitor, flags=re.M)
           if "Chip Count" in block]
mapping = (ROOT / "ownership_pid_mapping.txt").read_text()
for mode in ("control", "candidate"):
    path = ROOT / mode
    summary = json.loads((path / "summary.json").read_text())
    service = json.loads((ROOT / f"{mode}_service.json").read_text())
    rows = list(map(json.loads, (path / "results.jsonl").read_text().splitlines()))
    assert len(rows) == 100 and summary["failed_request_count"] == summary["unsent_request_count"] == 0
    assert hashlib.sha256((path / "tables.jsonl").read_bytes()).hexdigest() == "1f77a0233333ba8dbf01434dc7de3b6b3dee75e611e38554de47d6a29bf1ba85"
    records[mode] = {row["sequence"]: row for row in rows}
    assert len(records[mode]) == 100
    config = summary["api_configuration"]
    configurations[mode] = config
    assert config == service["configuration"]
    active = peak = 0
    for _, change in sorted([(r["dispatch_offset_s"], 1) for r in rows]
                            + [(r["completion_offset_s"], -1) for r in rows]):
        active += change
        peak = max(peak, active)
        assert 0 <= active <= 2
    assert peak == 2 and active == 0
    outputs = [r["service_result"]["response"] for r in rows]
    stops = Counter(o["stop_reason"] for o in outputs)
    pid = int(re.search(r"NSpid:\s+(\d+)\s+"+str(service["worker_pid"])+r"\b", mapping)[1])
    begin = summary["actual_start_epoch_s"]
    end = begin + summary["run_wall_s"]
    during = [pids for timestamp,pids in samples if begin <= timestamp <= end]
    assert samples[0][0] < begin and samples[-1][0] > end
    assert during and all(pids == {pid} for pids in during)
    report[mode] = {
        "tables_per_s": summary["successful_completion_qps"], "latency_s": summary["latency_s"],
        "stop_reasons": dict(stops), "peak_outstanding": peak, "npu_host_pid": pid,
        "ownership_samples": len(during),
        "mean_embedding_device_s": mean(o["device_stage_s"]["vision_embeddings"] for o in outputs),
        "mean_prefill_wall_s": mean(o["timing_s"]["vision_and_text_prefill_wall"] for o in outputs),
        "decode_device_s_lifetime_including_warmup": service["summary"]["timing_s"]["decode_model_and_argmax_device"],
        "graph_calls_including_warmup": service["summary"]["graph_calls"],
        "mean_decode_device_event_interval_ms_including_warmup": 1000 * service["summary"]["timing_s"]["decode_model_and_argmax_device"] / service["summary"]["graph_calls"],
        "device_interval_note": "Device events bracket submission/execution; this is not the sum of kernel durations. Whole-server totals include the separate warm request.",
    }
before_config, after_config = configurations["control"], configurations["candidate"]
packed_comparison = after_config["decode_optimization"] == before_config["decode_optimization"] + "_packed_mlp"
if packed_comparison:
    assert before_config["vision_linear_patch_projection"] == after_config["vision_linear_patch_projection"]
else:
    assert before_config["decode_optimization"] == after_config["decode_optimization"]
    assert not before_config["vision_linear_patch_projection"] and after_config["vision_linear_patch_projection"]
report["comparison_kind"] = "packed_mlp" if packed_comparison else "linear_patch_projection"
for field in ("decode_vocab", "token_selection", "cache_length", "max_new_tokens",
              "batch_size", "preprocessor", "vision_attention_weight_padding", "max_prefill_interruptions"):
    assert configurations["control"][field] == configurations["candidate"][field], field
differences = []
for sequence, row in records["control"].items():
    other = records["candidate"][sequence]
    assert row["request_id"] == other["request_id"]
    before, after = row["service_result"]["response"], other["service_result"]["response"]
    assert before["crop_size"] == after["crop_size"] and before["input_tokens"] == after["input_tokens"]
    assert before["vision"]["real_vision_tokens"] == after["vision"]["real_vision_tokens"]
    if before["token_ids"] != after["token_ids"]:
        differences.append({"request_id": row["request_id"], "before_raw": before["raw_text"],
                            "after_raw": after["raw_text"], "before_ids": before["token_ids"],
                            "after_ids": after["token_ids"]})
report["native_identical"] = 100-len(differences)
report["native_differences"] = differences
review = []
source = ROOT.parent / "table_b1_latency_full_04fbc8e/client/tables.jsonl"
ground_truth = {r["request_id"]: r["gt_html"] for r in
                map(json.loads, source.read_text().splitlines())}
def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]
for difference in differences:
    before, after = difference["before_raw"], difference["after_raw"]
    edits = [{"operation": tag, "before_offset": i, "after_offset": j,
              "before": before[i:ie], "after": after[j:je]}
             for tag, i, ie, j, je in SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
             if tag != "equal"]
    item = {"request_id": difference["request_id"], "character_edits": edits}
    # Offline quality inspection only. Nothing in this report is read by serving.
    if difference["request_id"] == "page_001227_table_10":
        gt = re.search(r"S750A_rp</td><td>(.*?)</td>", ground_truth[difference["request_id"]])[1]
        cells = [re.search(r"S750A_rp<fcel>(.*?)<nl>", text)[1] for text in (before, after)]
        item["cell_label"] = "S750A_rp"
        item["cell_character_edit_distance_to_gt"] = dict(zip(("control", "candidate"), [distance(cell, gt) for cell in cells]))
        item["assessment"] = "Content difference, not formatting. Equivalent projection has measured FP16 drift; first-divergence logits were not captured."
    review.append(item)
report["output_review"] = review
report["first_performance_gate_passed"] = (report["candidate"]["tables_per_s"] >= 3.0
    and report["candidate"]["latency_s"]["p95"] < 2.0
    and report["candidate"]["stop_reasons"] == {"eos": 100})
(ROOT / "comparison.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k != "native_differences"}, indent=2))
