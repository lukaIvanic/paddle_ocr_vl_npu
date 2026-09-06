"""CPU-only audit of the requested eight-lane production validation sweep."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "table_packed_noevents_23d5518c_20260906/validation1000_a"
read = lambda p: json.loads(p.read_text())
reference = read(OLD / "summary.json")["api_configuration"]
manifest = (OLD / "tables.jsonl").read_bytes()
FIELDS = (
    "recognizer_model", "device", "dtype", "decode_backend", "decode_optimization",
    "decode_vocab", "token_selection", "decode_attention", "decode_cache_update",
    "cache_length", "max_new_tokens", "vision_mlp", "vision_linear_weight_format",
    "vision_backend", "vision_attention", "decode_device_timing", "compact_decode_control",
    "vision_attention_weight_padding", "vision_linear_patch_projection", "vision_promptfa_align_128",
    "vision_sequence_alignment", "vision_packing", "vision_prompt_fa_layout", "text_backend",
    "text_packing", "preprocessor", "linear_weight_format", "request_scheduling_metrics",
    "max_prefill_interruptions",
)
samples = []
for block in re.split(r"(?=^2026-\d\d-\d\dT)", (ROOT / "host_npu6_monitor.log").read_text(), flags=re.M):
    if "Chip Count" in block:
        samples.append((datetime.fromisoformat(block.splitlines()[0]).timestamp(),
                        set(map(int, re.findall(r"Process id:(\d+)", block)))))
report, base_outputs = {}, None
for name, batch, concurrency in [("b1c1",1,1),("b2c2",2,2),("b3c3",3,3),
                                 ("b4c4",4,4),("b6c6",6,6),("b7c7",7,7),("b8c7",8,7),("b8c8",8,8)]:
    lane = ROOT / name
    if not (lane / "measured/summary.json").exists():
        continue
    summary = read(lane / "measured/summary.json")
    assert summary["request_count"] == summary["requested_request_count"] == 1000
    assert summary["failed_request_count"] == summary["unsent_request_count"] == 0
    assert (lane / "measured/tables.jsonl").read_bytes() == manifest
    config = summary["api_configuration"]
    assert config["batch_size"] == batch
    assert summary["max_in_flight"] == summary["observed_max_in_flight"] == concurrency
    assert {k:config[k] for k in FIELDS} == {k:reference[k] for k in FIELDS}
    assert config["setup_gc"]["enabled"] and config["setup_gc"]["gc_remains_enabled"]
    assert config == read(lane / "warm/summary.json")["api_configuration"]
    assert read(lane / "warm/summary.json")["request_count"] == 1
    rows = sorted(map(json.loads, (lane / "measured/results.jsonl").read_text().splitlines()),
                  key=lambda r:r["sequence"])
    assert [r["sequence"] for r in rows] == list(range(1,1001))
    assert [r["request_id"] for r in rows] == summary["dispatch_request_ids"]
    active = peak = 0
    for _, delta in sorted([(r["dispatch_offset_s"],1) for r in rows] + [(r["completion_offset_s"],-1) for r in rows]):
        active += delta
        peak = max(peak, active)
        assert 0 <= active <= concurrency
    assert active == 0 and peak == concurrency
    outputs = [r["service_result"]["response"] for r in rows]
    stops = Counter(o["stop_reason"] for o in outputs)
    differences = []
    if base_outputs is None:
        base_outputs = outputs
    for i,(a,b) in enumerate(zip(base_outputs, outputs),1):
        assert all(a[k] == b[k] for k in ("input_tokens","projected_image_tokens","crop_size"))
        assert a["vision"]["real_vision_tokens"] == b["vision"]["real_vision_tokens"]
        if a["token_ids"] != b["token_ids"]:
            differences.append(i)
    owned = set(map(int,re.findall(r"Process id:(\d+)", (lane / "ownership_running.log").read_text())))
    assert len(owned) == 1
    begin = summary["actual_start_epoch_s"]
    end = begin + summary["run_wall_s"]
    during = [(t,p) for t,p in samples if begin <= t <= end]
    clean = bool(during and samples[0][0] < begin and samples[-1][0] > end
                 and all(p == owned for _,p in during))
    report[name] = dict(batch=batch, concurrency=concurrency, requests=1000,
        qps=summary["completion_qps"], latency_s=summary["latency_s"],
        eos_qps=stops["eos"]/summary["run_wall_s"], stop_reasons=dict(stops),
        errors=0, manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        optimized_contract_matches=True, actual_peak_in_flight=peak,
        same_input_shapes_and_vision_tokens=True, native_id_differences_vs_c1=differences,
        owned_host_pids=sorted(owned), ownership_samples=len(during), clean_timing=clean,
        ownership_max_gap_s=max((b[0]-a[0] for a,b in zip(during,during[1:])), default=None))
    print(name, "QPS",round(summary["completion_qps"],4),"P95",round(summary["latency_s"]["p95"],4),
          "clean",clean,"stops",dict(stops),"ID differences",len(differences))
(ROOT / "analysis.json").write_text(json.dumps(report,indent=2)+"\n")
