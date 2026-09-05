"""Summarize three real-request embedding captures, not serving performance."""
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
capture = json.loads((ROOT / "profile/capture.json").read_text())
assert capture["full_request_warmups"] == 5
assert capture["profiled_forwards"] == 3
assert all(row["pixel_shape"] == [1, 3840, 3, 14, 14]
           and row["grid"] == [[1, 48, 80]] for row in capture["observations"])
profiles = []
for path in sorted((ROOT / "profile").rglob("kernel_details.csv")):
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 15
    assert sum(r["Type"] == "Conv2D" for r in rows) == 1
    assert sum(r["Type"] == "ResizeBilinearV2" for r in rows) == 1
    groups = defaultdict(float)
    for row in rows:
        assert row["Device_id"] == "6"
        groups[row["Type"]] += float(row["Duration(us)"])
    profiles.append({"source": str(path.relative_to(ROOT)),
                     "kernel_sum_us": sum(groups.values()), "by_type_us": dict(groups)})
assert len(profiles) == 3
records = list(map(json.loads, (ROOT / "client/results.jsonl").read_text().splitlines()))
assert len(records) == 8
outputs = [row["service_result"]["response"] for row in records]
assert all(row["status"] == "ok" for row in records)
assert all(out["stop_reason"] == "eos" for out in outputs)
assert len({tuple(out["token_ids"]) for out in outputs}) == 1
average = {kind: mean(p["by_type_us"][kind] for p in profiles)
           for kind in profiles[0]["by_type_us"]}
total = mean(p["kernel_sum_us"] for p in profiles)
result = {"diagnostic_only": True, "profiles": profiles,
          "mean_kernel_sum_us": total, "mean_by_type_us": average,
          "convolution_share": average["Conv2D"]/total,
          "all_eight_eos_native_identical": True,
          "note": "Profiling synchronizes and exports during requests. These HTTP latencies are not performance results."}
(ROOT / "analysis.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k != "profiles"}, indent=2))
