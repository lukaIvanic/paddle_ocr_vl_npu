"""CPU-only cross-run evidence rollup; never imported by serving code."""
import contextlib
import copy
import io
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
B2 = ROOT.parent / "table_packed_noevents_23d5518c_20260906"
B5 = ROOT.parent / "table_packed_noevents_b5c5_b0f1cac7_20260906"


def audit(root, batch, worker, qps, p95):
    sys.argv = [str(B2 / "audit.py"), "--root", str(root),
                "--batch-size", str(batch), "--max-in-flight", str(batch),
                "--worker-pid", str(worker), "--target-qps", str(qps),
                "--target-p95", str(p95)]
    with contextlib.redirect_stdout(io.StringIO()):
        state = runpy.run_path(str(B2 / "audit.py"))
    return state


def read(path):
    return json.loads(path.read_text())


def inference_contract(config):
    """Remove only setup measurements; retain all configuration/cache keys."""
    config = copy.deepcopy(config)
    for stage in ("vision_prefill", "text_prefill"):
        for key in ("compile_wrapper_total_s", "compile_first_call_total_s"):
            del config[stage][key]
        for bucket in config[stage]["per_bucket"].values():
            for key in ("compile_wrapper_s", "compile_first_call_s"):
                del bucket[key]
    for key in ("setup_collected", "frozen_objects", "setup_wall_s"):
        del config["setup_gc"][key]
    return config


first = audit(B2, 2, 1994482, 3, 2)
second = audit(B5, 5, 2037247, 5, 3)
replacement = audit(ROOT, 5, 3228768, 5, 3)
names = ("client", "seed2", "validation1000_a", "validation1000_b")
assert all(first["report"]["runs"][name]["qualifying_timing"] for name in names)
assert all(second["report"]["runs"][name]["qualifying_timing"] for name in names[:3])
assert not second["report"]["runs"]["validation1000_b"]["qualifying_timing"]
assert replacement["report"]["runs"]["validation1000_b"]["qualifying_timing"]

old_config = read(B5 / "client/summary.json")["api_configuration"]
new_config = read(ROOT / "validation1000_b/summary.json")["api_configuration"]
assert inference_contract(old_config) == inference_contract(new_config)
assert read(ROOT / "warm/summary.json")["api_configuration"] == new_config
assert read(ROOT / "warm/summary.json")["request_count"] == 1
assert (ROOT / "warm_exit_code.txt").read_text().strip() == "0"
assert (ROOT / "client_exit_code.txt").read_text().strip() == "0"
assert (ROOT / "server_exit_code.txt").read_text().strip() == "0"
service = read(ROOT / "service.json")
assert service["summary"]["requests"] == 1001
assert service["configuration"] == new_config
assert "No process in device." in (ROOT / "ownership_released.log").read_text()
assert "OWNED_PIDS_RELEASED" in (ROOT / "ownership_released.log").read_text()
cleanup = (ROOT / "final_cleanup.log").read_text()
assert "SERVER_WORKER_AND_MONITOR_STOPPED" in cleanup
assert "No process in device." in cleanup
for run_root, run_names in ((B2, names), (B5, names[:3])):
    preceding_end = 0
    for name in run_names:
        summary = read(run_root / name / "summary.json")
        assert summary["actual_start_epoch_s"] > preceding_end
        preceding_end = summary["actual_start_epoch_s"] + summary["run_wall_s"]
assert read(ROOT / "validation1000_b/summary.json")["actual_start_epoch_s"] > preceding_end

compare = replacement["compare_outputs"]
comparison = compare(B5 / "validation1000_a", ROOT / "validation1000_b")
assert comparison["count"] == 1000
assert not comparison["input_differences"] and not comparison["stop_differences"]
# Numerical differences are inspection data, not an automatic parity gate.
(ROOT / "replacement_output_comparison.json").write_text(
    json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")

report = {
    "milestone1": first["report"]["runs"],
    "milestone2": {**{n: second["report"]["runs"][n] for n in names[:3]},
                   "validation1000_b": replacement["report"]["runs"]["validation1000_b"]},
    "retained_invalid_attempt": str(B5 / "validation1000_b"),
    "replacement_inference_contract_identical": True,
    "replacement_service_requests_including_warmup": 1001,
    "owned_jobs_stopped_and_npu_released": True,
    "configuration_comparison_excludes_only": [
        "vision/text prefill compile-wrapper and first-call timing measurements",
        "setup GC collected/frozen object counts and collection wall time"],
    "replacement_output_comparison": {k: v for k, v in comparison.items() if k != "differences"},
    "scope": "Closed-loop ordinary serving on physical NPU6; not an offered-QPS test.",
    "caps": "All1000 latencies retained; 988 EOS and 12 unchanged KV4096 caps. EOS-only QPS qualifies.",
}
(ROOT / "final_audit.json").write_text(json.dumps(report, indent=2) + "\n")
for milestone in ("milestone1", "milestone2"):
    for name, result in report[milestone].items():
        print(milestone, name, result["request_completion_qps"],
              result["eos_completion_qps"], result["p95_s"], result["qualifying_timing"])
print("replacement native IDs identical:", comparison["native_ids_identical"], "/1000")
