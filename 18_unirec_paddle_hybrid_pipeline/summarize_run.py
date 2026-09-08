"""Summarize hybrid measurements without counting cooperative pauses as decode."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def distribution(values):
    values = sorted(values)
    def percentile(q):
        position = (len(values) - 1) * q
        lo = int(position)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)
    return {"count": len(values), "sum": sum(values), "mean": statistics.mean(values),
            "p50": percentile(.5), "p99": percentile(.99), "max": max(values)}


def summarize(run):
    engines = {}
    for name, engine in run["engines"].items():
        s = engine["summary"]
        raw = s["raw_decode_token_slots"]
        useful = s["effective_decode_tokens"]
        active = s.get("active_decode_token_slots", useful)
        scheduled = run["action_wall_s"][f"{name}.decode"]
        if name == "paddle":
            seconds = s["timing_s"]["decode_model_and_argmax_device"]
            basis = "device events: decode model and argmax"
        else:
            seconds = s["decode_s"]
            basis = "host elapsed: graph, token selection and blocking CPU token read"
        engines[name] = {
            "capacity": engine["capacity"],
            "ready_capacity": engine.get("ready_capacity"),
            "prefill_request_counts": engine.get("prefill_request_counts"),
            "cpu_preparation": engine.get("cpu_preparation"),
            "graph_calls": s.get("decode_iterations", s.get("graph_calls")),
            "raw_token_slots": raw,
            "active_token_slots": active,
            "useful_decode_tokens": useful,
            "active_slot_utilization": active / raw if raw else None,
            "useful_slot_utilization": useful / raw if raw else None,
            "mean_active_slots": engine["capacity"] * active / raw if raw else None,
            "scheduled_decode_wall_s": scheduled,
            "scheduled_raw_tok_s": raw / scheduled if scheduled else None,
            "scheduled_useful_tok_s": useful / scheduled if scheduled else None,
            "execution_timing_basis": basis,
            "execution_s": seconds,
            "execution_raw_tok_s": raw / seconds if seconds else None,
            "execution_useful_tok_s": useful / seconds if seconds else None,
            "prefill_tokens": engine["prefill_tokens"],
            "prefill_device_s": engine["prefill_device_s"],
        }
        tokens, spans = engine["prefill_tokens"], engine["prefill_device_s"]
        if name == "paddle":
            engines[name]["vision_useful_token_fraction"] = tokens["vision_real"] / tokens["vision_physical"]
            engines[name]["prefill_rates"] = {
                "vision_real_patches_s": tokens["vision_real"] / spans["vision_prefill"],
                "vision_physical_patches_s": tokens["vision_physical"] / spans["vision_prefill"],
                "text_input_tokens_s": tokens["text_input"] / spans["text_prefill"],
            }
        elif engine.get("vision_runtime"):
            engines[name]["text_prefill_useful_token_fraction"] = tokens["text_real_source"] / tokens["text_physical_source"]
            vision = engine["vision_runtime"]
            physical_rows = sum(b["batch_size"] * vision["bucket_calls"][b["key"]]
                                for b in vision["buckets"])
            real_rows = sum(vision["bucket_real_rows"].values())
            engines[name]["vision_bucket_row_utilization"] = real_rows / physical_rows if physical_rows else None
            engines[name]["prefill_rates"] = {
                "vision_real_output_tokens_per_envelope_s": tokens["text_real_source"] / spans["vision_encode_envelope"],
                "text_real_source_tokens_per_envelope_s": tokens["text_real_source"] / spans["text_prefill_envelope"],
                "text_physical_source_tokens_per_envelope_s": tokens["text_physical_source"] / spans["text_prefill_envelope"],
            }
    return {
        "pages": run["pages"], "processing_wall_s": run["wall_s"],
        "processing_pages_s": run["pages_per_s"], "setup_s": run["setup_s"],
        "including_setup_pages_s": run["pages"] / (run["wall_s"] + run["setup_s"]),
        "action_wall_s": run["action_wall_s"], "engines": engines,
        "page_preparation": run.get("page_preparation"),
        "peak_torch_allocated_GiB": run["peak_torch_allocated_bytes"] / 2**30,
        "peak_torch_reserved_GiB": run["peak_torch_reserved_bytes"] / 2**30,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    report = summarize(json.loads((args.run_dir / "run_summary.json").read_text()))
    stops, routes = Counter(), Counter()
    seen = set()
    lengths = {}
    for line in (args.run_dir / "recognition_trace.jsonl").open():
        record = json.loads(line)
        if record["request_id"] in seen:
            raise ValueError(f"Duplicate request: {record['request_id']}")
        seen.add(record["request_id"])
        stops[f"{record['model']}.{record['stop_reason']}"] += 1
        routes[f"{record['model']}.{record['prompt']}"] += 1
        # UniRec stores its decoder-start token; Paddle starts with the
        # prefill-generated token. Both traces retain EOS when produced.
        count = len(record["token_ids"]) - (record["model"] == "unirec")
        lengths.setdefault(f"{record['model']}.{record['prompt']}", []).append(count)
    report.update(crops=len(seen), stops=dict(stops), routes=dict(routes),
                  generated_token_lengths={k: distribution(v) for k, v in lengths.items()})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
