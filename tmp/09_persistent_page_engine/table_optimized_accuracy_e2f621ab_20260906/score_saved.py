"""Adapt saved HTTP results to the existing process-parallel table scorer.

CPU-only. Ground truth is used exclusively after inference. Score each of the
665 source tables once, not the repeated serving-load occurrences.
"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, default=Path("/workspace/repos/OmniDocBench_eval"))
    parser.add_argument("--teds-workers", type=int, default=12)
    parser.add_argument("--teds-timeout-s", type=float, default=120)
    args = parser.parse_args()
    if args.teds_workers < 1:
        parser.error("--teds-workers must be positive")
    baseline = ROOT / "tmp/09_persistent_page_engine/table_b1_latency_full_04fbc8e/client/tables.jsonl"
    source = [json.loads(s) for s in baseline.read_text().splitlines()]
    assert len(source) == len({r["request_id"] for r in source}) == 665
    saved = sorted((json.loads(s) for s in args.results.read_text().splitlines()), key=lambda r:r["sequence"])
    unique = {}
    for r in saved:
        assert r["status"] == "ok" and r["service_result"]["http_status"] == 200
        response = r["service_result"]["response"]
        if r["request_id"] in unique:
            assert unique[r["request_id"]]["token_ids"] == response["token_ids"], r["request_id"]
        else:
            unique[r["request_id"]] = response
    assert set(unique) == {r["request_id"] for r in source}
    records = []
    for r in source:
        response = unique[r["request_id"]]
        assert list(response["crop_size"]) == r["crop_size"]
        records.append({"request_id": r["request_id"], "page_name": r["page_name"],
                        "annotation_index": r["annotation_index"], "gt_html": r["gt_html"],
                        "pred_html": response["text"], "stop_reason": response["stop_reason"]})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tables.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records))
    spec = importlib.util.spec_from_file_location("table_api_score", ROOT / "09_persistent_page_engine/scripts/run_omnidocbench_table_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scores = module._score(records, args)
    module._print_score(scores, args.teds_timeout_s)
    assert scores["teds_timeout_count"] == scores["teds_error_count"] == 0, "Incomplete TEDS evaluation; retain and retry scoring, not OCR."


if __name__ == "__main__":
    main()
