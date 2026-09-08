"""CPU-only, post-run accuracy report. Reads native IDs; never encodes outputs."""
import argparse
from collections import defaultdict
import difflib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "09_persistent_page_engine"))
sys.path.insert(0, str(ROOT / "09_persistent_page_engine/scripts"))
from run_omnidocbench_table_api import _score


def load(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def prepare(run, base_dir, gt, extra_runs):
    from pipeline.layout_output import normalize_recognition_text

    paths = sorted(run.glob("budget*/qps*/measured/results.jsonl"))
    reference = load(run / "budget4096/qps1/measured/results.jsonl")
    reference.sort(key=lambda x: x["sequence"])
    assert len(reference) == 100
    expected = [(x["sequence"], x["request_id"]) for x in reference]
    groups = {str(p.relative_to(run).parent.parent): load(p) for p in paths}
    for extra in extra_runs:
        assert json.loads((extra / "status.json").read_text())["status"] == "screening_complete"
        groups.update({extra.name + "/" + str(p.relative_to(extra).parent.parent): load(p)
                       for p in sorted(extra.glob("budget*/qps*/measured/results.jsonl"))})
    custom = load(base_dir / "table_poisson_frontier_screen1000_be691de1_20260907/b2/qps1/measured/results.jsonl")
    groups["custom_b2_qps1_first100"] = [x for x in custom if x["sequence"] <= 100]
    unique, records, views = {}, [], {}
    for name, rows in groups.items():
        rows.sort(key=lambda x: x["sequence"])
        assert [(x["sequence"], x["request_id"]) for x in rows] == expected, name
        views[name] = []
        for x in rows:
            response = x["service_result"]["response"]
            assert x["status"] == "ok" and isinstance(response["token_ids"], list)
            raw = response.get("raw_text", response["text"])
            html = normalize_recognition_text("table", raw)
            if name.startswith("custom"):
                assert html == response["text"], "Custom postprocessing contract changed"
            source = gt[x["request_id"]]
            key = (x["request_id"], html)
            if key not in unique:
                unique[key] = len(records)
                records.append(dict(request_id=f"variant_{len(records)}", page_name=source["page_name"],
                    annotation_index=source["annotation_index"], gt_html=source["gt_html"], pred_html=html))
            views[name].append(dict(request_id=x["request_id"], native_ids=response["token_ids"],
                raw=raw, html=html, score_index=unique[key]))
    return records, views


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--evaluator-root", type=Path, default=Path("/workspace/repos/OmniDocBench_eval"))
    parser.add_argument("--teds-workers", type=int, default=4)
    parser.add_argument("--teds-timeout-s", type=float, default=120)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--score-prepared", action="store_true")
    parser.add_argument("--extra-run", type=Path, action="append", default=[])
    args = parser.parse_args()
    run = Path(__file__).resolve().parent
    args.output_dir = run / "accuracy"
    args.output_dir.mkdir(exist_ok=True)
    base_dir = ROOT / "tmp/09_persistent_page_engine"
    gt = {x["request_id"]: x for x in load(base_dir / "table_b1_latency_full_04fbc8e/client/tables.jsonl")}
    prepared = args.output_dir / "prepared.json"
    if args.score_prepared:
        payload = json.loads(prepared.read_text())
        records, views = payload["records"], payload["views"]
    else:
        records, views = prepare(run, base_dir, gt, args.extra_run)
        prepared.write_text(json.dumps(dict(records=records, views=views), ensure_ascii=False)+"\n")
        if args.prepare_only:
            print(f"Prepared {len(records)} unique scoring pairs across {len(views)} runs")
            return
    scores = _score(records, args)
    per_variant = scores["per_table"]
    controls = views["custom_b2_qps1_first100"]
    summaries, differences = {}, []
    for name, items in views.items():
        pages, structures = defaultdict(list), defaultdict(list)
        matches = dict(native_ids=0, raw=0, html=0)
        values, shape_values = [], []
        for item, control in zip(items, controls):
            score = per_variant[item["score_index"]]
            cs = per_variant[control["score_index"]]
            page = gt[item["request_id"]]["page_name"]
            pages[page].append(score["TEDS"])
            structures[page].append(score["TEDS_structure_only"])
            values.append(score["TEDS"])
            shape_values.append(score["TEDS_structure_only"])
            for key in matches:
                matches[key] += item[key] == control[key]
            if item["native_ids"] != control["native_ids"]:
                differences.append(dict(run=name, request_id=item["request_id"],
                    raw_equal=item["raw"] == control["raw"], html_equal=item["html"] == control["html"],
                    teds=score["TEDS"], custom_teds=cs["TEDS"], delta=score["TEDS"]-cs["TEDS"],
                    raw_diff=[dict(kind=t, custom=control["raw"][i:j], vllm=item["raw"][a:b])
                        for t,i,j,a,b in difflib.SequenceMatcher(None,control["raw"],item["raw"],autojunk=False).get_opcodes() if t != "equal"]))
        mean = lambda xs: sum(xs)/len(xs)
        summaries[name] = dict(count=len(items), matches_custom=matches, sample_TEDS=mean(values),
            page_TEDS=mean([mean(v) for v in pages.values()]),
            sample_structure_TEDS=mean(shape_values), page_structure_TEDS=mean([mean(v) for v in structures.values()]))
    report = dict(note="Matched first100 subset, not corpus-wide. Identical GT/prediction pairs scored once; native IDs untouched.",
        unique_scored_pairs=len(records), teds_errors=scores["teds_error_count"], teds_timeouts=scores["teds_timeout_count"],
        custom_source="table_poisson_frontier_screen1000_be691de1_20260907/b2/qps1/measured/results.jsonl", runs=summaries, differences=differences)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "differences"}, indent=2))


if __name__ == "__main__":
    main()
