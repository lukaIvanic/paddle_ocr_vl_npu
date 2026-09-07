"""NPU smoke, paired selected-crop timings and unchanged full-page evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

from run_pixel_cap_ablation import execute, dump
from run_vision_timing_production import build_command, validate


def traces(output):
    result = {}
    for line in (output / "generation_trace.jsonl").open():
        row = json.loads(line)
        if row["request_id"] in result:
            raise ValueError("duplicate replay request")
        result[row["request_id"]] = row
    return result


def check(run, reference_run, reference, selection, max_pixels, expected_pages, smoke=False):
    output = run / "output"
    summary = json.loads((output / "run_summary_shard_00.json").read_text())
    assert summary["completed"] == expected_pages and summary["failed"] == 0
    assert summary["processor_max_pixels"] == max_pixels
    assert summary["streaming"]["layout_source"] == "frozen_live_layout_selective_crop_replay"
    assert summary["streaming"]["requests_admitted_by_phase"].get("layout", 0) == 0
    original = traces(reference_run / "output")
    actual = traces(output)
    pages = {json.loads(line)["image"] for line in (output / "progress_shard_00.jsonl").open()}
    expected = {r["request_id"] for r in selection["crops"] if r["page"] in pages}
    assert set(actual) == expected, "unexpected or missing crop inference"
    image_id = json.loads((Path(reference["model"]) / "config.json").read_text())["image_token_id"]
    for key, row in actual.items():
        for field in ("page", "phase", "block_index", "block_type", "bbox", "image_sha256", "chat_prompt"):
            assert row[field] == original[key][field], (key, field)
        assert row["prompt_token_ids"].count(image_id) * 4 * 196 <= max_pixels
        if max_pixels == reference["processor_max_pixels"]:
            assert row["prompt_token_ids"] == original[key]["prompt_token_ids"], "baseline token inputs changed"
    affected_pages = {r["page"] for r in selection["crops"]}
    for name in pages - affected_pages:
        file = f"{Path(name).stem}.md"
        assert (output / "predictions" / file).read_bytes() == (reference_run / "output/predictions" / file).read_bytes()
    if not smoke:
        validate(output, reference, expected_pages)
    report = {"passed": True, "pages": expected_pages, "inferred_crops": len(actual),
              "unaffected_pages_markdown_exact": len(pages-affected_pages),
              "pipeline_wall_s_selected_replay_only": summary["pipeline_wall_s"]}
    dump(run / "replay_validation.json", report)
    print("REPLAY_VALIDATED", json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    root, reference_run = args.root.resolve(), args.reference_run.resolve()
    reference = json.loads((reference_run / "output/run_summary_shard_00.json").read_text())
    manifest = root / "frozen/manifest.json"
    frozen = json.loads(manifest.read_text())
    selection = json.loads((root / "frozen/selection.json").read_text())
    assert frozen["baseline_reconstruction_exact_pages"] == 1651
    assert frozen["reference_run"] == str(reference_run)
    (root / "pid.txt").write_text(str(os.getpid()) + "\n")
    (root / "commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True))
    (root / "visible_device.txt").write_text(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "") + "\n")
    code = 1
    try:
        with Path(".runtime_cache/11_mineru_2_5_pro_inference/serving_validation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            cases = [("control", reference["processor_max_pixels"]), ("capped", frozen["candidate_max_pixels"])]
            if not args.skip_smoke:
                cases.insert(0, ("smoke", frozen["candidate_max_pixels"]))
            for label, maximum in cases:
                run = root / label
                run.mkdir(exist_ok=False)
                is_smoke = label == "smoke"
                count = len(json.loads((root / "frozen/smoke_dataset.json").read_text())) if is_smoke else 1651
                command = build_command(reference, run / "output", count, maximum)
                command[command.index("--warmup-pages") + 1] = "0"
                command += ["--crop-replay-manifest", str(manifest)]
                if is_smoke:
                    command[command.index("--dataset-json") + 1] = str(root / "frozen/smoke_dataset.json")
                print(f"INFERENCE_START {label} max_pixels={maximum}", flush=True)
                execute(command, run)
                check(run, reference_run, reference, selection, maximum, count, smoke=is_smoke)
                (run / "exit_code.txt").write_text("0\n")
                print(f"INFERENCE_FINISH {label}", flush=True)
        for label in ("control", "capped"):
            run = root / label
            os.environ.update(RUN_ROOT=str(run), DATASET_JSON=reference["dataset_json"], LIMIT="1651")
            print(f"EVALUATION_START {label}", flush=True)
            execute(["bash", "11_mineru_2_5_pro_inference/run_serving_accuracy.sh"], run, "evaluation_launcher.log")
            print(f"EVALUATION_FINISH {label}", flush=True)
        code = 0
    finally:
        (root / "exit_code.txt").write_text(str(code) + "\n")


if __name__ == "__main__":
    main()
