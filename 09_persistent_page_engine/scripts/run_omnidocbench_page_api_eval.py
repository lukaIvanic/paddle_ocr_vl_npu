#!/usr/bin/env python3
"""Run and officially score OmniDocBench through the full-page OCR API.

Product users need only the CLI in ``parse_args``. The script sends independent
page requests, saves Markdown predictions, runs the committed robust evaluator,
calculates page-weighted CDM, and prints the official three-part Overall score.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import urllib.parse
import urllib.request


HERE = Path(__file__).resolve().parent
PAGE_CLIENT = HERE / "run_omnidocbench_page_api.py"
EVAL_WRAPPER = HERE / "run_omnidocbench_eval.py"
CDM_RUNNER = HERE / "run_cdm_from_matched_formulas.py"

EXPECTED_EVALUATOR_COMMIT = "2b161d010d2e3aff77a0edef359ea3a6411d23cd"
EXPECTED_DATASET_JSON_SHA256 = (
    "a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496"
)
EXPECTED_IMAGE_COUNT = 1651
EXPECTED_IMAGES_SHA256 = (
    "58feeb96c60fcfab12ba4348c4e093ceaf1b707658dbfd0e08c24d7821d4c221"
)

REFERENCE = {
    "pages": 1651,
    "pages_per_s": 1.9511916431145628,
    "text_edit_distance": 0.05071430060872228,
    "text_score": 0.9492856993912777,
    "formula_edit_distance": 0.09032563564247499,
    "formula_sample_cdm": 0.9705318877551026,
    "formula_page_cdm": 0.9740841005676619,
    "table_sample_teds": 0.9305174944340554,
    "table_page_teds": 0.9444293305373284,
    "table_page_structure_teds": 0.9687623943678403,
    "reading_order_edit_distance": 0.14030146339439298,
    "official_overall": 0.955933043498756,
}


# =============================================================================
# PRODUCT TEAM INTERFACE
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    product = parser.add_argument_group("product benchmark")
    product.add_argument(
        "--api-url", default="http://127.0.0.1:8766/v1/pages"
    )
    product.add_argument("--dataset-json", type=Path, required=True)
    product.add_argument("--images-dir", type=Path, required=True)
    product.add_argument("--evaluator-root", type=Path, required=True)
    product.add_argument("--output-dir", type=Path, required=True)
    product.add_argument("--http-workers", type=int, default=64)
    product.add_argument("--request-timeout-s", type=float, default=3600.0)
    product.add_argument(
        "--score-only",
        action="store_true",
        help="Rescore saved generation/predictions without calling the API.",
    )

    advanced = parser.add_argument_group("advanced controls")
    advanced.add_argument("--offset", type=int, default=0)
    advanced.add_argument(
        "--limit", type=int,
        help="Evaluate a bounded page subset. The default is all remaining pages.",
    )
    advanced.add_argument("--match-workers", type=int, default=24)
    advanced.add_argument("--teds-workers", type=int, default=12)
    advanced.add_argument("--page-timeout-s", type=float, default=120.0)
    advanced.add_argument("--fallback-timeout-s", type=float, default=180.0)
    advanced.add_argument("--teds-timeout-s", type=float, default=120.0)
    advanced.add_argument("--cdm-workers", type=int)
    advanced.add_argument(
        "--drain-server", action=argparse.BooleanOptionalAction, default=True,
        help="Drain the inference worker after the benchmark (default: true).",
    )
    advanced.add_argument(
        "--skip-image-fingerprint",
        action="store_true",
        help="Check paths and JSON, but skip hashing all image bytes.",
    )
    advanced.add_argument(
        "--allow-evaluator-mismatch",
        action="store_true",
        help="Continue with a different or source-modified evaluator checkout.",
    )
    return parser.parse_args()


# =============================================================================
# INTERNAL VALIDATION AND ORCHESTRATION
# Product users should not edit below this line.
# =============================================================================

def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _red_warning(message: str) -> None:
    print(f"\033[1;31mWARNING: {message}\033[0m", file=sys.stderr, flush=True)


def _run(command: list[str], *, cwd: Path, log_path: Path, stage: str) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[{stage}] START", flush=True)
    print("[command] " + " ".join(map(str, command)), flush=True)
    started = time.perf_counter()
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        assert process.stdout is not None
        while True:
            chunk = os.read(process.stdout.fileno(), 64 * 1024)
            if not chunk:
                break
            log.write(chunk)
            log.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        return_code = process.wait()
    wall_s = time.perf_counter() - started
    print(f"[{stage}] END exit={return_code} wall_s={wall_s:.3f}", flush=True)
    if return_code != 0:
        raise RuntimeError(
            f"{stage} failed with exit {return_code}; see {log_path}"
        )
    return wall_s


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def _validate_evaluator(root: Path, allow_mismatch: bool) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not (root / "pdf_validation.py").is_file():
        raise FileNotFoundError(f"invalid evaluator root: {root}")
    try:
        commit = _git_output(root, "rev-parse", "HEAD")
        source_status = _git_output(root, "status", "--porcelain", "--", "src")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(f"evaluator is not a readable Git checkout: {root}") from exc
    exact = commit == EXPECTED_EVALUATOR_COMMIT and not source_status
    result = {
        "root": str(root),
        "commit": commit,
        "expected_commit": EXPECTED_EVALUATOR_COMMIT,
        "source_status": source_status,
        "exact": exact,
    }
    print(
        f"EVALUATOR_CHECK exact={exact} commit={commit} "
        f"source_clean={not bool(source_status)}",
        flush=True,
    )
    if not exact:
        _red_warning(
            "the evaluator checkout differs from the validated commit or has "
            "source changes; scores might not be comparable"
        )
        if not allow_mismatch:
            raise RuntimeError(
                "use the pinned evaluator or pass --allow-evaluator-mismatch"
            )
    return result


def _validate_runtime() -> dict[str, str]:
    required = ("pdflatex", "kpsewhich", "magick", "gs")
    paths = {name: shutil.which(name) or "" for name in required}
    missing = [name for name, path in paths.items() if not path]
    if missing:
        raise RuntimeError(
            "missing evaluator runtime tools: " + ", ".join(missing)
            + "; run setup_omnidocbench_eval_runtime.sh and source "
            "omnidocbench_eval_env.sh"
        )
    print(
        "RUNTIME_CHECK PASS "
        + " ".join(f"{name}={path}" for name, path in paths.items()),
        flush=True,
    )
    return paths


def _validate_api(api_url: str) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(api_url)
    ready_url = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, "/ready", "", "")
    )
    with urllib.request.urlopen(ready_url, timeout=10) as response:
        payload = json.loads(response.read())
    if payload.get("ready") is not True:
        raise RuntimeError(f"page API is not ready: {payload}")
    print(
        f"API_CHECK PASS url={api_url} worker_pid={payload.get('worker_pid')}",
        flush=True,
    )
    return payload


def _dataset_manifest(
    dataset_json: Path,
    images_dir: Path,
    *,
    hash_images: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    print(
        "DATASET_CHECK_START checking OmniDocBench.json and referenced images; "
        "inference starts after this check",
        flush=True,
    )
    dataset_bytes = dataset_json.read_bytes()
    pages = json.loads(dataset_bytes)
    names = [Path(page["page_info"]["image_path"]).name for page in pages]
    if len(names) != len(set(names)):
        raise ValueError("duplicate image basenames in OmniDocBench.json")

    total_bytes = 0
    aggregate = hashlib.sha256()
    for index, name in enumerate(sorted(names), start=1):
        path = images_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        total_bytes += size
        if hash_images:
            entry = {"path": name, "bytes": size, "sha256": _sha256_file(path)}
            aggregate.update(
                (json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
                .encode()
            )
        if index % 100 == 0 or index == len(names):
            print(f"checked_images={index}/{len(names)}", flush=True)

    json_hash = hashlib.sha256(dataset_bytes).hexdigest()
    images_hash = aggregate.hexdigest() if hash_images else None
    exact = (
        len(names) == EXPECTED_IMAGE_COUNT
        and json_hash == EXPECTED_DATASET_JSON_SHA256
        and (not hash_images or images_hash == EXPECTED_IMAGES_SHA256)
    )
    manifest = {
        "dataset_json_sha256": json_hash,
        "referenced_image_count": len(names),
        "referenced_images_total_bytes": total_bytes,
        "referenced_images_aggregate_sha256": images_hash,
        "image_hashing_enabled": hash_images,
        "matches_repository_authority": exact,
    }
    print(
        f"DATASET_CHECK_RESULT matches={exact} images={len(names)}/1651 "
        f"json_sha256={json_hash} images_sha256={images_hash}",
        flush=True,
    )
    if not exact:
        _red_warning(
            "dataset inputs differ from the validated OmniDocBench v1.6 "
            "fingerprints; the run will continue"
        )
    return pages, manifest


def _select_pages(
    pages: list[dict[str, Any]], offset: int, limit: int | None
) -> list[dict[str, Any]]:
    if offset < 0 or offset > len(pages):
        raise ValueError(f"invalid offset: {offset}")
    selected = pages[offset:] if limit is None else pages[offset : offset + limit]
    if not selected:
        raise ValueError("selected page set is empty")
    if limit is not None and len(selected) != limit:
        raise ValueError(f"requested {limit} pages, got {len(selected)}")
    return selected


def _available_memory_gib() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return max(1, int(line.split()[1]) // 1024 // 1024)
    except OSError:
        pass
    return 2


def _cdm_workers(requested: int | None) -> int:
    if requested is not None:
        if requested <= 0:
            raise ValueError("--cdm-workers must be positive")
        return requested
    return max(1, min(96, os.cpu_count() or 1, _available_memory_gib() // 2))


def _write_eval_config(
    path: Path,
    *,
    ground_truth: Path,
    predictions: Path,
    match_workers: int,
    teds_workers: int,
) -> None:
    def yaml_string(value: Path) -> str:
        return json.dumps(str(value))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""end2end_eval:
  metrics:
    text_block:
      metric: [Edit_dist]
    display_formula:
      metric: [Edit_dist]
    table:
      metric: [TEDS, Edit_dist]
      teds_workers: {teds_workers}
    reading_order:
      metric: [Edit_dist]
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: {yaml_string(ground_truth)}
    prediction:
      data_path: {yaml_string(predictions)}
    match_method: quick_match
    match_workers: {match_workers}
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
""",
        encoding="utf-8",
    )


def _score(
    *,
    metric_path: Path,
    stages_path: Path,
    cdm_summary_path: Path,
    generation_summary_path: Path,
    page_count: int,
) -> dict[str, Any]:
    metric = _json(metric_path)
    stages = _json(stages_path)
    cdm = _json(cdm_summary_path)
    evaluated = _json(Path(cdm["evaluated_samples"]))

    by_page: dict[str, list[float]] = defaultdict(list)
    for sample in evaluated:
        by_page[str(sample["img_id"])].append(float(sample["metric"]["CDM"]))
    sample_cdm = sum(v for values in by_page.values() for v in values) / len(
        evaluated
    )
    page_cdm = sum(sum(values) / len(values) for values in by_page.values()) / len(
        by_page
    )
    if abs(sample_cdm - float(cdm["scores"]["CDM"]["all"])) >= 1e-12:
        raise AssertionError("sample CDM does not match the CDM runner summary")

    text_edit = float(metric["text_block"]["page"]["Edit_dist"]["ALL"])
    formula_edit = float(metric["display_formula"]["page"]["Edit_dist"]["ALL"])
    sample_teds = float(metric["table"]["all"]["TEDS"]["all"])
    page_teds = float(metric["table"]["page"]["TEDS"]["ALL"])
    structure_teds = float(
        metric["table"]["page"]["TEDS_structure_only"]["ALL"]
    )
    reading_edit = float(metric["reading_order"]["page"]["Edit_dist"]["ALL"])
    overall = ((1.0 - text_edit) + page_cdm + page_teds) / 3.0

    teds_debug = stages["metrics"]["table"]["TEDS"]
    cdm_debug = cdm["debug"]
    result: dict[str, Any] = {
        "pages": page_count,
        "text_block": {
            "edit_distance": text_edit,
            "score": 1.0 - text_edit,
        },
        "display_formula": {
            "edit_distance": formula_edit,
            "sample_cdm": sample_cdm,
            "page_cdm": page_cdm,
            "sample_count": len(evaluated),
            "page_count": len(by_page),
        },
        "table": {
            "sample_teds": sample_teds,
            "page_teds": page_teds,
            "page_structure_teds": structure_teds,
            "teds_timeouts": int(teds_debug["timeout_case_count"]),
            "teds_errors": int(teds_debug["error_case_count"])
            + int(teds_debug.get("exception_case_count", 0)),
        },
        "reading_order": {"edit_distance": reading_edit},
        "official_overall": overall,
        "official_overall_percent": 100.0 * overall,
        "page_match_fallbacks": stages["page_match"]["fallbacks"],
        "cdm_timeouts": int(cdm_debug["timeout_case_count"]),
        "cdm_errors": int(cdm_debug["exception_case_count"]),
    }
    if generation_summary_path.is_file():
        generation = _json(generation_summary_path)
        result["generation"] = {
            key: generation[key]
            for key in (
                "count", "wall_s", "pages_per_s", "mean_response_s",
                "max_response_s",
            )
            if key in generation
        }
    if page_count == EXPECTED_IMAGE_COUNT:
        result["reference_910b2"] = REFERENCE
        comparable = {
            "pages_per_s": result.get("generation", {}).get("pages_per_s"),
            "text_edit_distance": text_edit,
            "formula_edit_distance": formula_edit,
            "formula_sample_cdm": sample_cdm,
            "formula_page_cdm": page_cdm,
            "table_sample_teds": sample_teds,
            "table_page_teds": page_teds,
            "table_page_structure_teds": structure_teds,
            "reading_order_edit_distance": reading_edit,
            "official_overall": overall,
        }
        result["signed_delta_vs_910b2"] = {
            key: None if value is None else value - float(REFERENCE[key])
            for key, value in comparable.items()
        }
    return result


def _summary_markdown(summary: dict[str, Any]) -> str:
    generation = summary.get("generation", {})
    table = summary["table"]
    formula = summary["display_formula"]
    lines = [
        "# OmniDocBench full-page API benchmark",
        "",
        f"- Pages: {summary['pages']}",
    ]
    if generation:
        lines.extend(
            [
                f"- Generation wall: {generation['wall_s']:.3f} s",
                f"- Throughput: {generation['pages_per_s']:.6f} pages/s",
            ]
        )
    lines.extend(
        [
            f"- Text-block Edit distance: "
            f"{summary['text_block']['edit_distance']:.6f}",
            f"- Official text score (1 - Edit distance): "
            f"{summary['text_block']['score']:.6f}",
            f"- Display-formula Edit distance: "
            f"{formula['edit_distance']:.6f}",
            f"- Formula page-CDM: {formula['page_cdm']:.6f}",
            f"- Formula sample-CDM: {formula['sample_cdm']:.6f}",
            f"- Table Page-TEDS: {table['page_teds']:.6f}",
            f"- Table sample-TEDS: {table['sample_teds']:.6f}",
            f"- Table structure-only Page-TEDS: "
            f"{table['page_structure_teds']:.6f}",
            f"- Reading-order Edit distance: "
            f"{summary['reading_order']['edit_distance']:.6f}",
            f"- Official Overall: {summary['official_overall_percent']:.4f}%",
            f"- TEDS timeouts/errors: "
            f"{table['teds_timeouts']}/{table['teds_errors']}",
            f"- CDM timeouts/errors: "
            f"{summary['cdm_timeouts']}/{summary['cdm_errors']}",
            "",
            "Official Overall = mean(text score, formula page-CDM, "
            "table Page-TEDS).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.http_workers <= 0 or args.match_workers <= 0 or args.teds_workers <= 0:
        raise ValueError("worker counts must be positive")
    dataset_json = args.dataset_json.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    evaluator_root = args.evaluator_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for support in (PAGE_CLIENT, EVAL_WRAPPER, CDM_RUNNER):
        if not support.is_file():
            raise FileNotFoundError(f"missing committed support script: {support}")
    evaluator_info = _validate_evaluator(
        evaluator_root, args.allow_evaluator_mismatch
    )
    runtime_info = _validate_runtime()
    pages, dataset_info = _dataset_manifest(
        dataset_json,
        images_dir,
        hash_images=not args.skip_image_fingerprint,
    )
    selected = _select_pages(pages, args.offset, args.limit)
    subset_path = output_dir / "OmniDocBench_subset.json"
    _write_json(subset_path, selected)
    _write_json(output_dir / "dataset_manifest.json", dataset_info)
    _write_json(output_dir / "evaluator_manifest.json", evaluator_info)
    _write_json(output_dir / "runtime_paths.json", runtime_info)

    generation_dir = output_dir / "generation"
    predictions_dir = generation_dir / "predictions"
    generation_summary = generation_dir / "run_summary.json"
    if args.score_only:
        predictions = list(predictions_dir.glob("*.md"))
        if len(predictions) != len(selected):
            raise RuntimeError(
                f"score-only expected {len(selected)} predictions, "
                f"found {len(predictions)} in {predictions_dir}"
            )
        print(
            f"SCORE_ONLY predictions={len(predictions)} directory={predictions_dir}",
            flush=True,
        )
    else:
        existing = list(predictions_dir.glob("*.md")) if predictions_dir.exists() else []
        if existing:
            raise RuntimeError(
                f"refusing to mix a new run with {len(existing)} saved predictions; "
                "use a new --output-dir or --score-only"
            )
        _validate_api(args.api_url)
        generation_command = [
            sys.executable,
            str(PAGE_CLIENT),
            "--api-url", args.api_url,
            "--dataset-json", str(dataset_json),
            "--images-dir", str(images_dir),
            "--offset", str(args.offset),
            "--limit", str(len(selected)),
            "--http-workers", str(args.http_workers),
            "--timeout-s", str(args.request_timeout_s),
            "--output-dir", str(generation_dir),
        ]
        if args.drain_server:
            generation_command.append("--drain-server")
        _run(
            generation_command,
            cwd=HERE.parent.parent,
            log_path=output_dir / "generation.log",
            stage="generation",
        )

    evaluation_root = output_dir / "evaluation"
    evaluation_work = evaluation_root / "work"
    result_dir = evaluation_work / "result"
    cdm_dir = evaluation_root / "cdm_native"
    for stale_dir in (result_dir, cdm_dir):
        if stale_dir.exists():
            print(f"CLEAR_STALE_SCORE_ARTIFACTS directory={stale_dir}", flush=True)
            shutil.rmtree(stale_dir)
    config_path = evaluation_work / "config.yaml"
    _write_eval_config(
        config_path,
        ground_truth=subset_path,
        predictions=predictions_dir,
        match_workers=args.match_workers,
        teds_workers=args.teds_workers,
    )
    eval_command = [
        sys.executable,
        str(EVAL_WRAPPER),
        "--config", "config.yaml",
        "--evaluator-root", str(evaluator_root),
        "--match-workers", str(args.match_workers),
        "--teds-workers", str(args.teds_workers),
        "--page-timeout-sec", str(args.page_timeout_s),
        "--fallback-timeout-sec", str(args.fallback_timeout_s),
        "--fallback-latex-timeout-sec", "30",
        "--teds-timeout-sec", str(args.teds_timeout_s),
    ]
    evaluation_wall = _run(
        eval_command,
        cwd=evaluation_work,
        log_path=output_dir / "evaluation.log",
        stage="matching-and-teds",
    )

    matched = result_dir / "predictions_quick_match_display_formula_result.json"
    metric = result_dir / "predictions_quick_match_metric_result.json"
    stages = result_dir / "predictions_quick_match_stage_execution.json"
    for artifact in (matched, metric, stages):
        if not artifact.is_file():
            raise FileNotFoundError(f"evaluator did not create: {artifact}")

    cdm_command = [
        sys.executable,
        str(CDM_RUNNER),
        "--input", str(matched),
        "--output-dir", str(cdm_dir),
        "--evaluator-root", str(evaluator_root),
        "--workers", str(_cdm_workers(args.cdm_workers)),
        "--save-name", "predictions_quick_match_cdm",
    ]
    cdm_wall = _run(
        cdm_command,
        cwd=HERE.parent.parent,
        log_path=output_dir / "cdm.log",
        stage="formula-cdm",
    )
    cdm_summary = cdm_dir / "cdm_run_summary.json"
    if not cdm_summary.is_file():
        raise FileNotFoundError(f"CDM did not create: {cdm_summary}")

    summary = _score(
        metric_path=metric,
        stages_path=stages,
        cdm_summary_path=cdm_summary,
        generation_summary_path=generation_summary,
        page_count=len(selected),
    )
    summary["evaluation_wall_s"] = evaluation_wall
    summary["cdm_wall_s"] = cdm_wall
    summary["dataset_manifest"] = dataset_info
    summary["evaluator_manifest"] = evaluator_info
    _write_json(output_dir / "benchmark_summary.json", summary)
    markdown = _summary_markdown(summary)
    (output_dir / "benchmark_summary.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown, flush=True)
    print(
        "PAGE_API_OFFICIAL_SUMMARY "
        f"pages={len(selected)} "
        f"pages_per_s={summary.get('generation', {}).get('pages_per_s')} "
        f"text_edit={summary['text_block']['edit_distance']:.6f} "
        f"page_cdm={summary['display_formula']['page_cdm']:.6f} "
        f"page_teds={summary['table']['page_teds']:.6f} "
        f"overall={summary['official_overall_percent']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
