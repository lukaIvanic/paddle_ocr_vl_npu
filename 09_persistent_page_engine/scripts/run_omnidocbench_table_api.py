#!/usr/bin/env python3
"""Send table crops to the OCR API.

Product users need only the CLI in ``parse_args``. Use ``--images`` for normal
PNG/JPEG crops or ``--omnidocbench`` for the fixed 665-table quality check.
Everything below the PRODUCT TEAM section is internal transport and evaluation
code; it should not need product-specific edits.
"""

from __future__ import annotations

import argparse
import collections
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import math
import multiprocessing
from multiprocessing.connection import wait as wait_for_process
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image


# =============================================================================
# PRODUCT TEAM INTERFACE
# =============================================================================

CROP_TYPES = ("text", "ocr", "table", "chart", "formula", "spotting", "seal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--images", type=Path, nargs="+", metavar="IMAGE",
        help="Recognize one or more already-cropped PNG/JPEG files.",
    )
    mode.add_argument(
        "--omnidocbench", action="store_true",
        help="Run and score all OmniDocBench table annotations.",
    )

    product = parser.add_argument_group("product client")
    product.add_argument("--api-url", default="http://127.0.0.1:8765/v1/ocr")
    product.add_argument("--output-dir", type=Path, required=True)
    product.add_argument("--crop-type", choices=CROP_TYPES, default="table")
    product.add_argument("--timeout-s", type=float, default=900.0)
    product.add_argument("--http-workers", type=int, default=64)

    benchmark = parser.add_argument_group("OmniDocBench evaluation")
    benchmark.add_argument(
        "--dataset-json", type=Path,
        default=Path("/workspace/datasets/OmniDocBench/OmniDocBench.json"),
    )
    benchmark.add_argument(
        "--images-dir", type=Path,
        default=Path("/workspace/datasets/OmniDocBench/images"),
    )
    benchmark.add_argument(
        "--evaluator-root", type=Path,
        default=Path("/workspace/repos/OmniDocBench_eval"),
    )
    benchmark.add_argument("--teds-workers", type=int, default=12)
    benchmark.add_argument("--teds-timeout-s", type=float, default=120.0)
    benchmark.add_argument(
        "--score-only", action="store_true",
        help="Rescore saved tables.jsonl without rerunning OCR.",
    )

    # Kept for existing runbooks. Normal product checks leave these unchanged.
    advanced = parser.add_argument_group("advanced controls")
    advanced.add_argument("--crop-padding", type=int, default=0)
    advanced.add_argument("--offset", type=int, default=0)
    advanced.add_argument("--limit-pages", type=int)
    advanced.add_argument(
        "--drain-server", action=argparse.BooleanOptionalAction, default=None,
    )
    advanced.add_argument("--fingerprint-only", action="store_true")
    advanced.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
    )
    return parser.parse_args()


# =============================================================================
# INTERNAL TRANSPORT AND OUTPUT
# Product users should not need to edit below this line.
# =============================================================================

def _request(
    api_url: str,
    request_id: str,
    image_bytes: bytes,
    crop_type: str,
    timeout_s: float,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {"crop_type": crop_type, "request_id": request_id}
    )
    request = urllib.request.Request(
        f"{api_url}?{query}",
        data=image_bytes,
        method="POST",
        headers={"Content-Type": "image/png"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP {exc.code}: {body}") from exc


def _drain(api_url: str, timeout_s: float) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(api_url)
    url = urllib.parse.urlunparse(parsed._replace(path="/v1/drain", query=""))
    request = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read()).get("summary", {})


def _safe_name(path: Path) -> str:
    name = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in path.stem
    ).strip("_")
    return name or "image"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _custom_image_job(
    index: int,
    image_path: Path,
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any]]:
    started = time.perf_counter()
    image_bytes = image_path.read_bytes()
    response = _request(
        args.api_url,
        f"image_{index:06d}_{_safe_name(image_path)}",
        image_bytes,
        args.crop_type,
        args.timeout_s,
    )
    return index, {
        "image_path": str(image_path.resolve()),
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "crop_type": args.crop_type,
        "client_wall_s": time.perf_counter() - started,
        "response": response,
    }


def _custom_markdown(record: dict[str, Any], relative_image: str) -> str:
    response = record["response"]
    return "\n".join(
        [
            f"# {Path(record['image_path']).name}",
            "",
            f"![Input image](<{relative_image}>)",
            "",
            "## Recognition",
            "",
            str(response["text"]).rstrip(),
            "",
            "## Run information",
            "",
            f"- Crop type: {record['crop_type']}",
            f"- Stop reason: {response['stop_reason']}",
            f"- Input tokens: {response['input_tokens']}",
            f"- Output tokens including EOS: "
            f"{response['generated_tokens_including_eos']}",
            f"- Request wall time: {record['client_wall_s']:.3f} s",
            "",
        ]
    )


def _run_images(args: argparse.Namespace) -> None:
    assert args.images
    missing = [path for path in args.images if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing image(s): " + ", ".join(map(str, missing)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(args.http_workers, len(args.images))) as pool:
        futures = [
            pool.submit(_custom_image_job, index, path, args)
            for index, path in enumerate(args.images)
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            index, record = future.result()
            records[index] = record
            print(f"completed={completed}/{len(futures)}", flush=True)

    ordered = [records[index] for index in range(len(args.images))]
    image_output = args.output_dir / "images"
    image_output.mkdir(exist_ok=True)
    sections = ["# Crop OCR results", ""]
    for index, record in enumerate(ordered, start=1):
        source = Path(record["image_path"])
        saved = image_output / f"{index:03d}_{_safe_name(source)}{source.suffix.lower()}"
        shutil.copy2(source, saved)
        relative = os.path.relpath(saved, args.output_dir)
        markdown = _custom_markdown(record, relative)
        sections.extend([markdown, "---", ""])
        print(f"\n=== {source} ===\n{record['response']['text']}", flush=True)

    wall_s = time.perf_counter() - started
    summary = {
        "images": len(ordered),
        "crop_type": args.crop_type,
        "wall_s": wall_s,
        "images_per_s": len(ordered) / wall_s,
        "stop_reasons": dict(
            Counter(item["response"]["stop_reason"] for item in ordered)
        ),
    }
    _write_json(args.output_dir / "results.json", ordered)
    (args.output_dir / "results.md").write_text(
        "\n".join(sections), encoding="utf-8"
    )
    _write_json(args.output_dir / "summary.json", summary)
    print(
        "\n# Crop OCR summary\n\n"
        f"- Images: {summary['images']}\n"
        f"- Crop type: {summary['crop_type']}\n"
        f"- Wall time: {wall_s:.3f} s\n"
        f"- Results: `{args.output_dir / 'results.md'}`\n",
        flush=True,
    )


# =============================================================================
# INTERNAL OMNIDOCBENCH BENCHMARK AND TEDS EVALUATION
# This section implements the fixed validation procedure. Product users should
# not edit it. The pinned upstream evaluator supplies normalization and TEDS.
# =============================================================================

EXPECTED_JSON_SHA256 = (
    "a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496"
)
EXPECTED_IMAGES_SHA256 = (
    "58feeb96c60fcfab12ba4348c4e093ceaf1b707658dbfd0e08c24d7821d4c221"
)
EXPECTED_IMAGE_COUNT = 1651


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict]:
    print(
        "DATASET_CHECK_START checking OmniDocBench.json and all referenced "
        "images; inference starts after this check",
        flush=True,
    )
    dataset_bytes = args.dataset_json.read_bytes()
    pages = json.loads(dataset_bytes)
    names = [Path(page["page_info"]["image_path"]).name for page in pages]
    if len(names) != len(set(names)):
        raise ValueError("duplicate image basenames in OmniDocBench.json")

    aggregate = hashlib.sha256()
    total_bytes = 0
    for index, name in enumerate(sorted(names), start=1):
        path = args.images_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        total_bytes += size
        entry = {"path": name, "bytes": size, "sha256": _sha256_file(path)}
        aggregate.update(
            (json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
        if index % 100 == 0 or index == len(names):
            print(f"fingerprinted_images={index}/{len(names)}", flush=True)

    json_hash = hashlib.sha256(dataset_bytes).hexdigest()
    images_hash = aggregate.hexdigest()
    manifest = {
        "dataset_json_sha256": json_hash,
        "referenced_image_count": len(names),
        "referenced_images_total_bytes": total_bytes,
        "referenced_images_aggregate_sha256": images_hash,
        "matches_repository_authority": (
            len(names) == EXPECTED_IMAGE_COUNT
            and json_hash == EXPECTED_JSON_SHA256
            and images_hash == EXPECTED_IMAGES_SHA256
        ),
    }
    _write_json(args.output_dir / "dataset_manifest.json", manifest)
    print(
        f"DATASET_CHECK_RESULT matches={manifest['matches_repository_authority']} "
        f"images={len(names)}/{EXPECTED_IMAGE_COUNT} json_sha256={json_hash} "
        f"images_sha256={images_hash}",
        flush=True,
    )
    if not manifest["matches_repository_authority"]:
        print(
            "\033[1;31mDATASET WARNING: inputs differ from the validated "
            "OmniDocBench v1.6 fingerprints. The run will continue.\033[0m",
            file=sys.stderr,
            flush=True,
        )
    return pages, manifest


def _bbox(poly: list[float], width: int, height: int, padding: int) -> tuple[int, ...]:
    xs, ys = poly[0::2], poly[1::2]
    return (
        max(0, math.floor(min(xs)) - padding),
        max(0, math.floor(min(ys)) - padding),
        min(width, math.ceil(max(xs)) + padding),
        min(height, math.ceil(max(ys)) + padding),
    )


def _table_jobs(pages: list[dict], args: argparse.Namespace) -> list[tuple]:
    selected = pages[args.offset :]
    if args.limit_pages is not None:
        selected = selected[: args.limit_pages]
    return [
        (page_index, page, annotation_index, annotation)
        for page_index, page in enumerate(selected, start=args.offset)
        for annotation_index, annotation in enumerate(page.get("layout_dets") or [])
        if not annotation.get("ignore")
        and annotation.get("category_type") == "table"
    ]


def _request_id(job: tuple) -> str:
    page_index, _page, annotation_index, annotation = job
    return (
        f"page_{page_index:06d}_table_"
        f"{annotation.get('anno_id', annotation_index)}"
    )


def _table_job(job: tuple, args: argparse.Namespace) -> dict[str, Any]:
    page_index, page, annotation_index, annotation = job
    page_name = Path(page["page_info"]["image_path"]).name
    with Image.open(args.images_dir / page_name) as opened:
        image = opened.convert("RGB")
    bbox = _bbox(annotation["poly"], image.width, image.height, args.crop_padding)
    crop = image.crop(bbox)
    encoded = io.BytesIO()
    crop.save(encoded, format="PNG", optimize=False)
    response = _request(
        args.api_url, _request_id(job), encoded.getvalue(), "table", args.timeout_s
    )
    return {
        "request_id": _request_id(job),
        "page_index": page_index,
        "page_name": page_name,
        "annotation_index": annotation_index,
        "bbox_xyxy": list(bbox),
        "crop_size": list(crop.size),
        "gt_html": annotation.get("html") or annotation.get("text") or "",
        "pred_html": response["text"],
        "stop_reason": response["stop_reason"],
        "input_tokens": response["input_tokens"],
        "projected_image_tokens": response["projected_image_tokens"],
        "output_tokens": response["generated_tokens_including_eos"],
        "worker_wall_s": response["worker_wall_s"],
        "http_wall_s": response["http_wall_s"],
        "device_stage_s": response.get("device_stage_s", {}),
        "vision": response.get("vision", {}),
        "text_prefill": response.get("text_prefill", {}),
    }


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {
        record["request_id"]: record
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for record in [json.loads(line)]
    }


def _stop_process(process: multiprocessing.Process) -> None:
    process.join(timeout=0.2)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)


def _teds_worker(sender: Any, pred: str, gt: str) -> None:
    try:
        from src.metrics.table_metric import TEDS

        sender.send(
            (
                "ok",
                TEDS(structure_only=False).evaluate(pred, gt),
                TEDS(structure_only=True).evaluate(pred, gt),
            )
        )
    except BaseException:
        sender.send(("error", traceback.format_exc(), 0.0))
    finally:
        sender.close()


def _teds_scores(samples: list[dict], args: argparse.Namespace, metric: Any) -> list[dict]:
    """Score each non-exact table in a killable process with a hard timeout."""
    from tqdm import tqdm

    def packed(index, score, structure, status="ok", error=None):
        return {"index": index, "score": score, "structure": structure,
                "status": status, "error": error}

    pending, results = collections.deque(), []
    for index, sample in enumerate(samples):
        if sample["pred"] and sample["pred"] == sample["gt"]:
            results.append(packed(index, 1.0, 1.0))
        else:
            pending.append((index, sample))
    exact = len(results)
    print(
        f"[process-TEDS] workers={args.teds_workers} samples={len(samples)} "
        f"exact={exact} timeout={args.teds_timeout_s}s",
        flush=True,
    )

    context = multiprocessing.get_context("fork")
    active: dict[Any, tuple[multiprocessing.Process, float, tuple]] = {}
    progress = tqdm(total=len(samples), initial=exact, ascii=True, ncols=140,
                    desc="TEDS (process-isolated)")
    try:
        while pending or active:
            while pending and len(active) < args.teds_workers:
                task = pending.popleft()
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(
                    target=_teds_worker,
                    args=(sender, task[1]["pred"], task[1]["gt"]),
                    name=f"omnidoc-teds-{task[0]}",
                )
                process.start()
                sender.close()
                active[receiver] = (process, time.monotonic(), task)

            for receiver in wait_for_process(list(active), timeout=0.05):
                process, _, (index, sample) = active.pop(receiver)
                try:
                    status, score, structure = receiver.recv()
                finally:
                    receiver.close()
                    _stop_process(process)
                if status != "ok":
                    error = str(score)
                    print(
                        f"TEDS error for {sample['page_name']}; score set to 0",
                        flush=True,
                    )
                    score = structure = 0.0
                else:
                    error = None
                results.append(packed(index, score, structure, status, error))
                progress.update(1)

            now = time.monotonic()
            for receiver, (process, started, task) in list(active.items()):
                index, sample = task
                if not process.is_alive():
                    # The child can send its result and exit between the
                    # connection wait above and this liveness check. Let the
                    # next loop consume that already-buffered result.
                    if receiver.poll():
                        continue
                    receiver.close()
                    active.pop(receiver)
                    raise RuntimeError(
                        f"TEDS worker died for {sample['page_name']} "
                        f"with exit code {process.exitcode}"
                    )
                if now - started <= args.teds_timeout_s:
                    continue
                receiver.close()
                _stop_process(process)
                active.pop(receiver)
                reason = f"timeout:{args.teds_timeout_s}"
                metric._log_teds_timeout(
                    sample, sample["gt"], sample["pred"],
                    args.teds_timeout_s, reason,
                )
                results.append(packed(index, 0.0, 0.0, "timeout", reason))
                progress.update(1)
    finally:
        progress.close()
        for receiver, (process, _, _) in active.items():
            receiver.close()
            _stop_process(process)

    results.sort(key=lambda item: item["index"])
    assert len(results) == len(samples)
    return results


def _score(records: list[dict], args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(args.evaluator_root.resolve()))
    import src.metrics.cal_metric as metric
    from src.core.preprocess import normalized_table

    def document(value: str) -> str:
        value = normalized_table(value, "html")
        return value if "<html" in value.lower() else f"<html><body>{value}</body></html>"

    samples = [
        {
            "page_name": record["page_name"],
            "img_id": record["page_name"],
            "gt_idx": record["annotation_index"],
            "pred_idx": record["annotation_index"],
            "gt": document(record["gt_html"]),
            "pred": document(record["pred_html"]),
        }
        for record in records
    ]
    raw_scores = _teds_scores(samples, args, metric)
    per_table, pages, page_structures = [], collections.defaultdict(list), collections.defaultdict(list)
    for record, score in zip(records, raw_scores):
        item = {
            "request_id": record["request_id"],
            "page_name": record["page_name"],
            "TEDS": float(score["score"]),
            "TEDS_structure_only": float(score["structure"]),
            "error": score["error"],
        }
        per_table.append(item)
        pages[item["page_name"]].append(item["TEDS"])
        page_structures[item["page_name"]].append(item["TEDS_structure_only"])

    page_scores = {name: sum(values) / len(values) for name, values in pages.items()}
    structure_scores = {
        name: sum(values) / len(values) for name, values in page_structures.items()
    }
    result = {
        "table_count": len(per_table),
        "table_page_count": len(page_scores),
        "sample_TEDS": sum(item["TEDS"] for item in per_table) / len(per_table),
        "sample_TEDS_structure_only": sum(
            item["TEDS_structure_only"] for item in per_table
        ) / len(per_table),
        "page_TEDS": sum(page_scores.values()) / len(page_scores),
        "page_TEDS_structure_only": sum(structure_scores.values()) / len(structure_scores),
        "teds_timeout_s": args.teds_timeout_s,
        "teds_timeout_count": sum(item["status"] == "timeout" for item in raw_scores),
        "teds_error_count": sum(item["status"] == "error" for item in raw_scores),
        "per_page_TEDS": page_scores,
        "per_table": per_table,
    }
    _write_json(args.output_dir / "scores.json", result)
    return result


def _distribution(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    percentile = lambda q: values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]
    return {
        "mean": sum(values) / len(values), "p50": percentile(0.5),
        "p95": percentile(0.95), "max": values[-1],
    }


def _summary(
    records: list[dict],
    generation_wall_s: float,
    manifest: dict,
    scores: dict,
    service: dict | None,
) -> dict:
    http = [float(record["http_wall_s"]) for record in records]
    worker = [float(record["worker_wall_s"]) for record in records]
    overhead = [max(0.0, left - right) for left, right in zip(http, worker)]
    stage_names = sorted({name for record in records for name in record["device_stage_s"]})
    return {
        "tables": len(records),
        "unique_requests": len({record["request_id"] for record in records}),
        "generation_wall_s": generation_wall_s,
        "tables_per_s": len(records) / generation_wall_s,
        "http_latency_s": _distribution(http),
        "worker_latency_s": _distribution(worker),
        "http_wrapper_overhead_s": {"sum": sum(overhead), **_distribution(overhead)},
        "input_tokens": sum(record["input_tokens"] for record in records),
        "output_tokens_including_eos": sum(record["output_tokens"] for record in records),
        "stop_reasons": dict(Counter(record["stop_reason"] for record in records)),
        "device_stage_s": {
            name: sum(record["device_stage_s"].get(name, 0.0) for record in records)
            for name in stage_names
        },
        "physical_vision_tokens": sum(
            record["vision"].get("physical_vision_tokens", 0) for record in records
        ),
        "real_vision_tokens": sum(
            record["vision"].get("real_vision_tokens", 0) for record in records
        ),
        "physical_text_tokens": sum(
            record["text_prefill"].get("physical_text_tokens", 0) for record in records
        ),
        "real_text_tokens": sum(
            record["text_prefill"].get("real_text_tokens", 0) for record in records
        ),
        "service_scheduler": service,
        "metrics": scores,
        "dataset_fingerprint": manifest,
    }


def _print_score(scores: dict, timeout_s: float) -> None:
    print(
        "\n# OmniDocBench table OCR score\n\n"
        f"- Tables: {scores['table_count']}\n"
        f"- Table pages: {scores['table_page_count']}\n"
        f"- Page-TEDS: {scores['page_TEDS']:.6f}\n"
        f"- Sample TEDS: {scores['sample_TEDS']:.6f}\n"
        f"- Page structure-only TEDS: {scores['page_TEDS_structure_only']:.6f}\n"
        f"- TEDS timeouts: {scores['teds_timeout_count']}\n"
        f"- TEDS errors: {scores['teds_error_count']}\n",
        flush=True,
    )
    if scores["teds_timeout_count"]:
        recommended = max(120.0, timeout_s * 2)
        print(
            "\033[1;31mTEDS WARNING: timeouts were scored as zero. "
            f"Rerun --score-only with --teds-timeout-s {recommended:g}. "
            "Do not rerun OCR.\033[0m",
            file=sys.stderr,
            flush=True,
        )


def _run_omnidocbench(args: argparse.Namespace) -> None:
    output = args.output_dir / "tables.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.score_only:
        records = list(_read_jsonl(output).values())
        if not records:
            raise FileNotFoundError(f"No saved predictions in {output}")
        scores = _score(records, args)
        _print_score(scores, args.teds_timeout_s)
        return

    pages, manifest = _check_dataset(args)
    if args.fingerprint_only:
        return
    jobs = _table_jobs(pages, args)
    completed = _read_jsonl(output) if args.resume else {}
    pending = [job for job in jobs if _request_id(job) not in completed]
    done = len(jobs) - len(pending)
    started = time.perf_counter()
    with output.open("a" if args.resume else "w", encoding="utf-8") as target:
        with ThreadPoolExecutor(max_workers=args.http_workers) as pool:
            futures = [pool.submit(_table_job, job, args) for job in pending]
            print(f"submitted={len(futures)} http_workers={args.http_workers}", flush=True)
            for future in as_completed(futures):
                record = future.result()
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                target.flush()
                done += 1
                elapsed = time.perf_counter() - started
                print(
                    f"completed={done}/{len(jobs)} page={record['page_index']} "
                    f"elapsed_s={elapsed:.1f} tables_per_s={done / elapsed:.3f}",
                    flush=True,
                )

    generation_wall_s = time.perf_counter() - started
    service = _drain(args.api_url, args.timeout_s) if args.drain_server else None
    records = list(_read_jsonl(output).values())
    scores = _score(records, args)
    summary = _summary(records, generation_wall_s, manifest, scores, service)
    _write_json(args.output_dir / "run_summary.json", summary)
    _print_score(scores, args.teds_timeout_s)
    markdown = (
        "# OmniDocBench table OCR summary\n\n"
        f"- Tables: {len(records)}\n"
        f"- Generation wall time: {generation_wall_s:.3f} s\n"
        f"- Throughput: {summary['tables_per_s']:.3f} tables/s\n"
        f"- Page-TEDS: {scores['page_TEDS']:.6f}\n"
        f"- Sample TEDS: {scores['sample_TEDS']:.6f}\n"
        f"- Page structure-only TEDS: {scores['page_TEDS_structure_only']:.6f}\n"
        f"- TEDS timeouts: {scores['teds_timeout_count']}\n"
        f"- TEDS errors: {scores['teds_error_count']}\n"
        f"- Stop reasons: {summary['stop_reasons']}\n"
    )
    (args.output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(f"\n{markdown}", end="", flush=True)


def main() -> None:
    args = parse_args()
    if args.http_workers < 1 or args.teds_workers < 1:
        raise ValueError("worker counts must be positive")
    if (args.score_only or args.fingerprint_only) and not args.omnidocbench:
        raise ValueError("--score-only and --fingerprint-only require --omnidocbench")
    if args.score_only and args.fingerprint_only:
        raise ValueError("--score-only and --fingerprint-only are mutually exclusive")
    if args.drain_server is None:
        args.drain_server = args.omnidocbench
    if args.omnidocbench:
        _run_omnidocbench(args)
    else:
        _run_images(args)


if __name__ == "__main__":
    main()
