#!/usr/bin/env python3
"""Run and officially score OmniDocBench through the full-page OCR API.

Product users need only the CLI in ``parse_args``. The script sends independent
page requests, saves Markdown predictions, runs the committed robust evaluator,
calculates page-weighted CDM, and prints the official three-part Overall score.

This file contains its HTTP client, robust evaluator runner, and CDM runner. It
does not import or execute another file from this repository. Evaluation imports
come only from the public OmniDocBench checkout selected by ``--evaluator-root``.
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


SCRIPT_PATH = Path(__file__).resolve()

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
# EMBEDDED STANDALONE RUNTIMES
# These sections are invoked only through the hidden dispatcher at EOF.
# They use Python's standard library and the pinned public OmniDocBench checkout.
# =============================================================================

# ---- Concurrent page HTTP client --------------------------------------------


import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def _page_client_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8766/v1/pages")
    parser.add_argument(
        "--dataset-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--http-workers", type=int, default=64)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _post(url: str, image_path: Path, timeout_s: float) -> dict[str, Any]:
    request_id = image_path.name
    query = urllib.parse.urlencode(
        {"request_id": request_id, "filename": image_path.name}
    )
    request = urllib.request.Request(
        f"{url}?{query}",
        data=image_path.read_bytes(),
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read())


def _page_client_main() -> None:
    args = _page_client_parse_args()
    annotations = json.loads(args.dataset_json.expanduser().resolve().read_text())
    subset = annotations[args.offset : args.offset + args.limit]
    if len(subset) != args.limit:
        raise ValueError(f"requested {args.limit} pages, got {len(subset)}")
    images_dir = args.images_dir.expanduser().resolve()
    paths = [
        images_dir / Path(item["page_info"]["image_path"]).name
        for item in subset
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} images: {missing[:5]}")
    output_dir = args.output_dir.expanduser().resolve()
    predictions_dir = output_dir / "predictions"
    responses_dir = output_dir / "responses"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    completed = 0
    response_s: list[float] = []
    with ThreadPoolExecutor(max_workers=args.http_workers) as executor:
        futures = {
            executor.submit(_post, args.api_url, path, args.timeout_s): path
            for path in paths
        }
        for future in as_completed(futures):
            path = futures[future]
            payload = future.result()
            (predictions_dir / f"{path.stem}.md").write_text(
                payload["markdown"],
                encoding="utf-8",
            )
            (responses_dir / f"{path.stem}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            completed += 1
            response_s.append(float(payload["http_wall_s"]))
            elapsed = time.perf_counter() - started
            print(
                f"PAGE_API_PROGRESS completed={completed}/{len(paths)} "
                f"elapsed_s={elapsed:.3f} pages_per_s={completed / elapsed:.3f}",
                flush=True,
            )
    wall_s = time.perf_counter() - started
    summary: dict[str, Any] = {
        "offset": args.offset,
        "count": len(paths),
        "wall_s": wall_s,
        "pages_per_s": len(paths) / wall_s,
        "mean_response_s": sum(response_s) / len(response_s),
        "max_response_s": max(response_s),
        "predictions_dir": str(predictions_dir),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("PAGE_API_SUMMARY " + json.dumps(summary, separators=(",", ":")), flush=True)


# ---- Process-isolated OmniDocBench evaluator --------------------------------


import argparse
import collections
import faulthandler
import json
import multiprocessing
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DATASET: Any = None
_PRIMARY_LATEX_TIMEOUT_SEC = "0"
_FALLBACK_LATEX_TIMEOUT_SEC = "30"
_PAGE_DEBUG_STACK_INTERVAL_SEC = 0.0
_PAGE_DEBUG_IMAGE_NAME = ""


_LINE_BOUNDED_MARKDOWN_TABLE_PATTERN = re.compile(
    r"\|[^\r\n]*\|[^\S\r\n]*(?:\r?\n|$)"
)


@dataclass
class _PageTask:
    index: int
    sample: dict[str, Any]
    mode: str


@dataclass
class _ActivePage:
    task: _PageTask
    process: multiprocessing.Process
    receiver: Any
    started_at: float


@dataclass
class _TedsTask:
    index: int
    sample: dict[str, Any]
    pred: str
    gt: str


@dataclass
class _ActiveTeds:
    task: _TedsTask
    process: multiprocessing.Process
    receiver: Any
    started_at: float


def _image_name(sample: dict[str, Any]) -> str:
    return os.path.basename(sample.get("page_info", {}).get("image_path", ""))


def _install_line_bounded_markdown_table_pattern() -> None:
    """Prevent catastrophic cross-line backtracking in OmniDocBench parsing.

    Upstream uses ``r'\|\s*.*?\s*\|\n'`` with ``re.DOTALL`` merely to decide
    whether at least two Markdown-table rows exist.  After a large HTML table
    is space-masked, every stray pipe can make that expression rescan the
    entire masked region.  This equivalent row detector never crosses a line.
    """
    import src.core.preprocess.extract as extract_module

    extract_module.md_table_reg = _LINE_BOUNDED_MARKDOWN_TABLE_PATTERN


def _page_debug_enabled(img_name: str) -> bool:
    return bool(
        _PAGE_DEBUG_STACK_INTERVAL_SEC > 0
        and (
            not _PAGE_DEBUG_IMAGE_NAME
            or img_name == _PAGE_DEBUG_IMAGE_NAME
        )
    )


def _item_text(item: dict[str, Any]) -> str:
    for key in ("content", "text", "latex", "html"):
        value = item.get(key)
        if value is not None:
            return str(value)
    return ""


def _item_summary(items) -> dict[str, Any]:
    resolved = list(items or [])
    lengths = [len(_item_text(item)) for item in resolved]
    categories = collections.Counter(
        item.get("fine_category_type")
        or item.get("category_type")
        or ""
        for item in resolved
    )
    return {
        "items": len(resolved),
        "content_chars": sum(lengths),
        "empty_items": sum(length == 0 for length in lengths),
        "largest_item_chars": max(lengths, default=0),
        "categories": dict(sorted(categories.items())),
    }


def _install_matcher_debug_telemetry(img_name: str) -> None:
    if not _page_debug_enabled(img_name):
        return

    import src.core.matching.match as matching_module

    if getattr(matching_module, "_exp09_debug_telemetry_installed", False):
        return
    original_distance = matching_module.compute_edit_distance_matrix_new
    original_assignment = matching_module.linear_sum_assignment
    original_table_to_text = matching_module.table_to_text_lines

    def debug_distance(gt_lines, matched_lines):
        gt_chars = sum(len(line) for line in gt_lines)
        pred_chars = sum(len(line) for line in matched_lines)
        distance_calls = len(gt_lines) * len(matched_lines)
        character_product = gt_chars * pred_chars
        started = time.monotonic()
        print(
            "[page-debug-distance-begin] "
            f"image={img_name} gt_records={len(gt_lines)} "
            f"pred_records={len(matched_lines)} gt_chars={gt_chars} "
            f"pred_chars={pred_chars} distance_calls={distance_calls} "
            f"character_product={character_product}",
            flush=True,
        )
        try:
            return original_distance(gt_lines, matched_lines)
        finally:
            print(
                "[page-debug-distance-end] "
                f"image={img_name} elapsed_s={time.monotonic() - started:.6f}",
                flush=True,
            )

    def debug_assignment(cost_matrix):
        started = time.monotonic()
        print(
            "[page-debug-assignment-begin] "
            f"image={img_name} shape={tuple(cost_matrix.shape)}",
            flush=True,
        )
        try:
            return original_assignment(cost_matrix)
        finally:
            print(
                "[page-debug-assignment-end] "
                f"image={img_name} elapsed_s={time.monotonic() - started:.6f}",
                flush=True,
            )

    def debug_table_to_text(content):
        raw = str(content or "")
        started = time.monotonic()
        print(
            "[page-debug-table-to-text-begin] "
            f"image={img_name} input_chars={len(raw)} "
            f"table_tags={raw.lower().count('<table')} "
            f"tr_tags={raw.lower().count('<tr')} "
            f"td_tags={raw.lower().count('<td')}",
            flush=True,
        )
        try:
            lines = original_table_to_text(content)
            return lines
        finally:
            if "lines" in locals():
                line_chars = sum(len(str(line or "")) for line in lines)
                empty_lines = sum(not str(line or "") for line in lines)
                line_count = len(lines)
            else:
                line_chars = -1
                empty_lines = -1
                line_count = -1
            print(
                "[page-debug-table-to-text-end] "
                f"image={img_name} elapsed_s={time.monotonic() - started:.6f} "
                f"lines={line_count} line_chars={line_chars} "
                f"empty_lines={empty_lines}",
                flush=True,
            )

    matching_module.compute_edit_distance_matrix_new = debug_distance
    matching_module.linear_sum_assignment = debug_assignment
    matching_module.table_to_text_lines = debug_table_to_text
    matching_module._exp09_debug_telemetry_installed = True


def _install_page_parser_debug_telemetry(img_name: str) -> None:
    if not _page_debug_enabled(img_name):
        return

    import src.dataset.end2end_dataset as dataset_module

    if getattr(dataset_module, "_exp09_parser_debug_installed", False):
        return
    original_md_tex_filter = dataset_module.md_tex_filter
    original_simple_match = dataset_module.match_gt2pred_simple

    def debug_md_tex_filter(pred_content):
        raw = str(pred_content or "")
        started = time.monotonic()
        print(
            "[page-debug-md-filter-begin] "
            f"image={img_name} input_chars={len(raw)} "
            f"input_bytes={len(raw.encode('utf-8'))} "
            f"table_tags={raw.lower().count('<table')} "
            f"tr_tags={raw.lower().count('<tr')} "
            f"td_tags={raw.lower().count('<td')}",
            flush=True,
        )
        try:
            parsed = original_md_tex_filter(pred_content)
            return parsed
        finally:
            if "parsed" in locals():
                summary = {
                    category: _item_summary(items)
                    for category, items in parsed.items()
                }
            else:
                summary = {"error_before_result": True}
            print(
                "[page-debug-md-filter-end] "
                f"image={img_name} elapsed_s={time.monotonic() - started:.6f} "
                f"summary={json.dumps(summary, ensure_ascii=False, separators=(',', ':'))}",
                flush=True,
            )

    def debug_simple_match(gt_items, pred_items, line_type, match_img_name):
        started = time.monotonic()
        print(
            "[page-debug-simple-match-begin] "
            f"image={match_img_name} line_type={line_type} "
            f"gt={json.dumps(_item_summary(gt_items), ensure_ascii=False, separators=(',', ':'))} "
            f"pred={json.dumps(_item_summary(pred_items), ensure_ascii=False, separators=(',', ':'))}",
            flush=True,
        )
        try:
            return original_simple_match(
                gt_items,
                pred_items,
                line_type,
                match_img_name,
            )
        finally:
            print(
                "[page-debug-simple-match-end] "
                f"image={match_img_name} line_type={line_type} "
                f"elapsed_s={time.monotonic() - started:.6f}",
                flush=True,
            )

    dataset_module.md_tex_filter = debug_md_tex_filter
    dataset_module.match_gt2pred_simple = debug_simple_match
    dataset_module._exp09_parser_debug_installed = True


def _install_timeout_safe_matcher() -> None:
    import src.dataset.end2end_dataset as dataset_module
    from src.core.matching import match_gt2pred_timeout_safe

    def timeout_safe_quick_match(
        gt_items,
        pred_items,
        line_type,
        img_name,
        truncated_timeout_sec=None,
        fallback_short_line_max_chars=None,
        fallback_target_chunk_chars=None,
        fallback_max_chunk_chars=None,
        fallback_max_chunk_span=8,
        fallback_order_window=None,
        fallback_order_penalty=0.08,
        split_pred_formula=True,
    ):
        del line_type, truncated_timeout_sec, split_pred_formula
        _install_matcher_debug_telemetry(img_name)
        print(
            "[page-debug-fallback-input] "
            f"image={img_name} "
            f"gt={json.dumps(_item_summary(gt_items), ensure_ascii=False, separators=(',', ':'))} "
            f"pred={json.dumps(_item_summary(pred_items), ensure_ascii=False, separators=(',', ':'))}",
            flush=True,
        )
        return match_gt2pred_timeout_safe(
            gt_items,
            pred_items,
            img_name,
            short_line_max_chars=fallback_short_line_max_chars,
            target_chunk_chars=fallback_target_chunk_chars,
            max_chunk_chars=fallback_max_chunk_chars,
            max_chunk_span=fallback_max_chunk_span,
            order_window=fallback_order_window,
            order_penalty=fallback_order_penalty,
            fallback_reason="page_timeout",
        )

    dataset_module.match_gt2pred_quick = timeout_safe_quick_match


def _page_process_entry(sender, task: _PageTask, pred_folder: str) -> None:
    img_name = _image_name(task.sample)
    debug_stack = _page_debug_enabled(img_name)
    try:
        _install_line_bounded_markdown_table_pattern()
        if debug_stack:
            faulthandler.enable()
            faulthandler.dump_traceback_later(
                _PAGE_DEBUG_STACK_INTERVAL_SEC,
                repeat=True,
            )
            print(
                "[page-debug-stack-enabled] "
                f"image={_image_name(task.sample)} mode={task.mode} "
                f"interval_s={_PAGE_DEBUG_STACK_INTERVAL_SEC:g}",
                flush=True,
            )
        if task.mode == "fallback":
            os.environ["OMNIDOCBENCH_LATEX_TO_TEXT_TIMEOUT_SEC"] = (
                _FALLBACK_LATEX_TIMEOUT_SEC
            )
            _install_timeout_safe_matcher()
        else:
            os.environ["OMNIDOCBENCH_LATEX_TO_TEXT_TIMEOUT_SEC"] = (
                _PRIMARY_LATEX_TIMEOUT_SEC
            )
        if debug_stack:
            _install_matcher_debug_telemetry(img_name)
            _install_page_parser_debug_telemetry(img_name)
        page_result = _DATASET._match_single_page(
            task.index,
            task.sample,
            pred_folder,
        )
        sender.send(("ok", page_result))
    except BaseException:
        sender.send(("error", traceback.format_exc()))
    finally:
        if debug_stack:
            faulthandler.cancel_dump_traceback_later()
        sender.close()


def _stop_process(process: multiprocessing.Process) -> None:
    process.join(timeout=0.2)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)


def _start_page_process(
    ctx,
    task: _PageTask,
    pred_folder: str,
) -> _ActivePage:
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_page_process_entry,
        args=(sender, task, pred_folder),
        name=f"omnidoc-page-{task.index}-{task.mode}",
    )
    process.start()
    sender.close()
    return _ActivePage(
        task=task,
        process=process,
        receiver=receiver,
        started_at=time.monotonic(),
    )


def _collect_page_matches_process_isolated(self, gt_samples, pred_folder):
    global _DATASET
    _DATASET = self

    from tqdm import tqdm

    ctx = multiprocessing.get_context("fork")
    worker_count = max(1, int(self.match_workers))
    primary_timeout = float(os.environ["OMNIDOCBENCH_PAGE_PROCESS_TIMEOUT_SEC"])
    fallback_timeout = float(
        os.environ["OMNIDOCBENCH_FALLBACK_PROCESS_TIMEOUT_SEC"]
    )
    pending = collections.deque(
        _PageTask(index=index, sample=sample, mode="primary")
        for index, sample in enumerate(gt_samples)
    )
    retries: collections.deque[_PageTask] = collections.deque()
    active: list[_ActivePage] = []
    results: list[dict[str, Any]] = []
    completed = 0

    print(
        "[process-match] "
        f"workers={worker_count} pages={len(gt_samples)} "
        f"primary_timeout={primary_timeout:g}s "
        f"fallback_timeout={fallback_timeout:g}s",
        flush=True,
    )
    progress = tqdm(
        total=len(gt_samples),
        ascii=True,
        ncols=140,
        desc="Matching pages (process-isolated)",
    )
    try:
        while pending or retries or active:
            while len(active) < worker_count and (retries or pending):
                task = retries.popleft() if retries else pending.popleft()
                active.append(_start_page_process(ctx, task, pred_folder))

            made_progress = False
            now = time.monotonic()
            for running in list(active):
                task = running.task
                timeout_sec = (
                    fallback_timeout if task.mode == "fallback" else primary_timeout
                )
                if running.receiver.poll():
                    made_progress = True
                    status, payload = running.receiver.recv()
                    running.receiver.close()
                    _stop_process(running.process)
                    active.remove(running)
                    if status != "ok":
                        raise RuntimeError(
                            f"page matcher failed for {_image_name(task.sample)} "
                            f"mode={task.mode}:\n{payload}"
                        )
                    results.append(payload)
                    completed += 1
                    progress.update(1)
                    continue

                if not running.process.is_alive():
                    made_progress = True
                    exit_code = running.process.exitcode
                    running.receiver.close()
                    active.remove(running)
                    raise RuntimeError(
                        f"page matcher exited without a result for "
                        f"{_image_name(task.sample)} mode={task.mode} "
                        f"exit_code={exit_code}"
                    )

                elapsed = now - running.started_at
                if elapsed <= timeout_sec:
                    continue

                made_progress = True
                running.receiver.close()
                _stop_process(running.process)
                active.remove(running)
                img_name = _image_name(task.sample)
                if task.mode == "primary":
                    print(
                        f"[process-match-timeout] {img_name}: "
                        f"primary exceeded {timeout_sec:g}s; retrying bounded fallback",
                        flush=True,
                    )
                    self._record_match_fallback(img_name, "page_timeout")
                    retries.append(
                        _PageTask(
                            index=task.index,
                            sample=task.sample,
                            mode="fallback",
                        )
                    )
                else:
                    raise TimeoutError(
                        f"bounded fallback exceeded {timeout_sec:g}s for {img_name}"
                    )

            if not made_progress:
                time.sleep(0.01)
    finally:
        progress.close()
        for running in active:
            running.receiver.close()
            _stop_process(running.process)

    assert completed == len(gt_samples), (completed, len(gt_samples))
    results.sort(key=lambda item: item["index"])
    return results


def _teds_process_entry(sender, pred: str, gt: str) -> None:
    try:
        from src.metrics.table_metric import TEDS

        score = TEDS(structure_only=False).evaluate(pred, gt)
        structure_score = TEDS(structure_only=True).evaluate(pred, gt)
        sender.send(("ok", (score, structure_score)))
    except BaseException:
        sender.send(("metric_error", traceback.format_exc()))
    finally:
        sender.close()


def _start_teds_process(ctx, task: _TedsTask) -> _ActiveTeds:
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_teds_process_entry,
        args=(sender, task.pred, task.gt),
        name=f"omnidoc-teds-{task.index}",
    )
    try:
        process.start()
    except BaseException as exc:
        sender.close()
        receiver.close()
        raise RuntimeError(
            "failed to start TEDS worker for "
            f"{task.sample.get('img_id')} gt_idx={task.sample.get('gt_idx')} "
            f"pred_idx={task.sample.get('pred_idx')}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    sender.close()
    return _ActiveTeds(
        task=task,
        process=process,
        receiver=receiver,
        started_at=time.monotonic(),
    )


def _collect_teds_process_isolated(
    samples: list[dict[str, Any]],
    worker_count: int,
    timeout_sec: float | None,
    metric_module: Any,
) -> list[dict[str, Any]]:
    from tqdm import tqdm

    ctx = multiprocessing.get_context("fork")
    pending: collections.deque[_TedsTask] = collections.deque()
    results: list[dict[str, Any]] = []
    exact_count = 0
    for index, sample in enumerate(samples):
        gt = sample["norm_gt"] if sample.get("norm_gt") else sample["gt"]
        pred = sample["norm_pred"] if sample.get("norm_pred") else sample["pred"]
        if pred and gt and pred == gt:
            exact_count += 1
            results.append(
                {
                    "original_index": index,
                    "score": 1.0,
                    "score_structure_only": 1.0,
                    "status": "ok",
                    "case_record": None,
                }
            )
        else:
            pending.append(
                _TedsTask(
                    index=index,
                    sample=sample,
                    pred=pred,
                    gt=gt,
                )
            )

    print(
        "[process-TEDS] "
        f"workers={worker_count} samples={len(samples)} exact={exact_count} "
        f"timeout={timeout_sec if timeout_sec is not None else 'none'}s",
        flush=True,
    )
    active: list[_ActiveTeds] = []
    progress = tqdm(
        total=len(samples),
        initial=exact_count,
        ascii=True,
        ncols=140,
        desc="TEDS (process-isolated)",
    )
    try:
        while pending or active:
            while len(active) < worker_count and pending:
                active.append(_start_teds_process(ctx, pending.popleft()))

            made_progress = False
            now = time.monotonic()
            for running in list(active):
                task = running.task
                if running.receiver.poll():
                    made_progress = True
                    status, payload = running.receiver.recv()
                    running.receiver.close()
                    _stop_process(running.process)
                    active.remove(running)
                    if status == "ok":
                        score, structure_score = payload
                        results.append(
                            {
                                "original_index": task.index,
                                "score": score,
                                "score_structure_only": structure_score,
                                "status": "ok",
                                "case_record": None,
                            }
                        )
                    else:
                        reason = str(payload)
                        print(
                            "TEDS score error for table "
                            f"{task.sample.get('gt_idx')} in "
                            f"{task.sample.get('img_id')}: {reason}. "
                            "The score is set to 0.",
                            flush=True,
                        )
                        results.append(
                            {
                                "original_index": task.index,
                                "score": 0.0,
                                "score_structure_only": 0.0,
                                "status": "error",
                                "case_record": metric_module._build_case_record(
                                    task.sample,
                                    reason=reason,
                                ),
                            }
                        )
                    progress.update(1)
                    continue

                if not running.process.is_alive():
                    made_progress = True
                    exit_code = running.process.exitcode
                    running.receiver.close()
                    active.remove(running)
                    raise RuntimeError(
                        "TEDS worker exited without a result for "
                        f"{task.sample.get('img_id')} "
                        f"gt_idx={task.sample.get('gt_idx')} "
                        f"pred_idx={task.sample.get('pred_idx')} "
                        f"exit_code={exit_code}"
                    )

                if timeout_sec is None:
                    continue
                elapsed = now - running.started_at
                if elapsed <= timeout_sec:
                    continue

                made_progress = True
                running.receiver.close()
                _stop_process(running.process)
                active.remove(running)
                reason = f"timeout:{timeout_sec}"
                metric_module._log_teds_timeout(
                    task.sample,
                    task.gt,
                    task.pred,
                    timeout_sec,
                    reason,
                )
                results.append(
                    {
                        "original_index": task.index,
                        "score": 0.0,
                        "score_structure_only": 0.0,
                        "status": "timeout",
                        "case_record": metric_module._build_case_record(
                            task.sample,
                            reason=reason,
                            timeout_sec=timeout_sec,
                            gt_length=len(task.gt),
                            pred_length=len(task.pred),
                        ),
                    }
                )
                progress.update(1)

            if not made_progress:
                time.sleep(0.01)
    finally:
        progress.close()
        for running in active:
            running.receiver.close()
            _stop_process(running.process)

    assert len(results) == len(samples), (len(results), len(samples))
    results.sort(key=lambda item: item["original_index"])
    return results


def _install_process_isolated_teds() -> None:
    import src.metrics.cal_metric as metric_module

    def evaluate(self, group_info=[], save_name="default"):
        samples = metric_module._as_sample_list(self.samples)
        results = _collect_teds_process_isolated(
            samples,
            max(1, int(self.max_workers)),
            self.timeout_sec,
            metric_module,
        )

        group_scores: dict[str, list[float]] = collections.defaultdict(list)
        group_structure_scores: dict[str, list[float]] = collections.defaultdict(
            list
        )
        per_table_score = {}
        timeout_cases = []
        error_cases = []
        for result in results:
            if result.get("status") == "timeout" and result.get("case_record"):
                timeout_cases.append(result["case_record"])
            elif result.get("status") == "error" and result.get("case_record"):
                error_cases.append(result["case_record"])

            sample = samples[result["original_index"]]
            score = result["score"]
            structure_score = result["score_structure_only"]
            group_scores["all"].append(score)
            group_structure_scores["all"].append(structure_score)
            if not sample.get("metric"):
                sample["metric"] = {}
            sample["metric"]["TEDS"] = score
            sample["metric"]["TEDS_structure_only"] = structure_score
            per_table_score[sample["img_id"] + "_" + str(sample["gt_idx"])] = {
                "TEDS": score,
                "TEDS_structure_only": structure_score,
            }
            for group in group_info:
                if metric_module._sample_matches_group(sample, group):
                    group_scores[str(group)].append(score)
                    group_structure_scores[str(group)].append(structure_score)

        with open(
            f"./result/{save_name}_per_table_TEDS.json",
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(per_table_score, output_file, indent=4, ensure_ascii=False)

        aggregate = {}
        for group_name, scores in group_scores.items():
            aggregate[group_name] = metric_module._safe_average(scores)
            if aggregate[group_name] == "NaN":
                print(f"Warning: Empty matched samples for {group_name}.")

        structure_aggregate = {}
        for group_name, scores in group_structure_scores.items():
            structure_aggregate[group_name] = metric_module._safe_average(scores)
            if structure_aggregate[group_name] == "NaN":
                print(f"Warning: Empty matched samples for {group_name}.")

        self.debug_info = {
            "stage": "TEDS",
            "scheduler": "process_isolated_parent_timeout",
            "workers": self.max_workers,
            "timeout_sec": self.timeout_sec,
            "sample_count": len(samples),
            "timeout_case_count": len(timeout_cases),
            "timeout_cases": metric_module._sort_case_records(timeout_cases),
            "error_case_count": len(error_cases),
            "error_cases": metric_module._sort_case_records(error_cases),
        }
        return self.samples, {
            "TEDS": aggregate,
            "TEDS_structure_only": structure_aggregate,
        }

    metric_module.call_TEDS.evaluate = evaluate


def _run_teds_only(args: argparse.Namespace) -> None:
    import src.metrics.cal_metric as metric_module

    input_path = Path(args.teds_only_input).resolve()
    output_dir = Path(args.teds_only_output_dir).resolve()
    samples = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise TypeError(
            f"expected a JSON list of matched table samples, got {type(samples).__name__}"
        )
    if (
        args.teds_expected_samples is not None
        and len(samples) != args.teds_expected_samples
    ):
        raise ValueError(
            f"expected {args.teds_expected_samples} table samples, got {len(samples)}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "result").mkdir()
    old_cwd = Path.cwd()
    started_at = time.monotonic()
    try:
        os.chdir(output_dir)
        runner = metric_module.call_TEDS(
            samples,
            {
                "teds_workers": args.teds_workers,
                "teds_timeout_sec": args.teds_timeout_sec,
            },
        )
        samples, aggregate = runner.evaluate([], "teds_recomputed")
    finally:
        os.chdir(old_cwd)

    per_page: dict[str, list[float]] = collections.defaultdict(list)
    per_page_structure: dict[str, list[float]] = collections.defaultdict(list)
    for sample in samples:
        img_id = sample["img_id"]
        if img_id.endswith((".jpg", ".png")):
            page_name = img_id
        else:
            page_name = "_".join(img_id.split("_")[:-1])
        per_page[page_name].append(sample["metric"]["TEDS"])
        per_page_structure[page_name].append(
            sample["metric"]["TEDS_structure_only"]
        )
    if (
        args.teds_expected_pages is not None
        and len(per_page) != args.teds_expected_pages
    ):
        raise ValueError(
            f"expected {args.teds_expected_pages} table pages, got {len(per_page)}"
        )
    per_page_scores = {
        page_name: sum(scores) / len(scores)
        for page_name, scores in sorted(per_page.items())
    }
    per_page_structure_scores = {
        page_name: sum(scores) / len(scores)
        for page_name, scores in sorted(per_page_structure.items())
    }
    page_average = (
        sum(per_page_scores.values()) / len(per_page_scores)
        if per_page_scores
        else "NaN"
    )
    page_structure_average = (
        sum(per_page_structure_scores.values()) / len(per_page_structure_scores)
        if per_page_structure_scores
        else "NaN"
    )
    summary = {
        "source_table_result": str(input_path),
        "elapsed_s": time.monotonic() - started_at,
        "sample_count": len(samples),
        "page_count": len(per_page_scores),
        "sample_aggregate": aggregate,
        "page_aggregate": {
            "TEDS": {"ALL": page_average},
            "TEDS_structure_only": {"ALL": page_structure_average},
        },
        "execution": runner.debug_info,
        "per_page_TEDS": per_page_scores,
        "per_page_TEDS_structure_only": per_page_structure_scores,
    }
    summary_path = output_dir / "teds_only_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[teds-only-summary] saved to {summary_path}", flush=True)


def _evaluator_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OmniDocBench evaluation with hard page isolation."
    )
    parser.add_argument("--config")
    parser.add_argument(
        "--evaluator-root",
        required=True,
    )
    parser.add_argument("--match-workers", type=int, default=24)
    parser.add_argument("--teds-workers", type=int, default=12)
    parser.add_argument("--page-timeout-sec", type=float, default=120.0)
    parser.add_argument("--fallback-timeout-sec", type=float, default=180.0)
    parser.add_argument("--fallback-latex-timeout-sec", type=float, default=30.0)
    parser.add_argument(
        "--page-debug-stack-interval-sec",
        type=float,
        default=0.0,
        help=(
            "Dump the selected page child process stack at this interval; "
            "zero disables diagnostic stack dumps."
        ),
    )
    parser.add_argument(
        "--page-debug-image-name",
        default="",
        help="Restrict page diagnostics to this image basename.",
    )
    parser.add_argument(
        "--teds-only-input",
        help="Recompute TEDS from an existing matched table-result JSON.",
    )
    parser.add_argument("--teds-only-output-dir")
    parser.add_argument("--teds-timeout-sec", type=float, default=120.0)
    parser.add_argument("--teds-expected-samples", type=int)
    parser.add_argument("--teds-expected-pages", type=int)
    return parser.parse_args()


def _evaluator_main() -> None:
    global _FALLBACK_LATEX_TIMEOUT_SEC
    global _PAGE_DEBUG_IMAGE_NAME
    global _PAGE_DEBUG_STACK_INTERVAL_SEC
    args = _evaluator_parse_args()
    if args.match_workers <= 0 or args.teds_workers <= 0:
        raise ValueError("worker counts must be positive")
    if args.page_timeout_sec <= 0 or args.fallback_timeout_sec <= 0:
        raise ValueError("page timeouts must be positive")
    if args.teds_timeout_sec <= 0:
        raise ValueError("TEDS timeout must be positive")
    if args.page_debug_stack_interval_sec < 0:
        raise ValueError("page debug stack interval must be non-negative")
    if bool(args.teds_only_input) != bool(args.teds_only_output_dir):
        raise ValueError(
            "--teds-only-input and --teds-only-output-dir must be used together"
        )
    if not args.teds_only_input and not args.config:
        raise ValueError("--config is required for full evaluation")

    evaluator_root = Path(args.evaluator_root).resolve()
    if not (evaluator_root / "pdf_validation.py").is_file():
        raise FileNotFoundError(f"invalid evaluator root: {evaluator_root}")
    sys.path.insert(0, str(evaluator_root))

    _install_process_isolated_teds()
    if args.teds_only_input:
        _run_teds_only(args)
        return

    os.environ["OMNIDOCBENCH_LATEX_TO_TEXT_TIMEOUT_SEC"] = "0"
    os.environ["OMNIDOCBENCH_PAGE_PROCESS_TIMEOUT_SEC"] = str(
        args.page_timeout_sec
    )
    os.environ["OMNIDOCBENCH_FALLBACK_PROCESS_TIMEOUT_SEC"] = str(
        args.fallback_timeout_sec
    )
    _FALLBACK_LATEX_TIMEOUT_SEC = str(args.fallback_latex_timeout_sec)
    _PAGE_DEBUG_STACK_INTERVAL_SEC = float(
        args.page_debug_stack_interval_sec
    )
    _PAGE_DEBUG_IMAGE_NAME = os.path.basename(args.page_debug_image_name)

    from src.core.pipeline import load_config, run_config
    from src.dataset.end2end_dataset import End2EndDataset

    config = load_config(args.config)
    for task_cfg in config.values():
        if not task_cfg:
            continue
        task_cfg["dataset"]["match_workers"] = args.match_workers
        table_cfg = task_cfg.get("metrics", {}).get("table")
        if table_cfg is not None:
            table_cfg["teds_workers"] = args.teds_workers

    End2EndDataset._collect_page_matches = (
        _collect_page_matches_process_isolated
    )
    run_config(config)


# ---- Direct CDM runner -------------------------------------------------------


import argparse
import json
import os
import sys
import time
from pathlib import Path


def _cdm_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--evaluator-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--save-name", default="predictions_quick_match_cdm")
    parser.add_argument("--save-vis", action="store_true")
    return parser.parse_args()


def _cdm_main() -> None:
    args = _cdm_parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.sample_limit is not None and args.sample_limit <= 0:
        raise ValueError("--sample-limit must be positive")

    input_path = args.input.resolve()
    evaluator_root = args.evaluator_root.resolve()
    output_dir = args.output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not (evaluator_root / "pdf_validation.py").is_file():
        raise FileNotFoundError(f"invalid evaluator root: {evaluator_root}")

    samples = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise TypeError(f"expected a list in {input_path}")
    if args.sample_limit is not None:
        samples = samples[: args.sample_limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CDM_SAVE_VIS"] = "1" if args.save_vis else "0"
    os.chdir(output_dir)
    sys.path.insert(0, str(evaluator_root))

    from src.metrics.cal_metric import call_CDM

    print(
        f"[cdm-direct] samples={len(samples)} workers={args.workers} "
        f"save_vis={args.save_vis} output={output_dir}",
        flush=True,
    )
    started = time.monotonic()
    metric = call_CDM(samples, {"cdm_workers": args.workers})
    _evaluated_samples, scores = metric.evaluate(
        save_name=args.save_name,
        max_workers=args.workers,
    )
    wall_s = time.monotonic() - started

    result_root = output_dir / "result"
    summary = {
        "input": str(input_path),
        "evaluator_root": str(evaluator_root),
        "sample_count": len(samples),
        "workers": args.workers,
        "save_vis": args.save_vis,
        "wall_s": wall_s,
        "samples_per_s": len(samples) / wall_s if wall_s else None,
        "scores": scores,
        "debug": metric.debug_info,
        "per_sample_scores": str(
            result_root / f"{args.save_name}_per_sample_CDM.json"
        ),
        "evaluated_samples": str(result_root / f"{args.save_name}_result.json"),
    }
    summary_path = output_dir / "cdm_run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(f"[cdm-direct] summary={summary_path}", flush=True)


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
            + "; install public TeX Live (including CJK resources), "
            "ImageMagick 7 with PDF support, and Ghostscript, then make "
            "pdflatex, kpsewhich, magick, and gs available on PATH"
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
            str(SCRIPT_PATH), "--_internal-page-client",
            "--api-url", args.api_url,
            "--dataset-json", str(dataset_json),
            "--images-dir", str(images_dir),
            "--offset", str(args.offset),
            "--limit", str(len(selected)),
            "--http-workers", str(args.http_workers),
            "--timeout-s", str(args.request_timeout_s),
            "--output-dir", str(generation_dir),
        ]
        _run(
            generation_command,
            cwd=output_dir,
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
        str(SCRIPT_PATH), "--_internal-evaluator",
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
        str(SCRIPT_PATH), "--_internal-cdm",
        "--input", str(matched),
        "--output-dir", str(cdm_dir),
        "--evaluator-root", str(evaluator_root),
        "--workers", str(_cdm_workers(args.cdm_workers)),
        "--save-name", "predictions_quick_match_cdm",
    ]
    cdm_wall = _run(
        cdm_command,
        cwd=output_dir,
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


def _entrypoint() -> None:
    internal = sys.argv[1] if len(sys.argv) > 1 else None
    if internal == "--_internal-page-client":
        del sys.argv[1]
        _page_client_main()
    elif internal == "--_internal-evaluator":
        del sys.argv[1]
        _evaluator_main()
    elif internal == "--_internal-cdm":
        del sys.argv[1]
        _cdm_main()
    else:
        main()


if __name__ == "__main__":
    _entrypoint()
