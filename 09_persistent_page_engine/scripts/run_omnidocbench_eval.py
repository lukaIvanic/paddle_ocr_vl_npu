#!/usr/bin/env python3
"""Run OmniDocBench evaluation with process-isolated page matching.

The upstream evaluator parallelizes page matching with threads.  Its display-
formula matcher is Python-heavy, so those threads contend on the GIL; a page
that takes a few seconds alone can take more than a minute in a large run.
The evaluator also cannot reliably interrupt a page stuck inside native code.

This entrypoint preserves the upstream matcher and metrics, but executes each
page in its own forked process.  A primary page process uses direct LaTeX
conversion (avoiding one subprocess launch per formula).  If it exceeds the
hard page deadline, it is terminated and retried once with the evaluator's
bounded chunked-Hungarian fallback and guarded LaTeX conversion.
"""

from __future__ import annotations

import argparse
import collections
import multiprocessing
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DATASET: Any = None
_PRIMARY_LATEX_TIMEOUT_SEC = "0"
_FALLBACK_LATEX_TIMEOUT_SEC = "30"


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


def _image_name(sample: dict[str, Any]) -> str:
    return os.path.basename(sample.get("page_info", {}).get("image_path", ""))


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
    try:
        if task.mode == "fallback":
            os.environ["OMNIDOCBENCH_LATEX_TO_TEXT_TIMEOUT_SEC"] = (
                _FALLBACK_LATEX_TIMEOUT_SEC
            )
            _install_timeout_safe_matcher()
        else:
            os.environ["OMNIDOCBENCH_LATEX_TO_TEXT_TIMEOUT_SEC"] = (
                _PRIMARY_LATEX_TIMEOUT_SEC
            )
        page_result = _DATASET._match_single_page(
            task.index,
            task.sample,
            pred_folder,
        )
        sender.send(("ok", page_result))
    except BaseException:
        sender.send(("error", traceback.format_exc()))
    finally:
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


def _install_teds_exact_shortcut() -> None:
    import src.metrics.cal_metric as metric_module

    original = metric_module._evaluate_teds_pair_with_timeout

    def evaluate_teds_pair(pred, gt, timeout_sec):
        if pred and gt and pred == gt:
            return 1.0, 1.0, None
        return original(pred, gt, timeout_sec)

    metric_module._evaluate_teds_pair_with_timeout = evaluate_teds_pair


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OmniDocBench evaluation with hard page isolation."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--evaluator-root",
        default="/workspace/repos/OmniDocBench_eval",
    )
    parser.add_argument("--match-workers", type=int, default=24)
    parser.add_argument("--teds-workers", type=int, default=12)
    parser.add_argument("--page-timeout-sec", type=float, default=120.0)
    parser.add_argument("--fallback-timeout-sec", type=float, default=180.0)
    parser.add_argument("--fallback-latex-timeout-sec", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    global _FALLBACK_LATEX_TIMEOUT_SEC
    args = _parse_args()
    if args.match_workers <= 0 or args.teds_workers <= 0:
        raise ValueError("worker counts must be positive")
    if args.page_timeout_sec <= 0 or args.fallback_timeout_sec <= 0:
        raise ValueError("page timeouts must be positive")

    evaluator_root = Path(args.evaluator_root).resolve()
    if not (evaluator_root / "pdf_validation.py").is_file():
        raise FileNotFoundError(f"invalid evaluator root: {evaluator_root}")
    sys.path.insert(0, str(evaluator_root))

    os.environ["OMNIDOCBENCH_LATEX_TO_TEXT_TIMEOUT_SEC"] = "0"
    os.environ["OMNIDOCBENCH_PAGE_PROCESS_TIMEOUT_SEC"] = str(
        args.page_timeout_sec
    )
    os.environ["OMNIDOCBENCH_FALLBACK_PROCESS_TIMEOUT_SEC"] = str(
        args.fallback_timeout_sec
    )
    _FALLBACK_LATEX_TIMEOUT_SEC = str(args.fallback_latex_timeout_sec)

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
    _install_teds_exact_shortcut()
    run_config(config)


if __name__ == "__main__":
    main()
