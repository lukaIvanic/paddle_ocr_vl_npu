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

TEDS uses the same parent-owned process pattern.  This avoids the evaluator's
nested thread/process scheduler, gives every table a hard parent-side timeout,
and exposes process-start failures instead of masking them with an invalid
``join()``.  ``--teds-only-input`` can recover TEDS directly from a previously
saved matched-table result without repeating page matching.
"""

from __future__ import annotations

import argparse
import collections
import json
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OmniDocBench evaluation with hard page isolation."
    )
    parser.add_argument("--config")
    parser.add_argument(
        "--evaluator-root",
        default="/workspace/repos/OmniDocBench_eval",
    )
    parser.add_argument("--match-workers", type=int, default=24)
    parser.add_argument("--teds-workers", type=int, default=12)
    parser.add_argument("--page-timeout-sec", type=float, default=120.0)
    parser.add_argument("--fallback-timeout-sec", type=float, default=180.0)
    parser.add_argument("--fallback-latex-timeout-sec", type=float, default=30.0)
    parser.add_argument(
        "--teds-only-input",
        help="Recompute TEDS from an existing matched table-result JSON.",
    )
    parser.add_argument("--teds-only-output-dir")
    parser.add_argument("--teds-timeout-sec", type=float, default=120.0)
    parser.add_argument("--teds-expected-samples", type=int)
    parser.add_argument("--teds-expected-pages", type=int)
    return parser.parse_args()


def main() -> None:
    global _FALLBACK_LATEX_TIMEOUT_SEC
    args = _parse_args()
    if args.match_workers <= 0 or args.teds_workers <= 0:
        raise ValueError("worker counts must be positive")
    if args.page_timeout_sec <= 0 or args.fallback_timeout_sec <= 0:
        raise ValueError("page timeouts must be positive")
    if args.teds_timeout_sec <= 0:
        raise ValueError("TEDS timeout must be positive")
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


if __name__ == "__main__":
    main()
