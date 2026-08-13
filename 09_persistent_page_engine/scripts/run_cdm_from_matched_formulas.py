#!/usr/bin/env python3
"""Run CDM directly from a saved OmniDocBench display-formula match result."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--evaluator-root",
        type=Path,
        default=Path("/workspace/repos/OmniDocBench_eval"),
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--save-name", default="predictions_quick_match_cdm")
    parser.add_argument("--save-vis", action="store_true")
    return parser.parse_args()


def _prepend_path(path: Path) -> None:
    current = os.environ.get("PATH", "")
    entries = current.split(os.pathsep) if current else []
    value = str(path)
    if value not in entries:
        os.environ["PATH"] = os.pathsep.join([value, *entries])


def _first_executable(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def _configure_cdm_runtime() -> dict[str, str]:
    """Resolve the persistent 910B tools and reject silent zero-score CDM runs."""
    tool_root = Path(
        os.environ.get(
            "OMNIDOCBENCH_TOOL_ROOT",
            "/workspace/.tools/omnidocbench_eval",
        )
    )
    tex_bin_dirs = sorted(
        tool_root.glob("texlive/*/bin/*"),
        reverse=True,
    )
    imagemagick_bin_dirs = sorted(
        tool_root.glob("imagemagick-*/bin"),
        reverse=True,
    )

    for bin_dir in [*tex_bin_dirs, *imagemagick_bin_dirs]:
        if bin_dir.is_dir():
            _prepend_path(bin_dir)

    pdflatex = (
        Path(os.environ["CDM_PDFLATEX"])
        if os.environ.get("CDM_PDFLATEX")
        else None
    )
    if pdflatex is None or not pdflatex.is_file():
        found = shutil.which("pdflatex")
        pdflatex = Path(found) if found else _first_executable(
            [path / "pdflatex" for path in tex_bin_dirs]
        )
    kpsewhich = (
        Path(os.environ["CDM_KPSEWHICH"])
        if os.environ.get("CDM_KPSEWHICH")
        else None
    )
    if kpsewhich is None or not kpsewhich.is_file():
        found = shutil.which("kpsewhich")
        kpsewhich = Path(found) if found else _first_executable(
            [path / "kpsewhich" for path in tex_bin_dirs]
        )
    if pdflatex is None or kpsewhich is None:
        raise RuntimeError(
            "CDM requires pdflatex and kpsewhich. Set CDM_PDFLATEX/CDM_KPSEWHICH "
            f"or install the persistent toolchain under {tool_root}."
        )

    os.environ["CDM_PDFLATEX"] = str(pdflatex)
    os.environ["CDM_KPSEWHICH"] = str(kpsewhich)
    for resource in ("CJK.sty", "c70gkai.fd"):
        probe = subprocess.run(
            [str(kpsewhich), resource],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if probe.returncode != 0 or not probe.stdout.strip():
            raise RuntimeError(f"CDM TeX resource is unavailable: {resource}")

    magick = shutil.which("magick") or shutil.which("convert")
    ghostscript = shutil.which("gs")
    if not magick or not ghostscript:
        raise RuntimeError(
            "CDM requires ImageMagick (magick/convert) and Ghostscript (gs); "
            f"resolved magick={magick!r}, gs={ghostscript!r}."
        )
    return {
        "pdflatex": str(pdflatex),
        "kpsewhich": str(kpsewhich),
        "imagemagick": magick,
        "ghostscript": ghostscript,
    }


def main() -> None:
    args = _parse_args()
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
    runtime = _configure_cdm_runtime()
    print(f"[cdm-direct] runtime={json.dumps(runtime, sort_keys=True)}", flush=True)
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


if __name__ == "__main__":
    main()
