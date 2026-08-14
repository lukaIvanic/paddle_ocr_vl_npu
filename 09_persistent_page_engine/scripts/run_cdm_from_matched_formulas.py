#!/usr/bin/env python3
"""Run CDM directly from a saved OmniDocBench display-formula match result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


EXPECTED_PDFLATEX = "1.40.28 (TeX Live 2025)"
EXPECTED_IMAGEMAGICK = "ImageMagick 7.1.1-47"
EXPECTED_GHOSTSCRIPT = "9.55.0"
REQUIRED_TEX_RESOURCES = {
    "CJK.sty": "25ea84e881c2fba7f86adfc747b5064ddcd9821c6cade7b5112187566468915a",
    "c70gkai.fd": "ca6b2a3acc180b7c43617d015c986fd0ac833ce5d863d0a1964e90c51024ce6e",
    "cmr10.tfm": "87f2d8981927644cbecaf3d639e96e348ea4e7be49d8804468bd8ba9ff3f5244",
    "cmsy10.tfm": "0ca13d421ac7133271aed7c935099ecf3d1d08ac9e15f81acb34a16564ab8a46",
    "cmex10.tfm": "0890bccea1dd4d27f001ac30e86c63af35bc803e0557c35aafb1903c8d208e92",
    "article.cls": "f9c8770a909e5b6526eb72284e457cd379b9bedbf3b508a96bd7f922d4cb24ec",
    "amsmath.sty": "4c0bfd7d67ea18214810a21244f8426da40d04af5bb90ed201b85cc5e211d321",
    "geometry.sty": "d5d36ad74051ad36288242b51438e2d9a5db2bd6c063b9b5704d0931fbc9f439",
    "xcolor.sty": "478888f8e8f00345fd645b6cdd6fa7db1e9efdd6f585b25c89bb3665d9029295",
}


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


def _command_output(argv: list[str]) -> str:
    result = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"CDM runtime command failed ({result.returncode}): {argv!r}\n{output}"
        )
    return output


def _tool_roots() -> list[Path]:
    script = Path(__file__).resolve()
    repo_tools = (
        script.parents[2] / ".runtime_cache/omnidocbench_eval/tools"
        if len(script.parents) > 2
        else None
    )
    values = (
        os.environ.get("OMNIDOCBENCH_EVAL_TOOLS_ROOT"),
        os.environ.get("OMNIDOCBENCH_TOOL_ROOT"),
        str(repo_tools) if repo_tools is not None else None,
        "/workspace/.tools/omnidocbench_eval",
    )
    roots: list[Path] = []
    for value in values:
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if path not in roots:
            roots.append(path)
    return roots


def _configure_cdm_runtime() -> dict[str, str]:
    """Select the frozen v1.6 tools and reject ambient/runtime drift."""
    tool_roots = _tool_roots()
    tex_bin_dirs = sorted(
        (path for root in tool_roots for path in root.glob("texlive/*/bin/*")),
        reverse=True,
    )
    imagemagick_bin_dirs = sorted(
        (path for root in tool_roots for path in root.glob("imagemagick-*/bin")),
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
            f"or install the persistent toolchain under one of {tool_roots}."
        )

    os.environ["CDM_PDFLATEX"] = str(pdflatex)
    os.environ["CDM_KPSEWHICH"] = str(kpsewhich)
    for resource, expected_sha256 in REQUIRED_TEX_RESOURCES.items():
        probe = subprocess.run(
            [str(kpsewhich), resource],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if probe.returncode != 0 or not probe.stdout.strip():
            raise RuntimeError(f"CDM TeX resource is unavailable: {resource}")
        resource_path = Path(probe.stdout.strip()).resolve()
        actual_sha256 = hashlib.sha256(resource_path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"CDM TeX resource drift: {resource} expected_sha256="
                f"{expected_sha256} actual_sha256={actual_sha256} "
                f"path={resource_path}"
            )

    magick = shutil.which("magick") or _first_executable(
        [path / "magick" for path in imagemagick_bin_dirs]
    )
    ghostscript = shutil.which("gs")
    if not magick or not ghostscript:
        raise RuntimeError(
            "CDM requires ImageMagick (magick/convert) and Ghostscript (gs); "
            f"resolved magick={magick!r}, gs={ghostscript!r}."
        )
    magick = str(Path(magick).resolve())
    ghostscript = str(Path(ghostscript).resolve())

    versions = {
        "pdflatex": _command_output([str(pdflatex), "--version"]),
        "imagemagick": _command_output([magick, "--version"]),
        "ghostscript": _command_output([ghostscript, "--version"]),
    }
    expected = {
        "pdflatex": EXPECTED_PDFLATEX,
        "imagemagick": EXPECTED_IMAGEMAGICK,
        "ghostscript": EXPECTED_GHOSTSCRIPT,
    }
    for name, needle in expected.items():
        if needle not in versions[name]:
            first_line = versions[name].splitlines()[0] if versions[name] else ""
            raise RuntimeError(
                f"CDM {name} version drift: expected {needle!r}, got {first_line!r}"
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
