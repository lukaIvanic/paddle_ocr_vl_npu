#!/usr/bin/env python3
"""Fingerprint every practical input to the OmniDocBench CDM runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "09_persistent_page_engine/scripts"))
from run_cdm_from_matched_formulas import _configure_cdm_runtime  # noqa: E402


PACKAGES = (
    "numpy",
    "Pillow",
    "scipy",
    "opencv-python",
    "opencv-python-headless",
    "python-Levenshtein",
    "latex2text",
    "pylatexenc",
)
EVALUATOR_FILES = (
    "src/metrics/cal_metric.py",
    "src/metrics/cdm_metric.py",
    "src/metrics/cdm/cdm.py",
    "src/metrics/cdm/modules/latex2bbox_color.py",
    "src/metrics/cdm/modules/latex_processor.py",
    "src/metrics/cdm/modules/ransac.py",
    "src/metrics/cdm/modules/visual_matcher.py",
    "src/metrics/cdm/modules/texlive_env.py",
)
TEX_RESOURCES = (
    "CJK.sty",
    "c70gkai.fd",
    "gkai00.tfm",
    "gkai01.tfm",
    "cmr10.tfm",
    "cmsy10.tfm",
    "cmex10.tfm",
    "article.cls",
    "amsmath.sty",
    "geometry.sty",
    "xcolor.sty",
)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(argv: list[str], *, timeout: float = 30) -> dict[str, Any]:
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "argv": argv,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as exc:
        return {"argv": argv, "error": f"{type(exc).__name__}: {exc}"}


def executable(path: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    result: dict[str, Any] = {
        "requested": path,
        "resolved": str(resolved),
        "exists": resolved.is_file(),
    }
    if resolved.is_file():
        result.update({"bytes": resolved.stat().st_size, "sha256": sha256(resolved)})
        result["ldd"] = command(["ldd", str(resolved)])
    return result


def git_fingerprint(root: Path) -> dict[str, Any]:
    return {
        "commit": command(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "status": command(["git", "-C", str(root), "status", "--porcelain=v1"]),
        "diff": command(["git", "-C", str(root), "diff", "--binary"]),
    }


def main() -> None:
    parsed = args()
    evaluator = parsed.evaluator_root.resolve()
    output = parsed.output.resolve()
    if not (evaluator / "pdf_validation.py").is_file():
        raise FileNotFoundError(f"invalid evaluator root: {evaluator}")

    runtime = _configure_cdm_runtime()
    sys.path.insert(0, str(evaluator))
    from src.metrics.cdm.modules.texlive_env import describe_tex_runtime

    tex = describe_tex_runtime()
    kpsewhich = runtime["kpsewhich"]
    resource_rows: dict[str, Any] = {}
    for name in TEX_RESOURCES:
        found = command([kpsewhich, name])
        path_text = str(found.get("stdout", "")).strip()
        row: dict[str, Any] = {"lookup": found, "path": path_text or None}
        if path_text and Path(path_text).is_file():
            path = Path(path_text).resolve()
            row.update({"resolved": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
        resource_rows[name] = row

    evaluator_files: dict[str, Any] = {}
    for relative in EVALUATOR_FILES:
        path = evaluator / relative
        evaluator_files[relative] = (
            None
            if not path.is_file()
            else {"bytes": path.stat().st_size, "sha256": sha256(path)}
        )

    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    relevant_env = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(("CDM_", "TEX", "MAGICK", "GS_", "LC_"))
        or key in {"LANG", "PATH", "LD_LIBRARY_PATH", "OMNIDOCBENCH_TOOL_ROOT"}
    }
    tools = {name: executable(path) for name, path in runtime.items()}
    versions = {
        "pdflatex": command([runtime["pdflatex"], "--version"]),
        "kpsewhich": command([runtime["kpsewhich"], "--version"]),
        "imagemagick": command([runtime["imagemagick"], "-version"]),
        "imagemagick_configure": command([runtime["imagemagick"], "-list", "configure"]),
        "imagemagick_policy": command([runtime["imagemagick"], "-list", "policy"]),
        "imagemagick_delegates": command([runtime["imagemagick"], "-list", "delegate"]),
        "ghostscript": command([runtime["ghostscript"], "--version"]),
        "locale": command(["locale"]),
        "fc_match_serif": command(["fc-match", "serif"]),
        "fc_match_sans": command(["fc-match", "sans"]),
        "fc_match_monospace": command(["fc-match", "monospace"]),
        "pip_freeze": command([sys.executable, "-m", "pip", "freeze"], timeout=60),
    }
    result = {
        "schema": 1,
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "uname": list(platform.uname()),
        },
        "evaluator_root": str(evaluator),
        "evaluator_git": git_fingerprint(evaluator),
        "evaluator_files": evaluator_files,
        "python_packages": packages,
        "runtime_paths": runtime,
        "runtime_tools": tools,
        "runtime_versions": versions,
        "tex_runtime": tex,
        "tex_resources": resource_rows,
        "environment": relevant_env,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(
        "UNIREC_CDM_RUNTIME_FINGERPRINT PASS "
        f"machine={platform.machine()} evaluator="
        f"{result['evaluator_git']['commit'].get('stdout', '').strip()} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
