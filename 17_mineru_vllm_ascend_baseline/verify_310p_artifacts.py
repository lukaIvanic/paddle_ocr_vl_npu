#!/usr/bin/env python3
"""Verify the exact MinerU and OmniDocBench artifacts for the 310P smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


REFERENCE_ROOT = Path(__file__).with_name("references") / "v16_1651_b626658"
EXPECTED_MODEL_MANIFEST_SHA256 = (
    "5e17a24da4023e2d3f4e7c51bf4b043f61cb353ec9039efe484dedf1f648afea"
)
EXPECTED_DATASET_JSON_SHA256 = (
    "a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496"
)
EXPECTED_IMAGE_MANIFEST_SHA256 = (
    "34f37943fc4469b1c01cb8589f7d9634d3285780421da78ed4bd4f0559c921fe"
)
EXPECTED_DATASET_PAGES = 1651
EXPECTED_OMNIDOCBENCH_COMMIT = "2b161d010d2e3aff77a0edef359ea3a6411d23cd"
EXPECTED_OMNIDOCBENCH_REMOTE_FRAGMENT = "opendatalab/OmniDocBench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--omnidocbench-repo", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_rows(rows: Iterable[list[Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(payload)
        digest.update(b"\n")
    return digest.hexdigest()


def expected_model_rows(reference_root: Path = REFERENCE_ROOT) -> list[list[Any]]:
    manifest = load_json(reference_root / "primary_model_manifest.json")
    return [
        [row["path"], row["size"], row["sha256"]]
        for row in sorted(manifest["files"], key=lambda item: item["path"])
    ]


def expected_image_rows(reference_root: Path = REFERENCE_ROOT) -> list[list[Any]]:
    manifest = load_json(reference_root / "combined_input_manifest.json")
    return [
        [row["dataset_index"], row["image"], row["size"], row["sha256"]]
        for row in manifest["pages"]
    ]


def verify_reference_bundle(reference_root: Path = REFERENCE_ROOT) -> None:
    model_digest = digest_rows(expected_model_rows(reference_root))
    image_digest = digest_rows(expected_image_rows(reference_root))
    prep = load_json(reference_root / "evaluation_prep_summary.json")
    problems = []
    if model_digest != EXPECTED_MODEL_MANIFEST_SHA256:
        problems.append(f"reference model digest changed: {model_digest}")
    if image_digest != EXPECTED_IMAGE_MANIFEST_SHA256:
        problems.append(f"reference image digest changed: {image_digest}")
    if prep.get("dataset_sha256") != EXPECTED_DATASET_JSON_SHA256:
        problems.append("reference dataset JSON digest changed")
    if prep.get("dataset_pages") != EXPECTED_DATASET_PAGES:
        problems.append("reference dataset page count changed")
    if prep.get("evaluator_commit") != EXPECTED_OMNIDOCBENCH_COMMIT:
        problems.append("reference evaluator commit changed")
    if problems:
        raise RuntimeError("; ".join(problems))


def verify_model(model_dir: Path) -> dict[str, Any]:
    expected_rows = expected_model_rows()
    expected = {row[0]: row for row in expected_rows}
    actual_paths = sorted(
        str(path.relative_to(model_dir))
        for path in model_dir.rglob("*")
        if path.is_file()
    )
    missing = sorted(set(expected) - set(actual_paths))
    unexpected = sorted(set(actual_paths) - set(expected))
    rows = []
    mismatches = []
    for name in sorted(expected):
        path = model_dir / name
        if not path.is_file():
            continue
        print(f"VERIFY_MODEL_FILE {name}", flush=True)
        actual = [name, path.stat().st_size, sha256_file(path)]
        rows.append(actual)
        if actual != expected[name]:
            mismatches.append(
                {
                    "path": name,
                    "expected_size": expected[name][1],
                    "actual_size": actual[1],
                    "expected_sha256": expected[name][2],
                    "actual_sha256": actual[2],
                }
            )
    manifest_sha256 = digest_rows(rows)
    return {
        "path": str(model_dir),
        "expected_file_count": len(expected_rows),
        "actual_file_count": len(actual_paths),
        "missing": missing,
        "unexpected": unexpected,
        "mismatches": mismatches,
        "manifest_sha256": manifest_sha256,
        "expected_manifest_sha256": EXPECTED_MODEL_MANIFEST_SHA256,
        "model_safetensors_sha256": next(
            (row[2] for row in rows if row[0] == "model.safetensors"),
            None,
        ),
        "match": (
            not missing
            and not unexpected
            and not mismatches
            and manifest_sha256 == EXPECTED_MODEL_MANIFEST_SHA256
        ),
    }


def dataset_image_name(sample: dict[str, Any]) -> str:
    value = (sample.get("page_info") or {}).get("image_path")
    if not value:
        raise ValueError("dataset sample has no page_info.image_path")
    return Path(value).name


def verify_dataset(dataset_json: Path, images_dir: Path) -> dict[str, Any]:
    dataset_sha256 = sha256_file(dataset_json)
    dataset = load_json(dataset_json)
    if not isinstance(dataset, list):
        raise TypeError("dataset JSON must contain a list")
    names = [dataset_image_name(sample) for sample in dataset]
    duplicate_names = sorted(
        name for name in set(names) if names.count(name) > 1
    )
    rows = []
    missing = []
    for index, name in enumerate(names):
        path = images_dir / name
        if not path.is_file():
            missing.append(name)
            continue
        if index % 100 == 0 or index + 1 == len(names):
            print(f"VERIFY_DATASET_IMAGES {index + 1}/{len(names)}", flush=True)
        rows.append([index, name, path.stat().st_size, sha256_file(path)])
    image_manifest_sha256 = digest_rows(rows)
    return {
        "dataset_json": str(dataset_json),
        "dataset_json_sha256": dataset_sha256,
        "expected_dataset_json_sha256": EXPECTED_DATASET_JSON_SHA256,
        "images_dir": str(images_dir),
        "page_count": len(dataset),
        "expected_page_count": EXPECTED_DATASET_PAGES,
        "duplicate_names": duplicate_names,
        "missing_count": len(missing),
        "missing_first": missing[:10],
        "image_manifest_sha256": image_manifest_sha256,
        "expected_image_manifest_sha256": EXPECTED_IMAGE_MANIFEST_SHA256,
        "match": (
            dataset_sha256 == EXPECTED_DATASET_JSON_SHA256
            and len(dataset) == EXPECTED_DATASET_PAGES
            and not duplicate_names
            and not missing
            and image_manifest_sha256 == EXPECTED_IMAGE_MANIFEST_SHA256
        ),
    }


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        stderr=subprocess.STDOUT,
    ).rstrip("\n")


def source_status_changes(status_lines: Iterable[str]) -> tuple[list[str], int]:
    source_dirty = []
    ignored_result_changes = 0
    for line in status_lines:
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path == "result" or path.startswith("result/"):
            ignored_result_changes += 1
        else:
            source_dirty.append(line)
    return source_dirty, ignored_result_changes


def verify_omnidocbench_repo(repo: Path) -> dict[str, Any]:
    commit = git_output(repo, "rev-parse", "HEAD")
    remote = git_output(repo, "remote", "get-url", "origin")
    status_lines = [
        line
        for line in git_output(
            repo,
            "status",
            "--short",
            "--untracked-files=no",
        ).splitlines()
        if line
    ]
    source_dirty, ignored_result_changes = source_status_changes(status_lines)
    return {
        "path": str(repo),
        "commit": commit,
        "expected_commit": EXPECTED_OMNIDOCBENCH_COMMIT,
        "origin": remote,
        "tracked_source_changes": source_dirty,
        "ignored_result_changes": ignored_result_changes,
        "match": (
            commit == EXPECTED_OMNIDOCBENCH_COMMIT
            and EXPECTED_OMNIDOCBENCH_REMOTE_FRAGMENT in remote
            and not source_dirty
        ),
    }


def main() -> None:
    args = parse_args()
    try:
        verify_reference_bundle()
        model_dir = args.model_dir.expanduser().resolve()
        dataset_json = args.dataset_json.expanduser().resolve()
        images_dir = args.images_dir.expanduser().resolve()
        omnidocbench_repo = args.omnidocbench_repo.expanduser().resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(model_dir)
        if not dataset_json.is_file():
            raise FileNotFoundError(dataset_json)
        if not images_dir.is_dir():
            raise FileNotFoundError(images_dir)
        if not omnidocbench_repo.is_dir():
            raise FileNotFoundError(omnidocbench_repo)
        result = {
            "model": verify_model(model_dir),
            "dataset": verify_dataset(dataset_json, images_dir),
            "omnidocbench_repo": verify_omnidocbench_repo(omnidocbench_repo),
        }
        result["status"] = (
            "PASS"
            if all(section["match"] for section in result.values())
            else "MISMATCH"
        )
        print("EXPERIMENT17_310P_ARTIFACTS " + json.dumps(result, sort_keys=True))
        raise SystemExit(0 if result["status"] == "PASS" else 1)
    except Exception as error:
        print(
            "EXPERIMENT17_310P_ARTIFACTS "
            + json.dumps(
                {
                    "status": "ERROR",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
