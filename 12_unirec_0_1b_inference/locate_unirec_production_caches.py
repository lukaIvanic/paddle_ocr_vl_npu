#!/usr/bin/env python3
"""Locate source-compatible K20 vision and compiled-FP32 layout caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import vision_bucket_presets  # noqa: E402
import vision_full_batch  # noqa: E402


LAYOUT_CONFIGURATION = (
    "depthwise_native_weightformat_native_readingorder_float32_"
    "frozenbn0_precomputedfrozenbn0_formattedfrozenbnbuffers0_"
    "msda_decomposed_cogview_direct_softmax"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        default=[],
        help="Directory to inspect recursively for cache roots. Repeatable.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        action="append",
        default=[],
        help="Directory whose command.sh files may name cache roots. Repeatable.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def existing_directory(path: Path) -> Path | None:
    candidate = path.expanduser().resolve()
    return candidate if candidate.is_dir() else None


def flag_value(words: list[str], flag: str) -> str | None:
    positions = [index for index, word in enumerate(words) if word == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(words):
        return None
    return words[positions[0] + 1]


def command_candidates(
    artifact_roots: list[Path],
) -> tuple[dict[Path, list[str]], dict[Path, list[str]], list[dict[str, Any]]]:
    compile_roots: dict[Path, list[str]] = {}
    layout_roots: dict[Path, list[str]] = {}
    commands = []
    for artifact_root in artifact_roots:
        root = existing_directory(artifact_root)
        if root is None:
            continue
        for command_path in root.rglob("command.sh"):
            try:
                words = shlex.split(command_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                commands.append(
                    {"path": str(command_path), "status": "unreadable", "error": str(error)}
                )
                continue
            compile_value = flag_value(words, "--compile-cache-dir")
            layout_value = flag_value(words, "--layout-cache-dir")
            row: dict[str, Any] = {"path": str(command_path), "status": "parsed"}
            if compile_value:
                compile_root = existing_directory(Path(compile_value))
                row["compile_cache_dir"] = compile_value
                row["compile_cache_exists"] = compile_root is not None
                if compile_root is not None:
                    compile_roots.setdefault(compile_root, []).append(str(command_path))
            if layout_value:
                layout_root = existing_directory(Path(layout_value))
                row["layout_cache_dir"] = layout_value
                row["layout_cache_exists"] = layout_root is not None
                if layout_root is not None:
                    layout_roots.setdefault(layout_root, []).append(str(command_path))
            commands.append(row)
    return compile_roots, layout_roots, commands


def direct_cache_candidates(
    search_roots: list[Path],
) -> tuple[dict[Path, list[str]], dict[Path, list[str]]]:
    compile_roots: dict[Path, list[str]] = {}
    layout_roots: dict[Path, list[str]] = {}
    for search_root in search_roots:
        root = existing_directory(search_root)
        if root is None:
            continue
        for directory, child_names, _files in os.walk(root):
            parent = Path(directory)
            retained_children = []
            for child_name in child_names:
                child = parent / child_name
                if child_name.startswith("vision_full_bucket_"):
                    compile_roots.setdefault(parent.resolve(), []).append(
                        f"filesystem:{child}"
                    )
                    continue
                if child_name == LAYOUT_CONFIGURATION:
                    layout_roots.setdefault(parent.resolve(), []).append(
                        f"filesystem:{child}"
                    )
                    continue
                retained_children.append(child_name)
            child_names[:] = retained_children
    return compile_roots, layout_roots


def merge_provenance(
    destination: dict[Path, list[str]], source: dict[Path, list[str]]
) -> None:
    for path, provenance in source.items():
        destination.setdefault(path, []).extend(provenance)


def k20_row(root: Path, provenance: list[str]) -> dict[str, Any]:
    specs = vision_bucket_presets.VISION_BUCKET_PRESETS["310p_k20_l4"]
    slots = vision_bucket_presets.assign_vision_bucket_cache_slots(
        specs,
        slot_count=max(10, len(specs)),
    )
    flat_keys = set(vision_full_batch.FLAT_GLOBAL_CONTEXT_BUCKET_KEYS)
    extended_keys = set(
        vision_full_batch.EXTENDED_FLAT_GLOBAL_CONTEXT_BUCKET_KEYS
    )
    buckets = {}
    newest_om_mtime_ns = 0
    for spec, slot in zip(specs, slots):
        key = spec.key
        if key in extended_keys:
            source_hash = (
                vision_full_batch._extended_flat_global_context_source_hash()
            )
            method = f"_forward_flat_bucket_slot_{slot}"
            mode = "direct_2d_extended"
        elif key in flat_keys:
            source_hash = vision_full_batch._flat_global_context_source_hash()
            method = f"_forward_flat_bucket_slot_{slot}"
            mode = "direct_2d_legacy"
        else:
            source_hash = vision_full_batch._source_hash()
            method = f"_forward_bucket_slot_{slot}"
            mode = "legacy_two_stage"
        directories = sorted(
            root.glob(
                f"vision_full_bucket_{key}_float16_src{source_hash}_"
                "dwconstant_grouped_all*wtorchair_internal*"
            )
        )
        modules = []
        oms = []
        for directory in directories:
            found = list(directory.glob(f"**/{method}/compiled_module"))
            modules.extend(found)
            for module in found:
                oms.extend(module.parent.glob("*.om"))
        for om in oms:
            newest_om_mtime_ns = max(newest_om_mtime_ns, om.stat().st_mtime_ns)
        buckets[key] = {
            "slot": slot,
            "method": method,
            "mode": mode,
            "source_hash": source_hash,
            "compiled_modules": [str(path) for path in sorted(set(modules))],
            "oms": [str(path) for path in sorted(set(oms))],
            "complete": bool(modules and oms),
        }
    missing = [key for key, row in buckets.items() if not row["complete"]]
    return {
        "root": str(root),
        "provenance": sorted(set(provenance)),
        "complete_bucket_count": len(buckets) - len(missing),
        "bucket_count": len(buckets),
        "missing_buckets": missing,
        "complete": not missing,
        "newest_om_mtime_ns": newest_om_mtime_ns,
        "buckets": buckets,
    }


def layout_row(root: Path, provenance: list[str]) -> dict[str, Any]:
    source_hash = hashlib.sha256(
        (SCRIPT_DIR / "layout_torchair.py").read_bytes()
    ).hexdigest()[:12]
    graph_root = (
        root
        / LAYOUT_CONFIGURATION
        / f"layout_b2_800x800_float32_src{source_hash}"
    )
    modules = sorted(set(graph_root.glob("**/forward/compiled_module")))
    oms = sorted(
        set(om for module in modules for om in module.parent.glob("*.om"))
    )
    newest_om_mtime_ns = max((path.stat().st_mtime_ns for path in oms), default=0)
    return {
        "root": str(root),
        "provenance": sorted(set(provenance)),
        "configuration": LAYOUT_CONFIGURATION,
        "source_hash": source_hash,
        "graph_root": str(graph_root),
        "compiled_modules": [str(path) for path in modules],
        "oms": [str(path) for path in oms],
        "complete": bool(modules and oms),
        "newest_om_mtime_ns": newest_om_mtime_ns,
    }


def select_latest_complete(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    complete = [row for row in rows if row["complete"]]
    if not complete:
        return None
    return max(
        complete,
        key=lambda row: (row["newest_om_mtime_ns"], row["root"]),
    )


def main() -> int:
    args = parse_args()
    search_roots = [path.expanduser().resolve() for path in args.search_root]
    artifact_roots = [path.expanduser().resolve() for path in args.artifact_root]
    command_compile, command_layout, commands = command_candidates(artifact_roots)
    direct_compile, direct_layout = direct_cache_candidates(search_roots)
    merge_provenance(command_compile, direct_compile)
    merge_provenance(command_layout, direct_layout)

    k20_candidates = [
        k20_row(path, provenance)
        for path, provenance in sorted(command_compile.items())
    ]
    layout_candidates = [
        layout_row(path, provenance)
        for path, provenance in sorted(command_layout.items())
    ]
    selected_k20 = select_latest_complete(k20_candidates)
    selected_layout = select_latest_complete(layout_candidates)
    payload = {
        "schema": "unirec_production_cache_locator_v1",
        "status": "ok" if selected_k20 and selected_layout else "missing",
        "search_roots": [str(path) for path in search_roots],
        "artifact_roots": [str(path) for path in artifact_roots],
        "selected_compile_cache": (
            selected_k20["root"] if selected_k20 is not None else None
        ),
        "selected_layout_cache_root": (
            selected_layout["root"] if selected_layout is not None else None
        ),
        "k20_candidates": k20_candidates,
        "layout_candidates": layout_candidates,
        "parsed_commands": commands,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for row in sorted(
        k20_candidates,
        key=lambda item: (-item["complete_bucket_count"], item["root"]),
    ):
        print(
            "UNIREC_CACHE_LOCATOR_K20_CANDIDATE "
            f"complete={int(row['complete'])} "
            f"buckets={row['complete_bucket_count']}/{row['bucket_count']} "
            f"missing={','.join(row['missing_buckets']) or 'none'} "
            f"root={row['root']}"
        )
    for row in sorted(
        layout_candidates,
        key=lambda item: (-int(item["complete"]), item["root"]),
    ):
        print(
            "UNIREC_CACHE_LOCATOR_LAYOUT_CANDIDATE "
            f"complete={int(row['complete'])} oms={len(row['oms'])} "
            f"root={row['root']}"
        )
    if selected_k20 is not None:
        print(f"UNIREC_K20_COMPILE_CACHE={selected_k20['root']}")
    if selected_layout is not None:
        print(f"UNIREC_FP32_B2_LAYOUT_CACHE={selected_layout['root']}")
    print(f"UNIREC_CACHE_LOCATOR_OUTPUT={args.output.resolve()}")
    if selected_k20 is None or selected_layout is None:
        return 1
    print("UNIREC_PRODUCTION_CACHE_LOCATOR: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
