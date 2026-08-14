#!/usr/bin/env python3
"""Select a deterministic performance-representative UniRec page subset."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import tarfile
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ARCHIVE = Path(__file__).with_name("references") / (
    "unirec_full1651_910b_470d8a6_text_outputs.tar.gz"
)

TEXT_LABELS = {
    "abstract",
    "algorithm",
    "content",
    "doc_title",
    "paragraph_title",
    "text",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--strata", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--random-restarts", type=int, default=1000)
    parser.add_argument("--local-search-passes", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-list", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def label_group(label: str) -> str:
    base, separator, suffix = label.rpartition("_")
    if not separator or not suffix.isdigit():
        base = label
    if base in TEXT_LABELS:
        return "text"
    if base in {"display_formula", "formula_number"}:
        return "formula"
    return base


def source_family(filename: str) -> str:
    if re.match(r"^page-[0-9a-f-]+\.", filename):
        return "page_uuid"
    if filename.startswith("color_textbook_"):
        return "color_textbook"
    return filename.split("_", 1)[0]


def document_id(filename: str) -> str:
    return re.sub(r"_page_\d+\.[^.]+$", "", filename)


def language_group(text: str) -> str:
    cjk = sum("\u3400" <= character <= "\u9fff" for character in text)
    latin = sum(character.isascii() and character.isalpha() for character in text)
    if cjk + latin < 20:
        return "sparse_or_symbolic"
    if cjk >= 2 * latin:
        return "cjk_dominant"
    if latin >= 2 * cjk:
        return "latin_dominant"
    return "mixed_language"


def load_workload(archive: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with tarfile.open(archive, "r:gz") as bundle:
        summary = json.load(bundle.extractfile("output/run_summary.json"))
        trace = [
            json.loads(line)
            for line in bundle.extractfile("output/recognition_trace.jsonl")
        ]
        page_records = []
        for member in bundle.getmembers():
            if not member.isfile():
                continue
            if not member.name.startswith("output/") or not member.name.endswith(
                ".json"
            ):
                continue
            if member.name.count("/") != 2:
                continue
            value = json.load(bundle.extractfile(member))
            page_records.append(value)

    page_records.sort(key=lambda value: Path(value["input_path"]).name)
    if len(page_records) != int(summary["page_count"]):
        raise RuntimeError(
            f"page JSON count mismatch: {len(page_records)} != {summary['page_count']}"
        )

    rows_by_page: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in trace:
        rows_by_page[int(row["page_index"])].append(row)

    pages: list[dict[str, Any]] = []
    for page_index, page_record in enumerate(page_records):
        filename = Path(page_record["input_path"]).name
        rows = rows_by_page[page_index]
        if rows and rows[0]["page"] != filename:
            raise RuntimeError(
                f"trace/page order mismatch at {page_index}: "
                f"{rows[0]['page']!r} != {filename!r}"
            )
        labels = collections.Counter(label_group(str(row["label"])) for row in rows)
        decoded_text = "\n".join(str(row.get("text", "")) for row in rows)
        encoder_lengths = [int(row["encoder_seq_len_hint"]) for row in rows]
        decode_lengths = [int(row["decode_token_count"]) for row in rows]
        processed_pixels = [
            int(row["processed_image_size"][0])
            * int(row["processed_image_size"][1])
            for row in rows
        ]
        pages.append(
            {
                "page_index": page_index,
                "filename": filename,
                "document_id": document_id(filename),
                "source_family": source_family(filename),
                "language_group": language_group(decoded_text),
                "width": int(page_record["width"]),
                "height": int(page_record["height"]),
                "page_pixels": int(page_record["width"])
                * int(page_record["height"]),
                "layout_blocks": len(page_record["recognition_results"]),
                "crop_count": len(rows),
                "encoder_tokens": sum(encoder_lengths),
                "max_encoder_tokens": max(encoder_lengths, default=0),
                "decode_tokens": sum(decode_lengths),
                "max_decode_tokens": max(decode_lengths, default=0),
                "processed_pixels": sum(processed_pixels),
                "cross_gt_512": sum(value > 512 for value in encoder_lengths),
                "cross_gt_768": sum(value > 768 for value in encoder_lengths),
                "cross_gt_1024": sum(value > 1024 for value in encoder_lengths),
                "length_cap_crops": sum(value >= 2047 for value in decode_lengths),
                "trace_prefill_s": sum(float(row["prefill_s"]) for row in rows),
                "label_counts": dict(sorted(labels.items())),
            }
        )

    if sum(page["crop_count"] for page in pages) != int(summary["crop_count"]):
        raise RuntimeError("trace crop count does not match run summary")
    return pages, summary


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def add_difficulty_and_strata(
    pages: list[dict[str, Any]], strata: int
) -> dict[str, dict[str, float]]:
    difficulty_features = {
        "crop_count": 0.8,
        "encoder_tokens": 1.2,
        "decode_tokens": 1.2,
        "max_encoder_tokens": 0.8,
        "max_decode_tokens": 0.8,
        "layout_blocks": 0.5,
        "page_pixels": 0.3,
    }
    rank_by_feature: dict[str, dict[int, float]] = {}
    for feature in difficulty_features:
        order = sorted(
            range(len(pages)),
            key=lambda index: (float(pages[index][feature]), pages[index]["filename"]),
        )
        rank_by_feature[feature] = {
            index: (position + 0.5) / len(order)
            for position, index in enumerate(order)
        }
    total_weight = sum(difficulty_features.values())
    for index, page in enumerate(pages):
        page["difficulty_score"] = sum(
            difficulty_features[feature] * rank_by_feature[feature][index]
            for feature in difficulty_features
        ) / total_weight
    ranked = sorted(
        range(len(pages)),
        key=lambda index: (pages[index]["difficulty_score"], pages[index]["filename"]),
    )
    for rank, index in enumerate(ranked):
        pages[index]["difficulty_rank"] = rank
        pages[index]["difficulty_stratum"] = min(
            strata - 1, rank * strata // len(pages)
        )
    return {
        feature: {
            "weight": weight,
            "p50": percentile((page[feature] for page in pages), 0.50),
            "p90": percentile((page[feature] for page in pages), 0.90),
            "p95": percentile((page[feature] for page in pages), 0.95),
            "p99": percentile((page[feature] for page in pages), 0.99),
        }
        for feature, weight in difficulty_features.items()
    }


def build_vectors(
    pages: list[dict[str, Any]], count: int
) -> tuple[list[dict[str, float]], dict[str, float], dict[str, float]]:
    continuous = (
        "crop_count",
        "encoder_tokens",
        "decode_tokens",
        "processed_pixels",
        "max_encoder_tokens",
        "max_decode_tokens",
        "layout_blocks",
        "page_pixels",
        "cross_gt_512",
        "cross_gt_768",
        "cross_gt_1024",
        "length_cap_crops",
        "trace_prefill_s",
    )
    weights = {feature: 1.0 for feature in continuous}
    weights.update(
        {
            "crop_count": 2.0,
            "encoder_tokens": 3.0,
            "decode_tokens": 3.0,
            "processed_pixels": 2.0,
            "cross_gt_512": 3.0,
            "cross_gt_768": 3.0,
            "cross_gt_1024": 4.0,
            "length_cap_crops": 2.0,
            "trace_prefill_s": 3.0,
        }
    )
    label_names = sorted(
        {
            label
            for page in pages
            for label in page["label_counts"]
        }
    )
    family_counts = collections.Counter(page["source_family"] for page in pages)
    family_names = sorted(name for name, value in family_counts.items() if value >= 10)
    language_names = sorted({page["language_group"] for page in pages})

    quantile_features = (
        "crop_count",
        "encoder_tokens",
        "decode_tokens",
        "max_encoder_tokens",
        "max_decode_tokens",
        "page_pixels",
    )
    thresholds = {
        (feature, probability): percentile(
            (page[feature] for page in pages), probability
        )
        for feature in quantile_features
        for probability in (0.90, 0.95, 0.99)
    }

    vectors: list[dict[str, float]] = []
    for page in pages:
        vector = {feature: float(page[feature]) for feature in continuous}
        vector["pages_with_cross_gt_512"] = float(page["cross_gt_512"] > 0)
        vector["pages_with_cross_gt_768"] = float(page["cross_gt_768"] > 0)
        vector["pages_with_cross_gt_1024"] = float(page["cross_gt_1024"] > 0)
        vector["pages_with_length_cap"] = float(page["length_cap_crops"] > 0)
        vector["zero_crop_pages"] = float(page["crop_count"] == 0)
        for name in label_names:
            vector[f"label_crops:{name}"] = float(page["label_counts"].get(name, 0))
            vector[f"pages_with_label:{name}"] = float(
                page["label_counts"].get(name, 0) > 0
            )
        for name in family_names:
            vector[f"source_family:{name}"] = float(page["source_family"] == name)
        for name in language_names:
            vector[f"language:{name}"] = float(page["language_group"] == name)
        for (feature, probability), threshold in thresholds.items():
            vector[f"tail:{feature}:p{int(probability * 100)}"] = float(
                float(page[feature]) >= threshold
            )
        vectors.append(vector)

    for key in vectors[0]:
        if key not in weights:
            if key.startswith("tail:"):
                weights[key] = 2.0
            elif key.startswith("source_family:"):
                weights[key] = 1.5
            elif key.startswith("language:"):
                weights[key] = 1.5
            elif key.startswith("pages_with_cross_gt_"):
                weights[key] = 3.0
            elif key.startswith("pages_with_label:"):
                weights[key] = 1.5
            else:
                weights[key] = 1.0

    fraction = count / len(pages)
    targets = {
        key: sum(vector[key] for vector in vectors) * fraction
        for key in vectors[0]
    }
    return vectors, targets, weights


def vector_sum(indices: Iterable[int], vectors: list[dict[str, float]]) -> dict[str, float]:
    result = {key: 0.0 for key in vectors[0]}
    for index in indices:
        for key, value in vectors[index].items():
            result[key] += value
    return result


def objective(
    sums: dict[str, float], targets: dict[str, float], weights: dict[str, float]
) -> float:
    total_weight = 0.0
    total = 0.0
    for key, target in targets.items():
        weight = weights[key]
        denominator = max(abs(target), 1.0)
        error = (sums[key] - target) / denominator
        total += weight * error * error
        total_weight += weight
    return total / total_weight


def select_pages(
    pages: list[dict[str, Any]],
    vectors: list[dict[str, float]],
    targets: dict[str, float],
    weights: dict[str, float],
    *,
    count: int,
    strata: int,
    seed: int,
    random_restarts: int,
    local_search_passes: int,
) -> tuple[list[int], float]:
    if count < strata or count % strata:
        raise ValueError("count must be a positive multiple of strata")
    quota = count // strata
    groups = {
        stratum: [
            index
            for index, page in enumerate(pages)
            if int(page["difficulty_stratum"]) == stratum
        ]
        for stratum in range(strata)
    }
    if any(len(group) < quota for group in groups.values()):
        raise RuntimeError("a difficulty stratum is smaller than its quota")

    random_source = random.Random(seed)
    best_indices: list[int] | None = None
    best_score = math.inf
    for _ in range(random_restarts):
        candidate = sorted(
            index
            for stratum in range(strata)
            for index in random_source.sample(groups[stratum], quota)
        )
        score = objective(vector_sum(candidate, vectors), targets, weights)
        if score < best_score:
            best_indices = candidate
            best_score = score
    if best_indices is None:
        raise RuntimeError("selection initialization failed")

    selected = set(best_indices)
    sums = vector_sum(selected, vectors)
    for _ in range(local_search_passes):
        changed = False
        for stratum in range(strata):
            selected_here = sorted(
                index
                for index in selected
                if int(pages[index]["difficulty_stratum"]) == stratum
            )
            available_here = [index for index in groups[stratum] if index not in selected]
            best_swap: tuple[int, int, dict[str, float], float] | None = None
            for old_index in selected_here:
                for new_index in available_here:
                    candidate_sums = {
                        key: sums[key]
                        - vectors[old_index][key]
                        + vectors[new_index][key]
                        for key in sums
                    }
                    score = objective(candidate_sums, targets, weights)
                    if score + 1e-15 < best_score:
                        best_score = score
                        best_swap = old_index, new_index, candidate_sums, score
            if best_swap is not None:
                old_index, new_index, sums, _ = best_swap
                selected.remove(old_index)
                selected.add(new_index)
                changed = True
        if not changed:
            break
    return sorted(selected), best_score


def metric_summary(pages: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "crop_count",
        "encoder_tokens",
        "decode_tokens",
        "processed_pixels",
        "max_encoder_tokens",
        "max_decode_tokens",
        "layout_blocks",
        "page_pixels",
        "cross_gt_512",
        "cross_gt_768",
        "cross_gt_1024",
        "length_cap_crops",
        "trace_prefill_s",
    )
    result: dict[str, Any] = {
        "page_count": len(pages),
        "metrics": {},
        "page_flags": {
            "cross_gt_512": sum(page["cross_gt_512"] > 0 for page in pages),
            "cross_gt_768": sum(page["cross_gt_768"] > 0 for page in pages),
            "cross_gt_1024": sum(page["cross_gt_1024"] > 0 for page in pages),
            "length_cap": sum(page["length_cap_crops"] > 0 for page in pages),
            "zero_crop": sum(page["crop_count"] == 0 for page in pages),
        },
        "label_crops": dict(
            sorted(
                sum(
                    (collections.Counter(page["label_counts"]) for page in pages),
                    collections.Counter(),
                ).items()
            )
        ),
        "label_pages": dict(
            sorted(
                {
                    label: sum(page["label_counts"].get(label, 0) > 0 for page in pages)
                    for label in {
                        value
                        for page in pages
                        for value in page["label_counts"]
                    }
                }.items()
            )
        ),
        "source_families": dict(
            sorted(collections.Counter(page["source_family"] for page in pages).items())
        ),
        "language_groups": dict(
            sorted(collections.Counter(page["language_group"] for page in pages).items())
        ),
    }
    for key in keys:
        values = [float(page[key]) for page in pages]
        result["metrics"][key] = {
            "sum": sum(values),
            "mean": sum(values) / len(values),
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values),
        }
    return result


def comparison_summary(
    full: dict[str, Any], selected: dict[str, Any]
) -> dict[str, Any]:
    metrics = {}
    for key, full_metric in full["metrics"].items():
        selected_metric = selected["metrics"][key]
        metrics[key] = {
            "mean_relative_error": (
                selected_metric["mean"] / full_metric["mean"] - 1.0
                if full_metric["mean"]
                else 0.0
            ),
            "p50_relative_error": (
                selected_metric["p50"] / full_metric["p50"] - 1.0
                if full_metric["p50"]
                else 0.0
            ),
            "p90_relative_error": (
                selected_metric["p90"] / full_metric["p90"] - 1.0
                if full_metric["p90"]
                else 0.0
            ),
            "p95_relative_error": (
                selected_metric["p95"] / full_metric["p95"] - 1.0
                if full_metric["p95"]
                else 0.0
            ),
            "p99_relative_error": (
                selected_metric["p99"] / full_metric["p99"] - 1.0
                if full_metric["p99"]
                else 0.0
            ),
        }
    return {"metrics": metrics}


def main() -> None:
    args = parse_args()
    if args.count <= 0 or args.strata <= 0:
        raise ValueError("count and strata must be positive")
    archive = args.reference_archive.expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    try:
        archive_label = str(archive.relative_to(repo_root))
    except ValueError:
        archive_label = str(archive)
    pages, run_summary = load_workload(archive)
    if args.count >= len(pages):
        raise ValueError("subset count must be smaller than the workload")
    difficulty_contract = add_difficulty_and_strata(pages, args.strata)
    vectors, targets, weights = build_vectors(pages, args.count)
    selected_indices, score = select_pages(
        pages,
        vectors,
        targets,
        weights,
        count=args.count,
        strata=args.strata,
        seed=args.seed,
        random_restarts=args.random_restarts,
        local_search_passes=args.local_search_passes,
    )
    selected_pages = [pages[index] for index in selected_indices]
    full_summary = metric_summary(pages)
    subset_summary = metric_summary(selected_pages)
    filenames = [page["filename"] for page in selected_pages]
    selection_sha256 = hashlib.sha256("\n".join(filenames).encode()).hexdigest()
    output = {
        "schema": "unirec_representative_pages_v1",
        "source": {
            "reference_archive": archive_label,
            "reference_archive_sha256": sha256_file(archive),
            "project_commit": "470d8a6d01d4682fa7e15aad915e5ddf697e2fe0",
            "page_count": int(run_summary["page_count"]),
            "crop_count": int(run_summary["crop_count"]),
            "cross_cache_length": int(run_summary["cross_cache_length"]),
            "self_cache_length": int(run_summary["self_cache_length"]),
            "layout_batch_size": int(run_summary["layout_batch_size"]),
        },
        "selection": {
            "count": args.count,
            "strata": args.strata,
            "pages_per_stratum": args.count // args.strata,
            "seed": args.seed,
            "random_restarts": args.random_restarts,
            "local_search_passes": args.local_search_passes,
            "objective": score,
            "selection_sha256": selection_sha256,
            "method": (
                "equal-quota composite-difficulty strata followed by deterministic "
                "distribution-matching local search"
            ),
            "difficulty_contract": difficulty_contract,
        },
        "full_workload": full_summary,
        "selected_workload": subset_summary,
        "comparison": comparison_summary(full_summary, subset_summary),
        "pages": selected_pages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_list = args.output_list or args.output.with_suffix(".txt")
    output_list.parent.mkdir(parents=True, exist_ok=True)
    output_list.write_text("\n".join(filenames) + "\n", encoding="utf-8")
    print(
        "UNIREC_REPRESENTATIVE_SUBSET PASS "
        f"pages={len(selected_pages)} objective={score:.8f} "
        f"crops={int(subset_summary['metrics']['crop_count']['sum'])} "
        f"encoder_tokens={int(subset_summary['metrics']['encoder_tokens']['sum'])} "
        f"decode_tokens={int(subset_summary['metrics']['decode_tokens']['sum'])} "
        f"cross_gt_512={int(subset_summary['metrics']['cross_gt_512']['sum'])} "
        f"cross_gt_768={int(subset_summary['metrics']['cross_gt_768']['sum'])} "
        f"cross_gt_1024={int(subset_summary['metrics']['cross_gt_1024']['sum'])} "
        f"manifest={args.output} list={output_list}"
    )


if __name__ == "__main__":
    main()
