#!/usr/bin/env python3
"""Reusable NPU lab for PaddleOCR-VL text-prefill optimization experiments.

The replay mode prices an exact recorded workload. The profile mode measures
the same compiled graph repeatedly at one representative real length per
bucket. Device timing keeps the production boundary explicit: the headline
``text_prefill`` span is only the decoder transformer plus in-place KV writes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.modeling import (
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
)
from paddleocr_vl.model.text_decode import cast_decode_linear_weights_to_nz
from paddleocr_vl.model.text_prefill import (
    TEXT_PADDING_CHOICES,
    TextPrefillRuntime,
    parse_text_buckets,
    text_cache_dir_for_bucket,
)
from paddleocr_vl.serving.runtime_defaults import OPTIMIZED_TEXT_BUCKETS
from utils.timing import DeviceTimeline, synchronize


DEFAULT_MODEL = Path("/workspace/models/PaddleOCR-VL-1.6")
DEFAULT_CORPUS = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/text_lab"
    / "corpus_256p_minpixels_div4_5a37baf.json"
)
DEFAULT_REFERENCE = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/live_profile_router_h2d_concurrent_256p_5a37baf"
    / "run_summary.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp/09_persistent_page_engine/text_lab"
DEFAULT_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_text_torchair"
)
STAGES = (
    "text_token_embedding",
    "image_embed_scatter",
    "static_cache_alloc",
    "text_prefill_input_prep",
    "text_prefill",
    "prefill_lm_head",
    "prefill_argmax",
)


def _csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(piece) for piece in value.split(",") if piece.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "profile"), default="replay")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--reference-summary", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--cache-length", type=int, default=8192)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument(
        "--backend", choices=("torchair", "raw_eager"), default="torchair"
    )
    parser.add_argument("--padding", choices=TEXT_PADDING_CHOICES, default="auto")
    parser.add_argument(
        "--buckets", type=_csv_ints, default=tuple(OPTIMIZED_TEXT_BUCKETS)
    )
    parser.add_argument(
        "--profile-buckets",
        type=_csv_ints,
        help="Subset of routed compiled buckets to benchmark in profile mode.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--include-output-head",
        action="store_true",
        help="Also time the production LM-head and first-token argmax spans.",
    )
    parser.add_argument(
        "--allow-compile",
        action="store_true",
        help="Permit missing TorchAir graph caches to compile instead of failing.",
    )
    parser.add_argument("--name")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.cache_length <= 0:
        parser.error("--cache-length must be positive")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats positive")
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    args.buckets = parse_text_buckets(args.buckets)
    if args.profile_buckets is not None:
        missing = sorted(set(args.profile_buckets) - set(args.buckets))
        if missing:
            parser.error(f"--profile-buckets are absent from --buckets: {missing}")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _load_corpus(path: Path, max_items: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = path.expanduser().resolve()
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if corpus.get("kind") != "text_prefill_trace_replay":
        raise ValueError(f"not a text-prefill corpus: {path}")
    if not corpus.get("self_check", {}).get("passed"):
        raise ValueError(f"corpus self-check is not marked passed: {path}")
    items = list(corpus.get("items") or [])
    if not items:
        raise ValueError(f"corpus has no items: {path}")
    items.sort(key=lambda item: (int(item["source_index"]), int(item["source_line"])))
    if max_items is not None:
        items = items[:max_items]
    return corpus, items


def _reference_stages(path: Path) -> dict[str, float]:
    path = path.expanduser().resolve()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    recognition = payload.get("recognition") or {}
    return {
        str(name): float(seconds)
        for name, seconds in (recognition.get("device_stage_s") or {}).items()
    }


def _production_groups(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    seen: set[int] = set()
    for _key, members_iter in itertools.groupby(
        items,
        key=lambda item: item.get("production_group_id"),
    ):
        members = list(members_iter)
        group_id = members[0].get("production_group_id")
        if group_id is None:
            groups.extend([[member] for member in members])
            continue
        group_id = int(group_id)
        if group_id in seen:
            raise ValueError(
                f"production group {group_id} is non-contiguous after source ordering"
            )
        seen.add(group_id)
        groups.append(members)
    return groups


def _item_route_key(item: dict[str, Any]) -> str:
    bucket = item["route"].get("bucket")
    return "eager_overflow" if bucket is None else str(int(bucket))


@dataclass
class ResidentInput:
    item: dict[str, Any]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    projected_image_embeds: torch.Tensor


class TextPrefillLab:
    def __init__(
        self,
        args: argparse.Namespace,
        items: list[dict[str, Any]],
        corpus_distribution: dict[str, Any],
    ):
        import torch_npu  # noqa: F401

        self.args = args
        self.items = items
        self.corpus_distribution = corpus_distribution
        self.device = torch.device("npu:0")
        if not torch.npu.is_available():
            raise RuntimeError("text-prefill lab requires an available Ascend NPU")
        torch.npu.set_compile_mode(jit_compile=False)
        self.dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
        self.model_dir = _resolve_model_dir(args.model)

        synchronize(self.device)
        setup_started = time.perf_counter()
        model_started = time.perf_counter()
        self.model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(
            self.model_dir,
            dtype=self.dtype,
            device=self.device,
        ).eval()
        synchronize(self.device)
        self.model_load_s = time.perf_counter() - model_started

        format_started = time.perf_counter()
        self.weight_format = cast_decode_linear_weights_to_nz(self.model)
        synchronize(self.device)
        self.weight_format_s = time.perf_counter() - format_started
        self.linear_weight_format = str(self.weight_format["effective_mode"])

        runtime_buckets = tuple(args.buckets)
        if args.backend == "torchair":
            self._preflight_caches(runtime_buckets)

        runtime_started = time.perf_counter()
        self.runtime = TextPrefillRuntime(
            self.model,
            backend=args.backend,
            buckets=runtime_buckets,
            cache_root=args.cache_dir,
            cache_length=args.cache_length,
            device=self.device,
            dtype=self.dtype,
            model_dir=self.model_dir,
            linear_weight_format=self.linear_weight_format,
            padding=args.padding,
        )
        synchronize(self.device)
        self.runtime_setup_s = time.perf_counter() - runtime_started
        self.setup_s = time.perf_counter() - setup_started

        hidden_size = int(self.model.config.text_config.hidden_size)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(args.seed))
        embedding_std = float(self.model.model.embed_tokens.weight.float().std().item())
        basis = torch.randn((1, hidden_size), generator=generator, dtype=torch.float32)
        self.image_embedding_basis = (basis * embedding_std).to(
            device=self.device,
            dtype=self.dtype,
        )
        synchronize(self.device)

    def _preflight_caches(self, buckets: Iterable[int]) -> None:
        missing: list[Path] = []
        for bucket in buckets:
            cache_dir = text_cache_dir_for_bucket(
                self.args.cache_dir,
                bucket=int(bucket),
                cache_length=self.args.cache_length,
                dtype=self.dtype,
                device=self.device,
                model_dir=self.model_dir,
                linear_weight_format=self.linear_weight_format,
            )
            if not cache_dir.is_dir() or not any(cache_dir.iterdir()):
                missing.append(cache_dir)
        if missing and not self.args.allow_compile:
            rendered = "\n".join(f"  - {path}" for path in missing)
            raise RuntimeError(
                "missing compiled text-prefill graph caches; refusing accidental "
                f"compilation without --allow-compile:\n{rendered}"
            )

    def _resident(self, item: dict[str, Any]) -> ResidentInput:
        input_ids_cpu = torch.tensor([item["input_ids"]], dtype=torch.long)
        attention_mask_cpu = torch.ones_like(input_ids_cpu)
        grid_cpu = torch.tensor([item["grid_thw"]], dtype=torch.long)
        position_ids_cpu, _rope_deltas = self.model.get_rope_index(
            input_ids_cpu,
            grid_cpu,
            attention_mask_cpu,
        )
        image_tokens = int(
            (input_ids_cpu == int(self.model.config.image_token_id)).sum().item()
        )
        expected_image_tokens = int(item["projected_image_tokens"])
        if image_tokens != expected_image_tokens:
            raise RuntimeError(
                f"corpus image-token mismatch for {item['request_id']}: "
                f"ids={image_tokens} expected={expected_image_tokens}"
            )
        projected = self.image_embedding_basis.expand(image_tokens, -1).contiguous()
        return ResidentInput(
            item=item,
            input_ids=input_ids_cpu.to(self.device),
            attention_mask=attention_mask_cpu.to(self.device),
            position_ids=position_ids_cpu.to(self.device),
            projected_image_embeds=projected,
        )

    def _resident_group(self, items: list[dict[str, Any]]) -> list[ResidentInput]:
        resident = [self._resident(item) for item in items]
        synchronize(self.device)
        return resident

    @staticmethod
    def _stage_name(key: str) -> str:
        return key.rsplit("::", 1)[-1]

    def execute_group(
        self,
        items: list[dict[str, Any]],
        *,
        collect_records: bool,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        resident = self._resident_group(items)
        timeline = DeviceTimeline(self.device)
        held: list[Any] = []
        rows: list[dict[str, Any]] = []
        for member_index, source in enumerate(resident):
            prefix = f"{member_index:04d}::{source.item['source_index']:08d}"
            embeds = timeline.measure(
                f"{prefix}::text_token_embedding",
                lambda source=source: self.model.model.embed_tokens(source.input_ids),
            )

            def scatter(
                embeds: torch.Tensor = embeds,
                source: ResidentInput = source,
            ) -> torch.Tensor:
                image_mask = (
                    (source.input_ids == self.model.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(embeds)
                )
                return embeds.masked_scatter(image_mask, source.projected_image_embeds)

            embeds = timeline.measure(f"{prefix}::image_embed_scatter", scatter)
            cache = timeline.measure(
                f"{prefix}::static_cache_alloc",
                lambda embeds=embeds: self.model.allocate_static_cache(
                    batch_size=1,
                    cache_length=self.args.cache_length,
                    device=self.device,
                    dtype=embeds.dtype,
                    init_mode="empty",
                ),
            )
            route = self.runtime.route(int(embeds.shape[1]))
            if int(route["real_text_tokens"]) != int(source.item["input_tokens"]):
                raise RuntimeError(
                    f"runtime length drift for {source.item['request_id']}: "
                    f"runtime={route['real_text_tokens']} "
                    f"corpus={source.item['input_tokens']}"
                )
            prepared = timeline.measure(
                f"{prefix}::text_prefill_input_prep",
                lambda embeds=embeds, source=source, route=route: self.runtime.prepare(
                    embeds,
                    source.attention_mask,
                    source.position_ids,
                    route=route,
                ),
            )
            hidden = timeline.measure(
                f"{prefix}::text_prefill",
                lambda prepared=prepared, cache=cache: self.runtime.run_prepared(
                    prepared, cache
                ),
            )
            if self.args.include_output_head:
                logits = timeline.measure(
                    f"{prefix}::prefill_lm_head",
                    lambda hidden=hidden: self.model.lm_head(hidden),
                )
                token = timeline.measure(
                    f"{prefix}::prefill_argmax",
                    lambda logits=logits: torch.argmax(
                        logits[:, -1, :].float(), dim=-1, keepdim=True
                    ),
                )
                held.append(token)
            held.extend((embeds, cache, prepared, hidden))
            if collect_records:
                rows.append(
                    {
                        "source_index": int(source.item["source_index"]),
                        "request_id": source.item["request_id"],
                        "bucket": route.get("bucket"),
                        "execution": route["execution"],
                        "real_text_tokens": int(route["real_text_tokens"]),
                        "physical_text_tokens": int(route["physical_text_tokens"]),
                    }
                )
        spans = timeline.resolve_spans()
        stage_totals: dict[str, float] = defaultdict(float)
        row_by_index = {int(row["source_index"]): row for row in rows}
        for key, span in spans.items():
            stage = self._stage_name(key)
            seconds = float(span["seconds"])
            stage_totals[stage] += seconds
            if collect_records:
                source_index = int(key.split("::", 2)[1])
                row_by_index[source_index].setdefault("device_stage_s", {})[stage] = seconds
        del held, resident
        return dict(stage_totals), rows

    def replay(self) -> dict[str, Any]:
        groups = _production_groups(self.items)
        totals: dict[str, float] = defaultdict(float)
        rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        for group_index, group in enumerate(groups, 1):
            stage_totals, group_rows = self.execute_group(group, collect_records=True)
            for stage, seconds in stage_totals.items():
                totals[stage] += seconds
            rows.extend(group_rows)
            if group_index == 1 or group_index % 50 == 0 or group_index == len(groups):
                print(
                    f"REPLAY_PROGRESS groups={group_index}/{len(groups)} "
                    f"items={len(rows)}/{len(self.items)} "
                    f"text_prefill_s={totals.get('text_prefill', 0.0):.3f}",
                    flush=True,
                )
        replay_wall_s = time.perf_counter() - started
        return self._replay_report(groups, rows, dict(totals), replay_wall_s)

    def _replay_report(
        self,
        groups: list[list[dict[str, Any]]],
        rows: list[dict[str, Any]],
        totals: dict[str, float],
        replay_wall_s: float,
    ) -> dict[str, Any]:
        bucket_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = "eager_overflow" if row["bucket"] is None else str(row["bucket"])
            bucket_rows[key].append(row)
        by_bucket: dict[str, Any] = {}
        for key, members in sorted(
            bucket_rows.items(),
            key=lambda pair: (
                pair[0] == "eager_overflow",
                int(pair[0]) if pair[0].isdigit() else 0,
            ),
        ):
            transformer_s = sum(
                float(row["device_stage_s"].get("text_prefill", 0.0))
                for row in members
            )
            prep_s = sum(
                float(row["device_stage_s"].get("text_prefill_input_prep", 0.0))
                for row in members
            )
            real = sum(int(row["real_text_tokens"]) for row in members)
            physical = sum(int(row["physical_text_tokens"]) for row in members)
            by_bucket[key] = {
                "items": len(members),
                "real_text_tokens": real,
                "physical_text_tokens": physical,
                "text_prefill_s": transformer_s,
                "text_prefill_input_prep_s": prep_s,
                "mean_text_prefill_ms": transformer_s * 1000.0 / len(members),
                "effective_tok_per_s": real / transformer_s,
                "physical_tok_per_s": physical / transformer_s,
            }
        real_total = sum(int(row["real_text_tokens"]) for row in rows)
        physical_total = sum(int(row["physical_text_tokens"]) for row in rows)
        transformer_s = float(totals.get("text_prefill", 0.0))
        reference = _reference_stages(self.args.reference_summary)
        reference_compatible = (
            len(rows) == int(self.corpus_distribution["items"])
            and real_total == int(self.corpus_distribution["real_text_tokens"])
            and physical_total == int(self.corpus_distribution["physical_text_tokens"])
            and self.args.backend == "torchair"
        )
        comparison = {
            stage: {
                "lab_s": float(totals.get(stage, 0.0)),
                "reference_s": float(reference[stage]),
                "ratio": float(totals.get(stage, 0.0)) / float(reference[stage]),
                "delta_s": float(totals.get(stage, 0.0)) - float(reference[stage]),
            }
            for stage in STAGES
            if reference_compatible and stage in reference and stage in totals
        }
        transformer_ratio = comparison.get("text_prefill", {}).get("ratio")
        return {
            "mode": "replay",
            "workload": {
                "items": len(rows),
                "groups": len(groups),
                "real_text_tokens": real_total,
                "physical_text_tokens": physical_total,
                "padding_text_tokens": physical_total - real_total,
                "useful_token_fraction": real_total / physical_total,
            },
            "device_stage_s": totals,
            "throughput": {
                "effective_text_tok_per_s": real_total / transformer_s,
                "physical_text_tok_per_s": physical_total / transformer_s,
            },
            "by_bucket": by_bucket,
            "reference": {
                "summary_path": str(self.args.reference_summary.expanduser().resolve()),
                "compatible_workload": reference_compatible,
                "device_stage_s": reference,
                "comparison": comparison,
                "text_prefill_calibration": {
                    "ratio": transformer_ratio,
                    "within_10_percent": (
                        transformer_ratio is not None
                        and 0.9 <= float(transformer_ratio) <= 1.1
                    ),
                },
            },
            "wall_s": replay_wall_s,
            "records": rows,
        }

    def profile(self) -> dict[str, Any]:
        by_bucket: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in self.items:
            bucket = item["route"].get("bucket")
            if bucket is not None:
                by_bucket[int(bucket)].append(item)
        selected = (
            tuple(self.args.profile_buckets)
            if self.args.profile_buckets is not None
            else tuple(sorted(by_bucket))
        )
        missing = [bucket for bucket in selected if bucket not in by_bucket]
        if missing:
            raise ValueError(f"corpus has no representative for buckets: {missing}")

        results: dict[str, Any] = {}
        for bucket in selected:
            representative = min(
                by_bucket[bucket],
                key=lambda item: (
                    int(bucket) - int(item["input_tokens"]),
                    int(item["source_index"]),
                ),
            )
            resident = self._resident_group([representative])[0]
            embeds = self.model.model.embed_tokens(resident.input_ids)
            image_mask = (
                (resident.input_ids == self.model.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(embeds)
            )
            embeds = embeds.masked_scatter(image_mask, resident.projected_image_embeds)
            route = self.runtime.route(int(embeds.shape[1]))
            prepared = self.runtime.prepare(
                embeds, resident.attention_mask, resident.position_ids, route=route
            )
            cache = self.model.allocate_static_cache(
                batch_size=1,
                cache_length=self.args.cache_length,
                device=self.device,
                dtype=self.dtype,
                init_mode="empty",
            )
            synchronize(self.device)
            for _ in range(self.args.warmup):
                self.runtime.run_prepared(prepared, cache)
            synchronize(self.device)
            durations: list[float] = []
            for repeat in range(self.args.repeats):
                timeline = DeviceTimeline(self.device)
                output = timeline.measure(
                    f"repeat{repeat:04d}::text_prefill",
                    lambda prepared=prepared, cache=cache: self.runtime.run_prepared(
                        prepared, cache
                    ),
                )
                durations.append(timeline.resolve()[f"repeat{repeat:04d}::text_prefill"])
                del output
            total_s = sum(durations)
            real = int(route["real_text_tokens"])
            physical = int(route["physical_text_tokens"])
            results[str(bucket)] = {
                "representative_request_id": representative["request_id"],
                "real_text_tokens": real,
                "physical_text_tokens": physical,
                "padding_text_tokens": physical - real,
                "warmup": self.args.warmup,
                "repeats": self.args.repeats,
                "latency_ms": {
                    "min": min(durations) * 1000.0,
                    "median": statistics.median(durations) * 1000.0,
                    "p95": _percentile(durations, 0.95) * 1000.0,
                    "max": max(durations) * 1000.0,
                },
                "effective_tok_per_s": real * self.args.repeats / total_s,
                "physical_tok_per_s": physical * self.args.repeats / total_s,
                "samples_s": durations,
            }
            print(
                f"PROFILE bucket={bucket} real={real} "
                f"median_ms={statistics.median(durations) * 1000.0:.3f} "
                f"physical_tok_s={physical * self.args.repeats / total_s:.1f}",
                flush=True,
            )
            del resident, embeds, prepared, cache
        return {"mode": "profile", "buckets": results}


def _write_report(
    args: argparse.Namespace,
    corpus: dict[str, Any],
    lab: TextPrefillLab,
    result: dict[str, Any],
) -> Path:
    if args.output is not None:
        output = args.output.expanduser().resolve()
    else:
        name = args.name or time.strftime(f"{args.mode}_%Y%m%d_%H%M%S")
        output = (DEFAULT_OUTPUT_ROOT / f"{name}.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": "text_prefill_lab_result",
        "configuration": {
            "mode": args.mode,
            "corpus": str(args.corpus.expanduser().resolve()),
            "corpus_sha256": _sha256(args.corpus.expanduser().resolve()),
            "model": str(lab.model_dir),
            "device": str(lab.device),
            "dtype": str(lab.dtype),
            "backend": args.backend,
            "padding": args.padding,
            "buckets": list(lab.runtime.buckets),
            "cache_length": args.cache_length,
            "cache_dir": str(args.cache_dir.expanduser().resolve()),
            "allow_compile": bool(args.allow_compile),
            "include_output_head": bool(args.include_output_head),
            "embedding_policy": "real_token_embedding_plus_fixed_seed_synthetic_image_values",
            "seed": args.seed,
        },
        "corpus_contract": corpus.get("contract"),
        "setup": {
            "total_s": lab.setup_s,
            "model_load_s": lab.model_load_s,
            "weight_format_s": lab.weight_format_s,
            "runtime_setup_s": lab.runtime_setup_s,
            "weight_format": lab.weight_format,
            "runtime_metadata": lab.runtime.metadata,
        },
        "result": result,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def _print_replay(result: dict[str, Any]) -> None:
    workload = result["workload"]
    throughput = result["throughput"]
    print(
        "TEXT_REPLAY "
        f"items={workload['items']} groups={workload['groups']} "
        f"real={workload['real_text_tokens']} "
        f"physical={workload['physical_text_tokens']} "
        f"useful={workload['useful_token_fraction']:.4f}"
    )
    for stage in STAGES:
        if stage in result["device_stage_s"]:
            print(f"TEXT_STAGE {stage}={result['device_stage_s'][stage]:.6f}s")
    print(
        "TEXT_THROUGHPUT "
        f"effective={throughput['effective_text_tok_per_s']:.1f}tok/s "
        f"physical={throughput['physical_text_tok_per_s']:.1f}tok/s"
    )
    calibration = result["reference"]["text_prefill_calibration"]
    if calibration["ratio"] is not None:
        print(
            "TEXT_CALIBRATION "
            f"ratio={calibration['ratio']:.4f} "
            f"within_10_percent={calibration['within_10_percent']}"
        )


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    corpus, items = _load_corpus(args.corpus, args.max_items)
    lab = TextPrefillLab(args, items, dict(corpus["distribution"]))
    result = lab.replay() if args.mode == "replay" else lab.profile()
    output = _write_report(args, corpus, lab, result)
    if args.mode == "replay":
        _print_replay(result)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
