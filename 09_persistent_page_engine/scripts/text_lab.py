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
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
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
from paddleocr_vl.model.compile_utils import (
    TORCHAIR_EXECUTION_MODE,
    cache_key_part,
    import_torchair,
    short_file_hash,
    torch_npu_version_label,
    torchair_version_label,
)
from paddleocr_vl.model.text_cache_lease import (
    PackedKVCacheLease,
    PackedKVCachePool,
    static_cache_nbytes,
)
from paddleocr_vl.model.text_decode import (
    LocalPaddleOCRVLStaticCache,
    TextDecodeStage,
    cast_decode_linear_weights_to_nz,
)
from paddleocr_vl.model.text_direct_arena_prefill import (
    DirectArenaPackedTextPrefillRuntime,
)
from paddleocr_vl.model.text_packed_prefill import (
    PackedTextPrefillRuntime,
    PreparedPackedTextPrefill,
)
from paddleocr_vl.model.text_prefill import (
    TEXT_PADDING_CHOICES,
    TextPrefillStage,
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
DEFAULT_PACKED_CACHE_ROOT = REPO_ROOT / ".runtime_cache/09_text_lab_packed"
DEFAULT_PRODUCTION_PACKED_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_text_packed_torchair"
)
DEFAULT_DIRECT_ARENA_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_text_lab_direct_arena"
)
DEFAULT_PACKED_E2E_SUMMARY = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/text_pack_ab_on_256p_0fd1fbb"
    / "run_summary.json"
)
DEFAULT_BASELINE_LAB_RESULT = (
    DEFAULT_OUTPUT_ROOT / "replay_256p_9e14e4e" / "result.json"
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
    parser.add_argument(
        "--mode",
        choices=("replay", "profile", "packed", "direct_arena", "cache_lease"),
        default="replay",
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--reference-summary", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--packed-cache-dir", type=Path, default=DEFAULT_PACKED_CACHE_ROOT
    )
    parser.add_argument(
        "--production-packed-cache-dir",
        type=Path,
        default=DEFAULT_PRODUCTION_PACKED_CACHE_ROOT,
    )
    parser.add_argument(
        "--direct-arena-cache-dir",
        type=Path,
        default=DEFAULT_DIRECT_ARENA_CACHE_ROOT,
    )
    parser.add_argument(
        "--packed-e2e-summary",
        type=Path,
        default=DEFAULT_PACKED_E2E_SUMMARY,
    )
    parser.add_argument(
        "--baseline-lab-result", type=Path, default=DEFAULT_BASELINE_LAB_RESULT
    )
    parser.add_argument("--cache-length", type=int, default=8192)
    parser.add_argument("--arena-batch-size", type=int, default=16)
    parser.add_argument(
        "--ready-buffer-capacity",
        type=int,
        help="Ready-reservoir size for packed-lease HBM projection; default 4x arena.",
    )
    parser.add_argument(
        "--lease-decode-steps",
        type=int,
        default=3,
        help="Eager decode steps used to validate leased-prefix token parity.",
    )
    parser.add_argument("--pack-length", type=int, default=1024)
    parser.add_argument("--max-pack-members", type=int, default=32)
    parser.add_argument(
        "--pack-scope",
        action="append",
        choices=("production_group", "global"),
        help="Repeat to benchmark both realistic and corpus-wide pack formation.",
    )
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
    if args.arena_batch_size <= 0:
        parser.error("--arena-batch-size must be positive")
    if args.ready_buffer_capacity is not None and args.ready_buffer_capacity <= 0:
        parser.error("--ready-buffer-capacity must be positive")
    if args.lease_decode_steps < 0:
        parser.error("--lease-decode-steps must be non-negative")
    if args.pack_length <= 0 or args.max_pack_members <= 0:
        parser.error("--pack-length and --max-pack-members must be positive")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats positive")
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    args.buckets = parse_text_buckets(args.buckets)
    if args.profile_buckets is not None:
        missing = sorted(set(args.profile_buckets) - set(args.buckets))
        if missing:
            parser.error(f"--profile-buckets are absent from --buckets: {missing}")
    args.pack_scope = tuple(
        dict.fromkeys(args.pack_scope or ("production_group", "global"))
    )
    if (
        args.mode in ("packed", "direct_arena", "cache_lease")
        and args.backend != "torchair"
    ):
        parser.error(f"{args.mode} mode currently requires --backend torchair")
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
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    ungrouped: list[dict[str, Any]] = []
    for item in items:
        group_id = item.get("production_group_id")
        if group_id is None:
            ungrouped.append(item)
        else:
            grouped[int(group_id)].append(item)
    groups = [
        sorted(members, key=lambda item: int(item["source_index"]))
        for _group_id, members in sorted(grouped.items())
    ]
    groups.extend([[item] for item in ungrouped])
    if sum(len(group) for group in groups) != len(items):
        raise AssertionError("production grouping lost corpus items")
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


@dataclass
class PackedInput:
    members: list[dict[str, Any]]
    inputs_embeds: torch.Tensor
    position_ids: torch.Tensor
    segment_ids: torch.Tensor
    local_positions: torch.Tensor
    last_token_indices: torch.Tensor
    real_tokens: int


class PackedTextPrefillStage(TextPrefillStage):
    """One B=1 text graph containing causally isolated request segments."""

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        local_positions: torch.Tensor,
        last_token_indices: torch.Tensor,
        *flat_cache_tensors: torch.Tensor,
    ) -> torch.Tensor:
        key_caches = tuple(flat_cache_tensors[: self.num_layers])
        value_caches = tuple(flat_cache_tensors[self.num_layers :])
        valid = segment_ids >= 0
        allowed = (
            valid[:, None]
            & valid[None, :]
            & (segment_ids[:, None] == segment_ids[None, :])
            & (local_positions[:, None] >= local_positions[None, :])
        )
        attention_mask = torch.zeros(
            (1, 1, inputs_embeds.shape[1], inputs_embeds.shape[1]),
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        ).masked_fill(
            ~allowed[None, None],
            torch.finfo(inputs_embeds.dtype).min,
        )
        position_embeddings = self.text_model.rotary_emb(
            inputs_embeds, position_ids
        )
        hidden_states = inputs_embeds
        for layer_index, layer in enumerate(self.text_model.layers):
            residual = hidden_states
            attention_input = layer.input_layernorm(hidden_states)
            attention_output = self._attention(
                layer.self_attn,
                attention_input,
                attention_mask,
                position_embeddings,
                key_caches[layer_index],
                value_caches[layer_index],
            )
            hidden_states = layer.apply_blocks(residual, attention_output)
        hidden_states = self.text_model.norm(hidden_states)
        return torch.index_select(hidden_states, 1, last_token_indices)


def _first_fit_decreasing(
    items: list[dict[str, Any]],
    *,
    capacity: int,
    max_members: int,
) -> list[list[dict[str, Any]]]:
    packs: list[list[dict[str, Any]]] = []
    totals: list[int] = []
    for item in sorted(
        items,
        key=lambda value: (
            -int(value["input_tokens"]),
            int(value["source_index"]),
        ),
    ):
        length = int(item["input_tokens"])
        choices = [
            index
            for index, pack in enumerate(packs)
            if len(pack) < max_members and totals[index] + length <= capacity
        ]
        if choices:
            selected = max(choices, key=lambda index: totals[index])
            packs[selected].append(item)
            totals[selected] += length
        else:
            packs.append([item])
            totals.append(length)
    if sum(len(pack) for pack in packs) != len(items):
        raise AssertionError("text pack formation lost corpus items")
    return packs


def _form_text_packs(
    items: list[dict[str, Any]],
    *,
    scope: str,
    capacity: int,
    max_members: int,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    eligible = [item for item in items if int(item["input_tokens"]) <= capacity]
    overflow = [item for item in items if int(item["input_tokens"]) > capacity]
    if scope == "global":
        packs = _first_fit_decreasing(
            eligible,
            capacity=capacity,
            max_members=max_members,
        )
    elif scope == "production_group":
        packs = []
        eligible_ids = {int(item["source_index"]) for item in eligible}
        for group in _production_groups(items):
            members = [
                item
                for item in group
                if int(item["source_index"]) in eligible_ids
            ]
            if members:
                packs.extend(
                    _first_fit_decreasing(
                        members,
                        capacity=capacity,
                        max_members=max_members,
                    )
                )
    else:
        raise ValueError(f"unknown text packing scope: {scope!r}")
    return packs, overflow


def _packed_cache_dir(
    cache_root: Path,
    *,
    pack_length: int,
    max_members: int,
    dtype: torch.dtype,
    device: torch.device,
    model_dir: Path,
    linear_weight_format: str,
) -> Path:
    key = "_".join(
        (
            "text_packed_block_causal",
            f"mode{cache_key_part(TORCHAIR_EXECUTION_MODE)}",
            "bs1",
            f"seq{pack_length}",
            f"members{max_members}",
            f"cache{pack_length}",
            f"weights{cache_key_part(linear_weight_format)}",
            f"dtype{cache_key_part(dtype)}",
            f"model{short_file_hash(model_dir / 'config.json')}",
            f"torch{cache_key_part(torch.__version__)}",
            f"torchnpu{torch_npu_version_label(device)}",
            f"torchair{torchair_version_label(device)}",
            f"src{short_file_hash(Path(__file__).resolve())}",
        )
    )
    return cache_root.expanduser().resolve() / key


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

    def _setup_packed_graph(self) -> tuple[Any, Any, dict[str, Any]]:
        torchair, CompilerConfig = import_torchair()
        cache_dir = _packed_cache_dir(
            self.args.packed_cache_dir,
            pack_length=self.args.pack_length,
            max_members=self.args.max_pack_members,
            dtype=self.dtype,
            device=self.device,
            model_dir=self.model_dir,
            linear_weight_format=self.linear_weight_format,
        )
        cache_was_warm = cache_dir.is_dir() and any(cache_dir.iterdir())
        if not cache_was_warm and not self.args.allow_compile:
            raise RuntimeError(
                "missing packed text graph; pass --allow-compile for this "
                f"intentional new graph: {cache_dir}"
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        stage = PackedTextPrefillStage(self.model).eval()
        config = CompilerConfig()
        synchronize(self.device)
        wrapper_started = time.perf_counter()
        compiled = torchair.inference.cache_compile(
            stage.forward,
            config=config,
            dynamic=False,
            cache_dir=str(cache_dir),
            ge_cache=True,
        )
        synchronize(self.device)
        wrapper_s = time.perf_counter() - wrapper_started

        length = self.args.pack_length
        hidden_size = int(self.model.config.text_config.hidden_size)
        warm_embeds = torch.zeros(
            (1, length, hidden_size), device=self.device, dtype=self.dtype
        )
        warm_positions = torch.zeros(
            (3, 1, length), device=self.device, dtype=torch.int64
        )
        warm_segments = torch.zeros(
            (length,), device=self.device, dtype=torch.int64
        )
        warm_local_positions = torch.arange(
            length, device=self.device, dtype=torch.int64
        )
        warm_last_indices = torch.zeros(
            (self.args.max_pack_members,),
            device=self.device,
            dtype=torch.int64,
        )
        scratch_cache = self.model.allocate_static_cache(
            batch_size=1,
            cache_length=length,
            device=self.device,
            dtype=self.dtype,
            init_mode="empty",
        )
        synchronize(self.device)
        first_call_started = time.perf_counter()
        warm_output = compiled(
            warm_embeds,
            warm_positions,
            warm_segments,
            warm_local_positions,
            warm_last_indices,
            *scratch_cache.flat_tensors(),
        )
        synchronize(self.device)
        first_call_s = time.perf_counter() - first_call_started
        expected_shape = (1, self.args.max_pack_members, hidden_size)
        if tuple(warm_output.shape) != expected_shape:
            raise RuntimeError(
                "packed graph returned the wrong shape: "
                f"expected={expected_shape} got={tuple(warm_output.shape)}"
            )
        del (
            warm_output,
            warm_embeds,
            warm_positions,
            warm_segments,
            warm_local_positions,
            warm_last_indices,
        )
        return compiled, scratch_cache, {
            "cache_dir": str(cache_dir),
            "cache_was_warm": cache_was_warm,
            "new_graph_compiled": not cache_was_warm,
            "wrapper_s": wrapper_s,
            "first_call_s": first_call_s,
            "pack_length": length,
            "max_members": self.args.max_pack_members,
            "attention": "block_diagonal_causal_manual",
            "cache_boundary": "packed_scratch_cache_only",
        }

    def _prepare_packed_input(
        self,
        members: list[dict[str, Any]],
    ) -> PackedInput:
        if not members or len(members) > self.args.max_pack_members:
            raise ValueError(
                f"pack size must be 1..{self.args.max_pack_members}, "
                f"got {len(members)}"
            )
        input_parts: list[torch.Tensor] = []
        position_parts: list[torch.Tensor] = []
        segment_values: list[int] = []
        local_values: list[int] = []
        last_indices: list[int] = []
        real_tokens = 0
        for segment, item in enumerate(members):
            input_ids = torch.tensor([item["input_ids"]], dtype=torch.int64)
            attention_mask = torch.ones_like(input_ids)
            grid = torch.tensor([item["grid_thw"]], dtype=torch.int64)
            positions, _rope_deltas = self.model.get_rope_index(
                input_ids, grid, attention_mask
            )
            length = int(input_ids.shape[1])
            input_parts.append(input_ids)
            position_parts.append(positions)
            segment_values.extend([segment] * length)
            local_values.extend(range(length))
            real_tokens += length
            last_indices.append(real_tokens - 1)
        if real_tokens > self.args.pack_length:
            raise ValueError(
                f"pack has {real_tokens} tokens, limit={self.args.pack_length}"
            )

        pad_tokens = self.args.pack_length - real_tokens
        input_ids = torch.cat(input_parts, dim=1)
        if pad_tokens:
            input_ids = torch.cat(
                (
                    input_ids,
                    torch.full(
                        (1, pad_tokens),
                        int(self.model.config.pad_token_id),
                        dtype=torch.int64,
                    ),
                ),
                dim=1,
            )
        position_ids = torch.cat(position_parts, dim=2)
        if pad_tokens:
            position_ids = torch.cat(
                (
                    position_ids,
                    torch.ones((3, 1, pad_tokens), dtype=torch.int64),
                ),
                dim=2,
            )
        segment_ids = torch.tensor(
            segment_values + [-1] * pad_tokens,
            dtype=torch.int64,
        )
        local_positions = torch.tensor(
            local_values + [0] * pad_tokens,
            dtype=torch.int64,
        )
        last_token_indices = torch.zeros(
            (self.args.max_pack_members,), dtype=torch.int64
        )
        last_token_indices[: len(last_indices)] = torch.tensor(
            last_indices, dtype=torch.int64
        )

        expected_image_tokens = sum(
            int(member["projected_image_tokens"]) for member in members
        )
        image_tokens = int(
            (input_ids == int(self.model.config.image_token_id)).sum().item()
        )
        if image_tokens != expected_image_tokens:
            raise RuntimeError(
                "packed image-token accounting mismatch: "
                f"ids={image_tokens} expected={expected_image_tokens}"
            )

        input_ids = input_ids.to(self.device)
        inputs_embeds = self.model.model.embed_tokens(input_ids)
        image_mask = (
            (input_ids == self.model.config.image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
        )
        projected = self.image_embedding_basis.expand(
            image_tokens, -1
        ).contiguous()
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, projected)
        return PackedInput(
            members=members,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids.to(self.device),
            segment_ids=segment_ids.to(self.device),
            local_positions=local_positions.to(self.device),
            last_token_indices=last_token_indices.to(self.device),
            real_tokens=real_tokens,
        )

    def _packed_correctness(
        self,
        compiled: Any,
        scratch_cache: Any,
        members: list[dict[str, Any]],
    ) -> dict[str, Any]:
        packed = self._prepare_packed_input(members)
        packed_hidden = compiled(
            packed.inputs_embeds,
            packed.position_ids,
            packed.segment_ids,
            packed.local_positions,
            packed.last_token_indices,
            *scratch_cache.flat_tensors(),
        )[:, : len(members)]
        individual_hidden: list[torch.Tensor] = []
        for item in members:
            resident = self._resident(item)
            embeds = self.model.model.embed_tokens(resident.input_ids)
            image_mask = (
                (resident.input_ids == self.model.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(embeds)
            )
            embeds = embeds.masked_scatter(
                image_mask, resident.projected_image_embeds
            )
            route = self.runtime.route(int(embeds.shape[1]))
            prepared = self.runtime.prepare(
                embeds,
                resident.attention_mask,
                resident.position_ids,
                route=route,
            )
            cache = self.model.allocate_static_cache(
                batch_size=1,
                cache_length=self.args.cache_length,
                device=self.device,
                dtype=self.dtype,
                init_mode="empty",
            )
            individual_hidden.append(
                self.runtime.run_prepared(prepared, cache)[:, -1]
            )
        reference_hidden = torch.cat(individual_hidden, dim=0)
        synchronize(self.device)
        difference = (packed_hidden[0].float() - reference_hidden.float()).abs()
        packed_tokens = torch.argmax(
            self.model.lm_head(packed_hidden)[0].float(), dim=-1
        )
        reference_tokens = torch.argmax(
            self.model.lm_head(reference_hidden).float(), dim=-1
        )
        synchronize(self.device)
        return {
            "members": len(members),
            "real_tokens": packed.real_tokens,
            "source_indices": [int(item["source_index"]) for item in members],
            "hidden_max_abs": float(difference.max().item()),
            "hidden_mean_abs": float(difference.mean().item()),
            "first_token_matches": int((packed_tokens == reference_tokens).sum().item()),
            "first_token_total": len(members),
            "first_token_parity": float(
                (packed_tokens == reference_tokens).float().mean().item()
            ),
        }

    def _prepared_from_lab_pack(
        self,
        packed: PackedInput,
    ) -> PreparedPackedTextPrefill:
        lengths = tuple(int(item["input_tokens"]) for item in packed.members)
        offsets: list[int] = []
        cursor = 0
        for length in lengths:
            offsets.append(cursor)
            cursor += length
        return PreparedPackedTextPrefill(
            inputs_embeds=packed.inputs_embeds,
            position_ids=packed.position_ids,
            segment_ids=packed.segment_ids,
            local_positions=packed.local_positions,
            last_token_indices=packed.last_token_indices,
            segment_lengths=lengths,
            segment_offsets=tuple(offsets),
            real_seq_len=packed.real_tokens,
            physical_seq_len=self.args.pack_length,
        )

    def _direct_arena_slots(self, members: int) -> torch.Tensor:
        if members > self.args.arena_batch_size:
            raise ValueError(
                f"pack has {members} members but arena has "
                f"{self.args.arena_batch_size} slots"
            )
        slots = torch.zeros(
            (self.args.max_pack_members,),
            device=self.device,
            dtype=torch.int64,
        )
        slots[:members] = torch.arange(
            members,
            device=self.device,
            dtype=torch.int64,
        )
        return slots

    def _direct_arena_correctness(
        self,
        baseline: PackedTextPrefillRuntime,
        direct: DirectArenaPackedTextPrefillRuntime,
        members: list[dict[str, Any]],
    ) -> dict[str, Any]:
        packed = self._prepare_packed_input(members)
        prepared = self._prepared_from_lab_pack(packed)
        slots = self._direct_arena_slots(len(members))
        baseline_hidden = baseline.run_prepared(prepared)
        direct_hidden = direct.run_prepared(prepared, slots)
        synchronize(self.device)

        valid_baseline = baseline_hidden[:, : len(members)].float()
        valid_direct = direct_hidden[:, : len(members)].float()
        hidden_difference = (valid_baseline - valid_direct).abs()
        baseline_tokens = torch.argmax(
            self.model.lm_head(valid_baseline.to(self.dtype)).float(),
            dim=-1,
        )
        direct_tokens = torch.argmax(
            self.model.lm_head(valid_direct.to(self.dtype)).float(),
            dim=-1,
        )
        scratch = baseline.scratch_caches[prepared.physical_seq_len]
        kv_differences: list[torch.Tensor] = []
        for member_index, (offset, length) in enumerate(
            zip(prepared.segment_offsets, prepared.segment_lengths)
        ):
            for scratch_tensor, arena_tensor in zip(
                scratch.flat_tensors(),
                direct.arena_cache.flat_tensors(),
            ):
                kv_differences.append(
                    (
                        scratch_tensor[:, :, offset : offset + length, :].float()
                        - arena_tensor[
                            member_index : member_index + 1,
                            :,
                            :length,
                            :,
                        ].float()
                    )
                    .abs()
                    .max()
                )
        kv_max_abs = float(torch.stack(kv_differences).max().item())
        first_token_matches = int(
            (baseline_tokens == direct_tokens).sum().item()
        )
        synchronize(self.device)
        return {
            "members": len(members),
            "real_tokens": prepared.real_seq_len,
            "source_indices": [int(item["source_index"]) for item in members],
            "hidden_mean_abs": float(hidden_difference.mean().item()),
            "hidden_max_abs": float(hidden_difference.max().item()),
            "kv_valid_prefix_max_abs": kv_max_abs,
            "kv_valid_prefix_exact": kv_max_abs == 0.0,
            "first_token_matches": first_token_matches,
            "first_token_total": len(members),
            "first_token_parity": first_token_matches / len(members),
        }

    def _run_direct_arena_scope(
        self,
        baseline: PackedTextPrefillRuntime,
        direct: DirectArenaPackedTextPrefillRuntime,
        *,
        scope: str,
    ) -> dict[str, Any]:
        pack_member_limit = min(
            self.args.max_pack_members,
            self.args.arena_batch_size,
        )
        packs, overflow = _form_text_packs(
            self.items,
            scope=scope,
            capacity=self.args.pack_length,
            max_members=pack_member_limit,
        )
        if not packs:
            raise ValueError(f"scope {scope!r} did not form any direct packs")
        correctness_pack = max(
            packs,
            key=lambda pack: (
                len(pack),
                sum(int(item["input_tokens"]) for item in pack),
            ),
        )
        correctness = self._direct_arena_correctness(
            baseline,
            direct,
            correctness_pack,
        )

        baseline_durations: list[float] = []
        direct_durations: list[float] = []
        chunk_size = 16
        started = time.perf_counter()
        for chunk_start in range(0, len(packs), chunk_size):
            chunk = packs[chunk_start : chunk_start + chunk_size]
            timeline = DeviceTimeline(self.device)
            held: list[Any] = []
            for offset, members in enumerate(chunk):
                pack_index = chunk_start + offset
                packed = self._prepare_packed_input(members)
                prepared = self._prepared_from_lab_pack(packed)
                slots = self._direct_arena_slots(len(members))
                baseline_name = f"pack{pack_index:06d}::scratch"
                direct_name = f"pack{pack_index:06d}::direct_arena"
                calls = (
                    (
                        baseline_name,
                        lambda prepared=prepared: baseline.run_prepared(prepared),
                    ),
                    (
                        direct_name,
                        lambda prepared=prepared, slots=slots: direct.run_prepared(
                            prepared,
                            slots,
                        ),
                    ),
                )
                if pack_index % 2:
                    calls = tuple(reversed(calls))
                for name, call in calls:
                    held.append(timeline.measure(name, call))
                held.extend((packed, prepared, slots))
            spans = timeline.resolve()
            for name, seconds in spans.items():
                if name.endswith("::scratch"):
                    baseline_durations.append(float(seconds))
                else:
                    direct_durations.append(float(seconds))
            del held
            completed = min(chunk_start + chunk_size, len(packs))
            if completed == len(packs) or completed % 128 == 0:
                print(
                    f"DIRECT_ARENA_PROGRESS scope={scope} "
                    f"packs={completed}/{len(packs)} "
                    f"scratch_s={sum(baseline_durations):.3f} "
                    f"direct_s={sum(direct_durations):.3f}",
                    flush=True,
                )
        wall_s = time.perf_counter() - started
        baseline_s = sum(baseline_durations)
        direct_s = sum(direct_durations)
        real_tokens = sum(
            int(item["input_tokens"])
            for pack in packs
            for item in pack
        )
        physical_tokens = len(packs) * self.args.pack_length
        current_redistribution_s = None
        summary_path = self.args.packed_e2e_summary.expanduser().resolve()
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            current_redistribution_s = float(
                summary["recognition"]["device_stage_s"]["text_kv_redistribute"]
            )
        arena_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in direct.arena_cache.flat_tensors()
        )
        private_cache_bytes = arena_bytes // self.args.arena_batch_size
        current_combined_s = (
            None
            if current_redistribution_s is None
            else baseline_s + current_redistribution_s
        )
        return {
            "scope": scope,
            "policy": "first_fit_decreasing",
            "packs": len(packs),
            "packed_items": sum(len(pack) for pack in packs),
            "overflow_items": len(overflow),
            "pack_member_limit": pack_member_limit,
            "real_text_tokens": real_tokens,
            "physical_text_tokens": physical_tokens,
            "fill_fraction": real_tokens / physical_tokens,
            "scratch_graph_s": baseline_s,
            "direct_arena_graph_s": direct_s,
            "direct_graph_overhead_s": direct_s - baseline_s,
            "direct_vs_scratch_ratio": direct_s / baseline_s,
            "current_e2e_redistribution_s": current_redistribution_s,
            "projected_current_graph_plus_redistribution_s": current_combined_s,
            "projected_direct_time_reduction_fraction": (
                None
                if current_combined_s is None
                else 1.0 - direct_s / current_combined_s
            ),
            "scratch_effective_tok_per_s": real_tokens / baseline_s,
            "direct_effective_tok_per_s": real_tokens / direct_s,
            "scratch_physical_tok_per_s": physical_tokens / baseline_s,
            "direct_physical_tok_per_s": physical_tokens / direct_s,
            "correctness": correctness,
            "hbm": {
                "arena_bytes": arena_bytes,
                "private_cache_bytes_per_waiting_crop": private_cache_bytes,
                "current_64_crop_ready_capacity_bytes": private_cache_bytes * 64,
                "direct_arena_extra_resident_bytes": 0,
            },
            "wall_s": wall_s,
            "measurement_order": "alternating_scratch_and_direct",
            "scope_limit": (
                "lab assumes target arena rows are free; scheduler slot "
                "availability and decode cadence are not modeled"
            ),
        }

    def direct_arena(self) -> dict[str, Any]:
        buckets = (self.args.pack_length,)
        baseline = PackedTextPrefillRuntime(
            self.model,
            buckets=buckets,
            max_members=self.args.max_pack_members,
            cache_root=self.args.production_packed_cache_dir,
            destination_cache_length=self.args.cache_length,
            device=self.device,
            dtype=self.dtype,
            model_dir=self.model_dir,
            linear_weight_format=self.linear_weight_format,
        )
        direct = DirectArenaPackedTextPrefillRuntime(
            self.model,
            buckets=buckets,
            max_members=self.args.max_pack_members,
            arena_batch_size=self.args.arena_batch_size,
            arena_cache_length=self.args.cache_length,
            cache_root=self.args.direct_arena_cache_dir,
            device=self.device,
            dtype=self.dtype,
            model_dir=self.model_dir,
            linear_weight_format=self.linear_weight_format,
            allow_compile=self.args.allow_compile,
        )
        scopes = {
            scope: self._run_direct_arena_scope(
                baseline,
                direct,
                scope=scope,
            )
            for scope in self.args.pack_scope
        }
        return {
            "mode": "direct_arena",
            "baseline_runtime": baseline.metadata,
            "direct_runtime": direct.metadata,
            "scopes": scopes,
            "integration_status": (
                "lab-only direct arena write; production scheduler still "
                "owns slot assignment and private ready caches"
            ),
        }

    def _new_static_cache(
        self,
        *,
        batch_size: int,
        cache_length: int,
        init_mode: str = "empty",
    ) -> LocalPaddleOCRVLStaticCache:
        return self.model.allocate_static_cache(
            batch_size=batch_size,
            cache_length=cache_length,
            device=self.device,
            dtype=self.dtype,
            init_mode=init_mode,
        )

    def _new_lease_pool(self) -> PackedKVCachePool:
        return PackedKVCachePool(
            lambda bucket: self._new_static_cache(
                batch_size=1,
                cache_length=bucket,
            )
        )

    @staticmethod
    def _run_packed_with_cache(
        runtime: PackedTextPrefillRuntime,
        prepared: PreparedPackedTextPrefill,
        cache: LocalPaddleOCRVLStaticCache,
    ) -> torch.Tensor:
        if int(cache.cache_length) != prepared.physical_seq_len:
            raise ValueError(
                "packed cache length does not match the prepared graph: "
                f"cache={cache.cache_length} "
                f"prepared={prepared.physical_seq_len}"
            )
        return runtime.compiled[prepared.physical_seq_len](
            prepared.inputs_embeds,
            prepared.position_ids,
            prepared.segment_ids,
            prepared.local_positions,
            prepared.last_token_indices,
            *cache.flat_tensors(),
        )

    @staticmethod
    def _copy_private_prefixes_to_arena(
        private_caches: Sequence[LocalPaddleOCRVLStaticCache],
        segment_lengths: Sequence[int],
        arena: LocalPaddleOCRVLStaticCache,
    ) -> int:
        if len(private_caches) != len(segment_lengths):
            raise ValueError("private cache copies do not align with segments")
        copied_bytes = 0
        for slot, (source_cache, length) in enumerate(
            zip(private_caches, segment_lengths)
        ):
            for destination, source in zip(
                arena.flat_tensors(),
                source_cache.flat_tensors(),
            ):
                prefix = source[:, :, : int(length), :]
                destination[slot : slot + 1, :, : int(length), :].copy_(prefix)
                copied_bytes += int(prefix.numel()) * int(prefix.element_size())
        return copied_bytes

    @staticmethod
    def _copy_leased_prefixes_to_arena(
        lease: PackedKVCacheLease,
        arena: LocalPaddleOCRVLStaticCache,
    ) -> int:
        copied_bytes = 0
        for slot, member in enumerate(lease.members):
            source_cache = member.cache_view()
            for destination, source in zip(
                arena.flat_tensors(),
                source_cache.flat_tensors(),
            ):
                destination[
                    slot : slot + 1,
                    :,
                    : member.length,
                    :,
                ].copy_(source)
                copied_bytes += int(source.numel()) * int(source.element_size())
        return copied_bytes

    @staticmethod
    def _lease_valid_snapshot(
        lease: PackedKVCacheLease,
    ) -> tuple[torch.Tensor, ...]:
        snapshots: list[torch.Tensor] = []
        for member in lease.members:
            snapshots.extend(
                tensor.clone() for tensor in member.cache_view().flat_tensors()
            )
        return tuple(snapshots)

    @staticmethod
    def _snapshot_max_abs(
        lease: PackedKVCacheLease,
        snapshot: Sequence[torch.Tensor],
    ) -> float:
        current = [
            tensor
            for member in lease.members
            for tensor in member.cache_view().flat_tensors()
        ]
        if len(current) != len(snapshot):
            raise ValueError("packed KV snapshot no longer aligns")
        maxima = [
            (actual.float() - expected.float()).abs().max()
            for actual, expected in zip(current, snapshot)
        ]
        return float(torch.stack(maxima).max().item()) if maxima else 0.0

    def _lease_reuse_probe(
        self,
        runtime: PackedTextPrefillRuntime,
        pool: PackedKVCachePool,
        packs: Sequence[list[dict[str, Any]]],
    ) -> dict[str, Any]:
        if len(packs) < 3:
            raise ValueError("packed KV reuse probe requires at least three packs")
        prepared = [
            self._prepared_from_lab_pack(self._prepare_packed_input(pack))
            for pack in packs[:3]
        ]
        first = pool.acquire(
            prepared[0].physical_seq_len,
            segment_offsets=prepared[0].segment_offsets,
            segment_lengths=prepared[0].segment_lengths,
        )
        first_output = self._run_packed_with_cache(
            runtime,
            prepared[0],
            first.cache,
        )
        synchronize(self.device)
        first_snapshot = self._lease_valid_snapshot(first)

        second = pool.acquire(
            prepared[1].physical_seq_len,
            segment_offsets=prepared[1].segment_offsets,
            segment_lengths=prepared[1].segment_lengths,
        )
        second_output = self._run_packed_with_cache(
            runtime,
            prepared[1],
            second.cache,
        )
        synchronize(self.device)
        first_while_second_live_max_abs = self._snapshot_max_abs(
            first,
            first_snapshot,
        )
        second_snapshot = self._lease_valid_snapshot(second)
        first_buffer = first.buffer_id
        second_buffer = second.buffer_id
        first.release_all()

        third = pool.acquire(
            prepared[2].physical_seq_len,
            segment_offsets=prepared[2].segment_offsets,
            segment_lengths=prepared[2].segment_lengths,
        )
        third_output = self._run_packed_with_cache(
            runtime,
            prepared[2],
            third.cache,
        )
        synchronize(self.device)
        second_while_first_reused_max_abs = self._snapshot_max_abs(
            second,
            second_snapshot,
        )
        reused_first_buffer = third.buffer_id == first_buffer
        distinct_live_buffers = first_buffer != second_buffer
        third_buffer = third.buffer_id
        second.release_all()
        third.release_all()
        del (
            first_output,
            second_output,
            third_output,
            first_snapshot,
            second_snapshot,
            prepared,
        )
        return {
            "distinct_buffers_while_both_live": distinct_live_buffers,
            "released_buffer_reused": reused_first_buffer,
            "buffer_ids": {
                "first": first_buffer,
                "second": second_buffer,
                "third": third_buffer,
            },
            "first_while_second_live_max_abs": first_while_second_live_max_abs,
            "second_while_first_reused_max_abs": (
                second_while_first_reused_max_abs
            ),
            "passed": (
                distinct_live_buffers
                and reused_first_buffer
                and first_while_second_live_max_abs == 0.0
                and second_while_first_reused_max_abs == 0.0
                and pool.active_buffers == 0
            ),
        }

    def _rope_deltas(
        self,
        members: Sequence[dict[str, Any]],
    ) -> torch.Tensor:
        values: list[torch.Tensor] = []
        for item in members:
            input_ids = torch.tensor([item["input_ids"]], dtype=torch.int64)
            attention_mask = torch.ones_like(input_ids)
            grid = torch.tensor([item["grid_thw"]], dtype=torch.int64)
            _positions, rope_deltas = self.model.get_rope_index(
                input_ids,
                grid,
                attention_mask,
            )
            values.append(rope_deltas)
        return torch.cat(values, dim=0).to(self.device)

    def _cache_lease_correctness(
        self,
        runtime: PackedTextPrefillRuntime,
        pool: PackedKVCachePool,
        members: list[dict[str, Any]],
    ) -> dict[str, Any]:
        members = members[: min(2, self.args.arena_batch_size)]
        if len(members) < 2:
            raise ValueError("packed KV correctness requires two members")
        packed = self._prepare_packed_input(members)
        prepared = self._prepared_from_lab_pack(packed)

        private_caches = [
            self._new_static_cache(
                batch_size=1,
                cache_length=self.args.cache_length,
            )
            for _member in members
        ]
        current_arena = self._new_static_cache(
            batch_size=len(members),
            cache_length=self.args.cache_length,
            init_mode="zeros",
        )
        leased_arena = self._new_static_cache(
            batch_size=len(members),
            cache_length=self.args.cache_length,
            init_mode="zeros",
        )

        current_hidden = runtime.run_prepared(prepared)
        current_redistribution_bytes = runtime.redistribute_cache(
            prepared,
            private_caches,
        )
        current_admission_bytes = self._copy_private_prefixes_to_arena(
            private_caches,
            prepared.segment_lengths,
            current_arena,
        )

        lease = pool.acquire(
            prepared.physical_seq_len,
            segment_offsets=prepared.segment_offsets,
            segment_lengths=prepared.segment_lengths,
        )
        leased_hidden = self._run_packed_with_cache(
            runtime,
            prepared,
            lease.cache,
        )
        leased_admission_bytes = self._copy_leased_prefixes_to_arena(
            lease,
            leased_arena,
        )
        synchronize(self.device)

        valid_current_hidden = current_hidden[:, : len(members)].float()
        valid_leased_hidden = leased_hidden[:, : len(members)].float()
        hidden_difference = (
            valid_current_hidden - valid_leased_hidden
        ).abs()
        current_tokens = torch.argmax(
            self.model.lm_head(valid_current_hidden.to(self.dtype)).float(),
            dim=-1,
        ).reshape(-1)
        leased_tokens = torch.argmax(
            self.model.lm_head(valid_leased_hidden.to(self.dtype)).float(),
            dim=-1,
        ).reshape(-1)

        kv_maxima: list[torch.Tensor] = []
        for slot, length in enumerate(prepared.segment_lengths):
            for current, leased in zip(
                current_arena.flat_tensors(),
                leased_arena.flat_tensors(),
            ):
                kv_maxima.append(
                    (
                        current[slot : slot + 1, :, :length, :].float()
                        - leased[slot : slot + 1, :, :length, :].float()
                    )
                    .abs()
                    .max()
                )
        kv_max_abs = float(torch.stack(kv_maxima).max().item())

        decode_steps: list[dict[str, Any]] = []
        current_input = current_tokens.reshape(-1, 1)
        leased_input = leased_tokens.reshape(-1, 1)
        current_position = torch.tensor(
            prepared.segment_lengths,
            device=self.device,
            dtype=torch.int64,
        )
        leased_position = current_position.clone()
        rope_deltas = self._rope_deltas(members)
        decode_stage = TextDecodeStage(self.model).eval()
        for step in range(self.args.lease_decode_steps):
            current_logits = decode_stage(
                current_input,
                current_position,
                rope_deltas,
                *current_arena.flat_tensors(),
            )
            leased_logits = decode_stage(
                leased_input,
                leased_position,
                rope_deltas,
                *leased_arena.flat_tensors(),
            )
            current_input = torch.argmax(
                current_logits[:, -1, :].float(),
                dim=-1,
                keepdim=True,
            )
            leased_input = torch.argmax(
                leased_logits[:, -1, :].float(),
                dim=-1,
                keepdim=True,
            )
            logit_difference = (
                current_logits.float() - leased_logits.float()
            ).abs()
            synchronize(self.device)
            matches = int((current_input == leased_input).sum().item())
            decode_steps.append(
                {
                    "step": step,
                    "token_matches": matches,
                    "tokens": len(members),
                    "token_parity": matches / len(members),
                    "logit_mean_abs": float(logit_difference.mean().item()),
                    "logit_max_abs": float(logit_difference.max().item()),
                }
            )
            current_position += 1
            leased_position += 1

        prefill_token_matches = int(
            (current_tokens == leased_tokens).sum().item()
        )
        lease.release_all()
        return {
            "members": len(members),
            "source_indices": [int(item["source_index"]) for item in members],
            "segment_lengths": list(prepared.segment_lengths),
            "alternate_cache_tensor_identity_accepted": True,
            "hidden_mean_abs": float(hidden_difference.mean().item()),
            "hidden_max_abs": float(hidden_difference.max().item()),
            "prefill_first_token_matches": prefill_token_matches,
            "prefill_first_token_total": len(members),
            "prefill_first_token_parity": prefill_token_matches / len(members),
            "arena_kv_valid_prefix_max_abs": kv_max_abs,
            "arena_kv_valid_prefix_exact": kv_max_abs == 0.0,
            "current_redistribution_bytes": current_redistribution_bytes,
            "current_admission_bytes": current_admission_bytes,
            "leased_admission_bytes": leased_admission_bytes,
            "decode_steps": decode_steps,
            "decode_token_parity": (
                1.0
                if not decode_steps
                else sum(row["token_matches"] for row in decode_steps)
                / sum(row["tokens"] for row in decode_steps)
            ),
            "passed": (
                kv_max_abs == 0.0
                and prefill_token_matches == len(members)
                and all(row["token_parity"] == 1.0 for row in decode_steps)
                and current_redistribution_bytes == current_admission_bytes
                and current_admission_bytes == leased_admission_bytes
            ),
        }

    def _lease_reservoir_projection(
        self,
        packs: Sequence[list[dict[str, Any]]],
        *,
        lease_bytes: int,
        private_cache_bytes: int,
    ) -> dict[str, Any]:
        ready_capacity = (
            self.args.ready_buffer_capacity
            if self.args.ready_buffer_capacity is not None
            else self.args.arena_batch_size * 4
        )
        owner_ids = [
            pack_index
            for pack_index, pack in enumerate(packs)
            for _member in pack
        ]
        window = min(ready_capacity, len(owner_ids))
        distinct_counts = [
            len(set(owner_ids[start : start + window]))
            for start in range(0, len(owner_ids) - window + 1)
        ]
        max_live_leases = max(distinct_counts, default=0)
        mean_live_leases = (
            statistics.mean(distinct_counts) if distinct_counts else 0.0
        )
        current_bytes = window * private_cache_bytes
        leased_bytes = max_live_leases * lease_bytes
        return {
            "ready_buffer_capacity": ready_capacity,
            "modeled_ready_members": window,
            "packed_members_in_scope": len(owner_ids),
            "max_live_leases": max_live_leases,
            "mean_live_leases": mean_live_leases,
            "private_cache_bytes_per_ready_member": private_cache_bytes,
            "packed_lease_bytes_per_buffer": lease_bytes,
            "current_private_ready_bytes": current_bytes,
            "leased_ready_bytes_high_water": leased_bytes,
            "saved_ready_bytes": current_bytes - leased_bytes,
            "saved_ready_fraction": (
                0.0 if current_bytes == 0 else 1.0 - leased_bytes / current_bytes
            ),
            "method": (
                "maximum distinct packed-cache owners across every contiguous "
                "ready-reservoir window"
            ),
            "excludes_overflow_items": True,
        }

    def _run_cache_lease_scope(
        self,
        runtime: PackedTextPrefillRuntime,
        pool: PackedKVCachePool,
        *,
        scope: str,
    ) -> dict[str, Any]:
        pack_member_limit = min(
            self.args.max_pack_members,
            self.args.arena_batch_size,
        )
        packs, overflow = _form_text_packs(
            self.items,
            scope=scope,
            capacity=self.args.pack_length,
            max_members=pack_member_limit,
        )
        if not packs:
            raise ValueError(f"scope {scope!r} did not form any cache-lease packs")
        private_caches = [
            self._new_static_cache(
                batch_size=1,
                cache_length=self.args.cache_length,
            )
            for _index in range(max(len(pack) for pack in packs))
        ]
        arena = self._new_static_cache(
            batch_size=self.args.arena_batch_size,
            cache_length=self.args.cache_length,
        )
        synchronize(self.device)

        totals = Counter()
        samples: dict[str, list[float]] = defaultdict(list)
        byte_totals = Counter()
        chunk_size = 16
        started = time.perf_counter()
        for chunk_start in range(0, len(packs), chunk_size):
            chunk = packs[chunk_start : chunk_start + chunk_size]
            timeline = DeviceTimeline(self.device)
            held: list[Any] = []
            for offset, members in enumerate(chunk):
                pack_index = chunk_start + offset
                packed = self._prepare_packed_input(members)
                prepared = self._prepared_from_lab_pack(packed)
                current_destinations = private_caches[: len(members)]

                def current_path() -> None:
                    held.append(
                        timeline.measure(
                            f"pack{pack_index:06d}::current_graph",
                            lambda: runtime.run_prepared(prepared),
                        )
                    )
                    byte_totals["current_redistribution"] += int(
                        timeline.measure(
                            f"pack{pack_index:06d}::current_redistribution",
                            lambda: runtime.redistribute_cache(
                                prepared,
                                current_destinations,
                            ),
                        )
                    )
                    byte_totals["current_admission"] += int(
                        timeline.measure(
                            f"pack{pack_index:06d}::current_admission",
                            lambda: self._copy_private_prefixes_to_arena(
                                current_destinations,
                                prepared.segment_lengths,
                                arena,
                            ),
                        )
                    )

                def leased_path() -> None:
                    lease = pool.acquire(
                        prepared.physical_seq_len,
                        segment_offsets=prepared.segment_offsets,
                        segment_lengths=prepared.segment_lengths,
                    )
                    held.append(
                        timeline.measure(
                            f"pack{pack_index:06d}::leased_graph",
                            lambda: self._run_packed_with_cache(
                                runtime,
                                prepared,
                                lease.cache,
                            ),
                        )
                    )
                    byte_totals["leased_admission"] += int(
                        timeline.measure(
                            f"pack{pack_index:06d}::leased_admission",
                            lambda: self._copy_leased_prefixes_to_arena(
                                lease,
                                arena,
                            ),
                        )
                    )
                    # The admission reads are enqueued on the same stream as
                    # the next graph, so returning the buffer here is ordered.
                    lease.release_all()

                calls = (current_path, leased_path)
                if pack_index % 2:
                    calls = tuple(reversed(calls))
                for call in calls:
                    call()
                held.extend((packed, prepared))

            spans = timeline.resolve()
            for name, seconds in spans.items():
                stage = name.rsplit("::", 1)[-1]
                totals[stage] += float(seconds)
                samples[stage].append(float(seconds))
            del held
            completed = min(chunk_start + chunk_size, len(packs))
            if completed == len(packs) or completed % 128 == 0:
                current_s = (
                    totals["current_graph"]
                    + totals["current_redistribution"]
                    + totals["current_admission"]
                )
                leased_s = totals["leased_graph"] + totals["leased_admission"]
                print(
                    f"CACHE_LEASE_PROGRESS scope={scope} "
                    f"packs={completed}/{len(packs)} "
                    f"current_s={current_s:.3f} leased_s={leased_s:.3f}",
                    flush=True,
                )
        wall_s = time.perf_counter() - started
        if pool.active_buffers:
            raise RuntimeError("packed KV timing left active leases")

        current_s = (
            totals["current_graph"]
            + totals["current_redistribution"]
            + totals["current_admission"]
        )
        leased_s = totals["leased_graph"] + totals["leased_admission"]
        lease_bytes = static_cache_nbytes(
            runtime.scratch_caches[self.args.pack_length]
        )
        private_cache_bytes = static_cache_nbytes(private_caches[0])
        return {
            "scope": scope,
            "policy": "first_fit_decreasing",
            "packs": len(packs),
            "packed_items": sum(len(pack) for pack in packs),
            "overflow_items": len(overflow),
            "pack_member_limit": pack_member_limit,
            "member_histogram": dict(
                sorted(Counter(len(pack) for pack in packs).items())
            ),
            "real_text_tokens": sum(
                int(item["input_tokens"])
                for pack in packs
                for item in pack
            ),
            "physical_text_tokens": len(packs) * self.args.pack_length,
            "device_s": {
                name: float(seconds) for name, seconds in sorted(totals.items())
            },
            "current_two_copy_s": current_s,
            "leased_one_copy_s": leased_s,
            "time_saved_s": current_s - leased_s,
            "speedup": current_s / leased_s,
            "copy_only_s": {
                "current": (
                    totals["current_redistribution"]
                    + totals["current_admission"]
                ),
                "leased": totals["leased_admission"],
            },
            "copied_bytes": {
                name: int(value) for name, value in sorted(byte_totals.items())
            },
            "latency_ms": {
                name: {
                    "mean": statistics.mean(values) * 1000.0,
                    "median": statistics.median(values) * 1000.0,
                    "p95": _percentile(values, 0.95) * 1000.0,
                }
                for name, values in sorted(samples.items())
            },
            "hbm": self._lease_reservoir_projection(
                packs,
                lease_bytes=lease_bytes,
                private_cache_bytes=private_cache_bytes,
            ),
            "wall_s": wall_s,
            "measurement_order": "alternating_current_and_leased_per_pack",
            "timing_boundary": (
                "packed transformer KV write plus KV-only redistribution and "
                "decode-arena admission copies; output head is excluded"
            ),
        }

    def cache_lease(self) -> dict[str, Any]:
        runtime = PackedTextPrefillRuntime(
            self.model,
            buckets=(self.args.pack_length,),
            max_members=self.args.max_pack_members,
            cache_root=self.args.production_packed_cache_dir,
            destination_cache_length=self.args.cache_length,
            device=self.device,
            dtype=self.dtype,
            model_dir=self.model_dir,
            linear_weight_format=self.linear_weight_format,
        )
        global_packs, _overflow = _form_text_packs(
            self.items,
            scope="global",
            capacity=self.args.pack_length,
            max_members=min(
                self.args.max_pack_members,
                self.args.arena_batch_size,
            ),
        )
        if len(global_packs) < 3:
            raise ValueError("cache-lease lab requires at least three packs")
        pool = self._new_lease_pool()
        reuse_probe = self._lease_reuse_probe(runtime, pool, global_packs)
        if not reuse_probe["passed"]:
            raise RuntimeError(f"packed KV reuse probe failed: {reuse_probe}")

        scopes = {
            scope: self._run_cache_lease_scope(
                runtime,
                pool,
                scope=scope,
            )
            for scope in self.args.pack_scope
        }
        correctness_pack = max(
            (pack for pack in global_packs if len(pack) > 1),
            key=lambda pack: (
                len(pack),
                sum(int(item["input_tokens"]) for item in pack),
            ),
        )
        correctness = self._cache_lease_correctness(
            runtime,
            pool,
            correctness_pack,
        )
        if pool.active_buffers:
            raise RuntimeError("cache-lease lab finished with active buffers")
        return {
            "mode": "cache_lease",
            "runtime": runtime.metadata,
            "reuse_probe": reuse_probe,
            "correctness": correctness,
            "pool": pool.stats(),
            "scopes": scopes,
            "integration_status": (
                "lab-only: packed caches are leased and admitted by slice; "
                "the production ready-request contract is unchanged"
            ),
        }

    def _load_baseline_records(self) -> tuple[dict[int, dict[str, Any]], float]:
        path = self.args.baseline_lab_result.expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload["result"]
        records = {
            int(record["source_index"]): record
            for record in result["records"]
        }
        selected_ids = {int(item["source_index"]) for item in self.items}
        missing = sorted(selected_ids - set(records))
        if missing:
            raise ValueError(
                f"baseline lab result is missing {len(missing)} selected items"
            )
        total_s = sum(
            float(records[index]["device_stage_s"]["text_prefill"])
            for index in selected_ids
        )
        return records, total_s

    def _run_packed_scope(
        self,
        compiled: Any,
        scratch_cache: Any,
        baseline_records: dict[int, dict[str, Any]],
        baseline_total_s: float,
        *,
        scope: str,
    ) -> dict[str, Any]:
        packs, overflow = _form_text_packs(
            self.items,
            scope=scope,
            capacity=self.args.pack_length,
            max_members=self.args.max_pack_members,
        )
        durations: list[float] = []
        chunk_size = 32
        started = time.perf_counter()
        for chunk_start in range(0, len(packs), chunk_size):
            chunk = packs[chunk_start : chunk_start + chunk_size]
            timeline = DeviceTimeline(self.device)
            outputs = []
            for offset, members in enumerate(chunk):
                packed = self._prepare_packed_input(members)
                name = f"pack{chunk_start + offset:06d}::text_prefill"
                output = timeline.measure(
                    name,
                    lambda packed=packed: compiled(
                        packed.inputs_embeds,
                        packed.position_ids,
                        packed.segment_ids,
                        packed.local_positions,
                        packed.last_token_indices,
                        *scratch_cache.flat_tensors(),
                    ),
                )
                outputs.append((packed, output))
            spans = timeline.resolve()
            durations.extend(float(spans[name]) for name in spans)
            del outputs
            completed = min(chunk_start + chunk_size, len(packs))
            if completed == len(packs) or completed % 128 == 0:
                print(
                    f"PACKED_PROGRESS scope={scope} "
                    f"packs={completed}/{len(packs)} "
                    f"device_s={sum(durations):.3f}",
                    flush=True,
                )
        wall_s = time.perf_counter() - started
        overflow_ids = [int(item["source_index"]) for item in overflow]
        overflow_s = sum(
            float(baseline_records[index]["device_stage_s"]["text_prefill"])
            for index in overflow_ids
        )
        packed_s = sum(durations)
        projected_s = packed_s + overflow_s
        real_tokens = sum(int(item["input_tokens"]) for item in self.items)
        packed_real_tokens = sum(
            int(item["input_tokens"]) for pack in packs for item in pack
        )
        overflow_physical_tokens = sum(
            int(item["route"]["physical_text_tokens"]) for item in overflow
        )
        physical_tokens = len(packs) * self.args.pack_length + overflow_physical_tokens
        calls = len(packs) + len(overflow)
        return {
            "scope": scope,
            "policy": "first_fit_decreasing",
            "packs": len(packs),
            "overflow_items": len(overflow),
            "total_transformer_calls": calls,
            "baseline_transformer_calls": len(self.items),
            "call_reduction_fraction": 1.0 - calls / len(self.items),
            "member_histogram": dict(
                sorted(Counter(len(pack) for pack in packs).items())
            ),
            "packed_real_tokens": packed_real_tokens,
            "packed_physical_tokens": len(packs) * self.args.pack_length,
            "pack_fill_fraction": packed_real_tokens
            / (len(packs) * self.args.pack_length),
            "packed_graph_s": packed_s,
            "overflow_baseline_s": overflow_s,
            "projected_transformer_s": projected_s,
            "baseline_transformer_s": baseline_total_s,
            "transformer_speedup": baseline_total_s / projected_s,
            "transformer_time_reduction_fraction": 1.0
            - projected_s / baseline_total_s,
            "effective_tok_per_s": real_tokens / projected_s,
            "physical_tok_per_s": physical_tokens / projected_s,
            "packed_graph_latency_ms": {
                "mean": statistics.mean(durations) * 1000.0,
                "median": statistics.median(durations) * 1000.0,
                "p95": _percentile(durations, 0.95) * 1000.0,
            },
            "wall_s": wall_s,
            "excludes": [
                "packed scratch KV redistribution into independent decode caches",
                "LM head and first-token argmax",
            ],
        }

    def packed(self) -> dict[str, Any]:
        baseline_records, baseline_total_s = self._load_baseline_records()
        compiled, scratch_cache, graph_setup = self._setup_packed_graph()
        global_packs, _overflow = _form_text_packs(
            self.items,
            scope="global",
            capacity=self.args.pack_length,
            max_members=self.args.max_pack_members,
        )
        candidates = [pack for pack in global_packs if len(pack) > 1]
        if not candidates:
            raise ValueError("corpus does not contain a multi-request text pack")
        correctness_pack = max(
            candidates,
            key=lambda pack: sum(int(item["input_tokens"]) for item in pack),
        )
        correctness = self._packed_correctness(
            compiled,
            scratch_cache,
            correctness_pack,
        )
        scopes = {
            scope: self._run_packed_scope(
                compiled,
                scratch_cache,
                baseline_records,
                baseline_total_s,
                scope=scope,
            )
            for scope in self.args.pack_scope
        }
        return {
            "mode": "packed",
            "graph_setup": graph_setup,
            "correctness": correctness,
            "scopes": scopes,
            "integration_status": (
                "transformer experiment only; packed KV redistribution must be "
                "implemented and measured before production integration"
            ),
        }


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
            "packed_cache_dir": str(args.packed_cache_dir.expanduser().resolve()),
            "production_packed_cache_dir": str(
                args.production_packed_cache_dir.expanduser().resolve()
            ),
            "direct_arena_cache_dir": str(
                args.direct_arena_cache_dir.expanduser().resolve()
            ),
            "packed_e2e_summary": str(
                args.packed_e2e_summary.expanduser().resolve()
            ),
            "baseline_lab_result": str(
                args.baseline_lab_result.expanduser().resolve()
            ),
            "pack_length": args.pack_length,
            "max_pack_members": args.max_pack_members,
            "arena_batch_size": args.arena_batch_size,
            "ready_buffer_capacity": (
                args.ready_buffer_capacity
                if args.ready_buffer_capacity is not None
                else args.arena_batch_size * 4
            ),
            "lease_decode_steps": args.lease_decode_steps,
            "pack_scope": list(args.pack_scope),
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


def _print_packed(result: dict[str, Any]) -> None:
    correctness = result["correctness"]
    print(
        "PACKED_CORRECTNESS "
        f"members={correctness['members']} "
        f"real_tokens={correctness['real_tokens']} "
        f"hidden_mean_abs={correctness['hidden_mean_abs']:.6f} "
        f"hidden_max_abs={correctness['hidden_max_abs']:.6f} "
        f"first_token_parity={correctness['first_token_parity']:.4f}"
    )
    for scope, row in result["scopes"].items():
        print(
            "PACKED_RESULT "
            f"scope={scope} packs={row['packs']} "
            f"overflow={row['overflow_items']} "
            f"fill={row['pack_fill_fraction']:.4f} "
            f"transformer_s={row['projected_transformer_s']:.6f} "
            f"speedup={row['transformer_speedup']:.3f}x "
            f"effective_tok_s={row['effective_tok_per_s']:.1f} "
            f"physical_tok_s={row['physical_tok_per_s']:.1f}"
        )


def _print_direct_arena(result: dict[str, Any]) -> None:
    for scope, row in result["scopes"].items():
        correctness = row["correctness"]
        print(
            "DIRECT_ARENA_RESULT "
            f"scope={scope} packs={row['packs']} "
            f"items={row['packed_items']} "
            f"scratch_s={row['scratch_graph_s']:.6f} "
            f"direct_s={row['direct_arena_graph_s']:.6f} "
            f"ratio={row['direct_vs_scratch_ratio']:.4f} "
            f"kv_exact={correctness['kv_valid_prefix_exact']} "
            f"first_token_parity={correctness['first_token_parity']:.4f}"
        )


def _print_cache_lease(result: dict[str, Any]) -> None:
    correctness = result["correctness"]
    reuse = result["reuse_probe"]
    print(
        "CACHE_LEASE_CORRECTNESS "
        f"kv_exact={correctness['arena_kv_valid_prefix_exact']} "
        f"prefill_token_parity={correctness['prefill_first_token_parity']:.4f} "
        f"decode_token_parity={correctness['decode_token_parity']:.4f} "
        f"reuse_passed={reuse['passed']}"
    )
    for scope, row in result["scopes"].items():
        hbm = row["hbm"]
        print(
            "CACHE_LEASE_RESULT "
            f"scope={scope} packs={row['packs']} "
            f"items={row['packed_items']} "
            f"current_s={row['current_two_copy_s']:.6f} "
            f"leased_s={row['leased_one_copy_s']:.6f} "
            f"speedup={row['speedup']:.3f}x "
            f"saved_s={row['time_saved_s']:.6f} "
            f"ready_hbm_saved_gib={hbm['saved_ready_bytes'] / 2**30:.3f}"
        )


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    corpus, items = _load_corpus(args.corpus, args.max_items)
    lab = TextPrefillLab(args, items, dict(corpus["distribution"]))
    if args.mode == "replay":
        result = lab.replay()
    elif args.mode == "profile":
        result = lab.profile()
    elif args.mode == "direct_arena":
        result = lab.direct_arena()
    elif args.mode == "cache_lease":
        result = lab.cache_lease()
    else:
        result = lab.packed()
    output = _write_report(args, corpus, lab, result)
    if args.mode == "replay":
        _print_replay(result)
    elif args.mode == "packed":
        _print_packed(result)
    elif args.mode == "direct_arena":
        _print_direct_arena(result)
    elif args.mode == "cache_lease":
        _print_cache_lease(result)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
