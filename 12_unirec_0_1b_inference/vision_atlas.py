"""Compiled HxW-invariant guarded-atlas execution for UniRec FocalSVTR."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn

from modeling_optimized_unirec import (
    LOCAL_UNIREC_STATIC_CACHE_LEN,
    LocalUniRecStaticCache,
    OptimizedUniRecRunner,
    UniRecPrefilledItem,
    import_torchair_cache_compile,
    synchronize_device,
)
from prefill_timing import PrefillDeviceTimeline


ATLAS_STAGE = 2
ATLAS_HEIGHT = 32
ATLAS_WIDTH = 128
ATLAS_GUARD = 3
ATLAS_MAX_MEMBERS = 16
ATLAS_CHANNELS = 384
ATLAS_STAGE_FACTOR = 16
PREFIX_SCALE = 4
PREFIX_HEIGHT = 64
PREFIX_WIDTH = 512
PREFIX_CHANNELS = 96
STAGE_HEIGHTS = (PREFIX_HEIGHT, 32, ATLAS_HEIGHT)
STAGE_WIDTHS = (PREFIX_WIDTH, 256, ATLAS_WIDTH)
STAGE_GUARDS = (4, 4, ATLAS_GUARD)
STAGE_SCALES = (4, 2, 1)


@dataclass(frozen=True)
class CropShape:
    source_index: int
    height: int
    width: int

    @property
    def tokens(self) -> int:
        return self.height * self.width


@dataclass(frozen=True)
class Placement:
    crop: CropShape
    member: int
    y: int
    x: int
    guard: int

    @property
    def inner_y(self) -> int:
        return self.y + self.guard

    @property
    def inner_x(self) -> int:
        return self.x + self.guard


@dataclass
class Shelf:
    y: int
    height: int
    next_x: int = 0


@dataclass
class AtlasPack:
    height: int
    width: int
    guard: int
    scale: int
    shelves: list[Shelf]
    placements: list[Placement]

    @classmethod
    def empty(
        cls, *, height: int, width: int, guard: int, scale: int
    ) -> "AtlasPack":
        return cls(
            height=height,
            width=width,
            guard=guard,
            scale=scale,
            shelves=[],
            placements=[],
        )

    def try_add(self, crop: CropShape) -> bool:
        if len(self.placements) >= ATLAS_MAX_MEMBERS:
            return False
        crop_height = crop.height * self.scale
        crop_width = crop.width * self.scale
        physical_h = crop_height + 2 * self.guard
        physical_w = crop_width + 2 * self.guard
        if physical_h > self.height or physical_w > self.width:
            return False
        selected = None
        for shelf in self.shelves:
            if physical_h <= shelf.height and shelf.next_x + physical_w <= self.width:
                selected = shelf
                break
        if selected is None:
            next_y = sum(shelf.height for shelf in self.shelves)
            if next_y + physical_h > self.height:
                return False
            selected = Shelf(y=next_y, height=physical_h)
            self.shelves.append(selected)
        self.placements.append(
            Placement(
                crop=crop,
                member=len(self.placements),
                y=selected.y,
                x=selected.next_x,
                guard=self.guard,
            )
        )
        selected.next_x += physical_w
        return True

    @property
    def real_tokens(self) -> int:
        return sum(
            placement.crop.tokens * self.scale * self.scale
            for placement in self.placements
        )


@dataclass(frozen=True)
class MultiStagePack:
    crops: tuple[CropShape, ...]
    stages: tuple[AtlasPack, AtlasPack, AtlasPack]


class GuardedAtlasStage(nn.Module):
    """Run one FocalSVTR stage on independent masked atlas regions."""

    def __init__(self, stage: nn.Module):
        super().__init__()
        self.blocks = stage.blocks

    @staticmethod
    def _global_context(
        ctx: torch.Tensor,
        membership: torch.Tensor,
        normalized_membership: torch.Tensor,
    ) -> torch.Tensor:
        _, channels, height, width = ctx.shape
        flat = ctx.reshape(channels, height * width)
        means = torch.mm(flat, normalized_membership.transpose(0, 1).contiguous())
        return torch.mm(means, membership).reshape(1, channels, height, width)

    def forward(
        self,
        atlas: torch.Tensor,
        valid_mask: torch.Tensor,
        membership: torch.Tensor,
        normalized_membership: torch.Tensor,
    ) -> torch.Tensor:
        x = atlas * valid_mask
        valid_nhwc = valid_mask.permute(0, 2, 3, 1)
        for block in self.blocks:
            shortcut = x
            normalized = block.norm1(x.permute(0, 2, 3, 1))
            modulation = block.modulation
            projected = modulation.f(normalized).permute(0, 3, 1, 2).contiguous()
            channels = normalized.shape[-1]
            q, ctx, gates = torch.split(
                projected,
                (channels, channels, modulation.focal_level + 1),
                dim=1,
            )
            q = q * valid_mask
            ctx = ctx * valid_mask
            gates = gates * valid_mask
            ctx_all = None
            for level, focal_layer in enumerate(modulation.focal_layers):
                ctx = focal_layer(ctx) * valid_mask
                contribution = ctx * gates[:, level : level + 1]
                ctx_all = contribution if ctx_all is None else ctx_all + contribution
            if ctx_all is None:
                raise AssertionError("FocalSVTR stage has no focal layers")
            global_context = self._global_context(
                ctx,
                membership,
                normalized_membership,
            )
            global_context = modulation.act(global_context) * valid_mask
            ctx_all = ctx_all + global_context * gates[:, modulation.focal_level :]
            modulator = modulation.h(ctx_all) * valid_mask
            modulated = q * modulator
            modulated = modulated.permute(0, 2, 3, 1).contiguous()
            modulated = modulation.proj(modulated) * valid_nhwc
            x_nhwc = shortcut.permute(0, 2, 3, 1) + modulated
            x = (
                (x_nhwc + block.mlp(block.norm2(x_nhwc))) * valid_nhwc
            ).permute(0, 3, 1, 2).contiguous()
        return x


class RoutedGuardedAtlasPrefix(nn.Module):
    """Run stages 0-2 from one fixed flat stage-0 token reservoir."""

    def __init__(self, stages: nn.ModuleList):
        super().__init__()
        self.stage0 = GuardedAtlasStage(stages[0])
        self.downsample0 = stages[0].downsample
        self.stage1 = GuardedAtlasStage(stages[1])
        self.downsample1 = stages[1].downsample
        self.stage2 = GuardedAtlasStage(stages[2])
        if self.downsample0 is None or self.downsample1 is None:
            raise ValueError("UniRec stages 0 and 1 must have downsamplers")

    def forward(
        self,
        packed_source: torch.Tensor,
        atlas0_to_source: torch.Tensor,
        stage0_to_source1: torch.Tensor,
        atlas1_to_source: torch.Tensor,
        stage1_to_source2: torch.Tensor,
        atlas2_to_source: torch.Tensor,
        source_to_atlas: torch.Tensor,
        valid_mask0: torch.Tensor,
        membership0: torch.Tensor,
        normalized_membership0: torch.Tensor,
        valid_mask1: torch.Tensor,
        membership1: torch.Tensor,
        normalized_membership1: torch.Tensor,
        valid_mask2: torch.Tensor,
        membership2: torch.Tensor,
        normalized_membership2: torch.Tensor,
    ) -> torch.Tensor:
        atlas = packed_source.index_select(1, atlas0_to_source).reshape(
            1, PREFIX_HEIGHT, PREFIX_WIDTH, PREFIX_CHANNELS
        ).permute(0, 3, 1, 2).contiguous()
        atlas = self.stage0(
            atlas,
            valid_mask0,
            membership0,
            normalized_membership0,
        )
        tokens, height, width = self.downsample0(atlas)
        source1 = tokens.index_select(1, stage0_to_source1)
        atlas = source1.index_select(1, atlas1_to_source).reshape(
            1, STAGE_HEIGHTS[1], STAGE_WIDTHS[1], -1
        ).permute(0, 3, 1, 2).contiguous()
        atlas = self.stage1(
            atlas,
            valid_mask1,
            membership1,
            normalized_membership1,
        )
        tokens, height, width = self.downsample1(atlas)
        source2 = tokens.index_select(1, stage1_to_source2)
        source2 = torch.cat((source2, torch.zeros_like(source2)), dim=1)
        atlas = source2.index_select(1, atlas2_to_source).reshape(
            1, STAGE_HEIGHTS[2], STAGE_WIDTHS[2], -1
        ).permute(0, 3, 1, 2).contiguous()
        output = self.stage2(
            atlas,
            valid_mask2,
            membership2,
            normalized_membership2,
        )
        output = output.permute(0, 2, 3, 1).reshape(
            1, ATLAS_HEIGHT * ATLAS_WIDTH, ATLAS_CHANNELS
        )
        return output.index_select(1, source_to_atlas)


def _source_hash() -> str:
    payload = Path(__file__).read_bytes()
    payload += Path(__file__).with_name("modeling_optimized_unirec.py").read_bytes()
    return hashlib.sha256(payload).hexdigest()[:12]


def _build_multi_stage_pack(crops: list[CropShape]) -> MultiStagePack | None:
    stage_packs = []
    for height, width, guard, scale in zip(
        STAGE_HEIGHTS,
        STAGE_WIDTHS,
        STAGE_GUARDS,
        STAGE_SCALES,
    ):
        pack = AtlasPack.empty(
            height=height,
            width=width,
            guard=guard,
            scale=scale,
        )
        for crop in crops:
            if not pack.try_add(crop):
                return None
        stage_packs.append(pack)
    return MultiStagePack(tuple(crops), tuple(stage_packs))


def _pack_shapes(
    shapes: list[CropShape],
) -> tuple[list[MultiStagePack], list[CropShape]]:
    ordered = sorted(
        shapes,
        key=lambda crop: (
            (crop.height + 2 * ATLAS_GUARD) * (crop.width + 2 * ATLAS_GUARD),
            crop.height,
            crop.width,
        ),
        reverse=True,
    )
    packs: list[MultiStagePack] = []
    overflow: list[CropShape] = []
    for crop in ordered:
        if _build_multi_stage_pack([crop]) is None:
            overflow.append(crop)
            continue
        for index, pack in enumerate(packs):
            candidate = _build_multi_stage_pack([*pack.crops, crop])
            if candidate is not None:
                packs[index] = candidate
                break
        else:
            pack = _build_multi_stage_pack([crop])
            if pack is None:
                raise AssertionError("fresh multi-stage atlas rejected a fitting crop")
            packs.append(pack)
    return packs, overflow


def _routing_maps(
    pack: AtlasPack,
    *,
    dtype: torch.dtype,
    device: torch.device,
    output_divisor: int = 1,
) -> tuple[torch.Tensor, ...]:
    if pack.scale % output_divisor:
        raise ValueError("atlas scale must be divisible by output divisor")
    height = pack.height // output_divisor
    width = pack.width // output_divisor
    cells = height * width
    atlas_to_source = [-1] * cells
    source_cursor = 0
    valid = torch.zeros((1, 1, height, width), dtype=dtype)
    membership = torch.zeros((ATLAS_MAX_MEMBERS, cells), dtype=dtype)
    for placement in pack.placements:
        crop = placement.crop
        crop_height = crop.height * pack.scale // output_divisor
        crop_width = crop.width * pack.scale // output_divisor
        if (
            placement.inner_y % output_divisor
            or placement.inner_x % output_divisor
        ):
            raise ValueError("atlas placement is not downsample-aligned")
        inner_y = placement.inner_y // output_divisor
        inner_x = placement.inner_x // output_divisor
        for row in range(crop_height):
            atlas_row = inner_y + row
            for column in range(crop_width):
                atlas_index = atlas_row * width + inner_x + column
                atlas_to_source[atlas_index] = source_cursor + row * crop_width + column
        valid[
            :,
            :,
            inner_y : inner_y + crop_height,
            inner_x : inner_x + crop_width,
        ] = 1
        member_map = membership[placement.member].view(height, width)
        member_map[
            inner_y : inner_y + crop_height,
            inner_x : inner_x + crop_width,
        ] = 1
        source_cursor += crop_height * crop_width
    padding_indices = iter(range(source_cursor, cells))
    for atlas_index, source_index in enumerate(atlas_to_source):
        if source_index < 0:
            atlas_to_source[atlas_index] = next(padding_indices)
    source_to_atlas = [-1] * cells
    for atlas_index, source_index in enumerate(atlas_to_source):
        source_to_atlas[source_index] = atlas_index
    if any(index < 0 for index in source_to_atlas):
        raise AssertionError("atlas routing is not a complete permutation")
    normalized = membership / membership.sum(dim=1, keepdim=True).clamp_min(1)
    return (
        torch.tensor(atlas_to_source, dtype=torch.long, device=device),
        torch.tensor(source_to_atlas, dtype=torch.long, device=device),
        valid.to(device),
        membership.to(device),
        normalized.to(device),
    )


def _routing_inputs(
    pack: MultiStagePack,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    stage0, stage1, stage2 = pack.stages
    route0 = _routing_maps(stage0, dtype=dtype, device=device)
    bridge0 = _routing_maps(
        stage0,
        dtype=dtype,
        device=device,
        output_divisor=2,
    )
    route1 = _routing_maps(stage1, dtype=dtype, device=device)
    bridge1 = _routing_maps(
        stage1,
        dtype=dtype,
        device=device,
        output_divisor=2,
    )
    route2 = _routing_maps(stage2, dtype=dtype, device=device)
    return (
        torch.zeros(
            (1, PREFIX_HEIGHT * PREFIX_WIDTH, PREFIX_CHANNELS),
            dtype=dtype,
            device=device,
        ),
        route0[0],
        bridge0[1],
        route1[0],
        bridge1[1],
        route2[0],
        route2[1],
        route0[2],
        route0[3],
        route0[4],
        route1[2],
        route1[3],
        route1[4],
        route2[2],
        route2[3],
        route2[4],
    )


class UniRecVisionAtlasRuntime:
    """Run eager patch stems and compiled HxW-invariant stages 0-2."""

    def __init__(self, runner: OptimizedUniRecRunner) -> None:
        if not runner.device.startswith("npu"):
            raise ValueError("compiled UniRec vision atlas requires an NPU")
        if runner.compile_cache_dir is None:
            raise ValueError("compiled UniRec vision atlas requires compile_cache_dir")
        self.runner = runner
        self.stages = runner.model.encoder.vision_encoder.layers
        self.stage = self.stages[ATLAS_STAGE]
        self.module = RoutedGuardedAtlasPrefix(self.stages).eval()
        self.cache_dir = runner.compile_cache_dir / (
            f"vision_atlas_stages0_2_{ATLAS_HEIGHT}x{ATLAS_WIDTH}_"
            f"m{ATLAS_MAX_MEMBERS}_{runner.dtype_name}_src{_source_hash()}"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

        config = CompilerConfig()
        config.mode.value = "max-autotune"
        cache_compile, import_path = import_torchair_cache_compile()
        self.compiled = cache_compile(
            self.module.forward,
            config=config,
            dynamic=False,
            cache_dir=str(self.cache_dir),
            ge_cache=True,
            fullgraph=True,
        )
        self.compile_api = import_path
        self.first_call = True
        self.stats = {
            "groups": 0,
            "packs": 0,
            "packed_members": 0,
            "overflow_members": 0,
            "real_stage_tokens": 0,
            "physical_stage_tokens": 0,
            "first_call_wall_s": None,
        }

    def warmup_inputs(self) -> tuple[torch.Tensor, ...]:
        """Build one fixed-shape input set for graph load and replay."""
        pack = _build_multi_stage_pack(
            [CropShape(source_index=0, height=4, width=60)]
        )
        if pack is None:
            raise AssertionError("vision atlas warmup crop must fit all stages")
        return _routing_inputs(
            pack,
            dtype=self.runner.dtype,
            device=torch.device(self.runner.device),
        )

    def _run_patch_stem(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        vision = self.runner.model.encoder.vision_encoder
        x = vision.patch_embed(pixel_values)
        height, width = x.shape[2:]
        x = vision.pos_drop(x.flatten(2).transpose(1, 2))
        return x, height, width

    def _run_stages_0_2_eager(
        self, x: torch.Tensor, height: int, width: int
    ) -> tuple[torch.Tensor, int, int]:
        for layer in self.stages[:ATLAS_STAGE]:
            x, height, width = layer(x, height, width)
        return x, height, width

    def _run_suffix(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        vision = self.runner.model.encoder.vision_encoder
        if self.stage.downsample is None:
            raise AssertionError("UniRec stage 2 must have a downsample")
        x = x.transpose(1, 2).reshape(1, -1, height, width)
        x, height, width = self.stage.downsample(x)
        for layer in vision.layers[ATLAS_STAGE + 1 :]:
            x, height, width = layer(x, height, width)
        return self.runner.model.encoder.vision_fc(x)

    def _run_stage2_eager(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        for block in self.stage.blocks:
            block.H, block.W = height, width
            x = block(x)
        return x

    def _run_atlas_packs(
        self,
        packs: list[MultiStagePack],
        stem_states: dict[int, torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        outputs: dict[int, torch.Tensor] = {}
        for pack in packs:
            values = _routing_inputs(
                pack,
                dtype=self.runner.dtype,
                device=torch.device(self.runner.device),
            )
            packed_source = values[0]
            cursor = 0
            for crop in pack.crops:
                source = stem_states[crop.source_index]
                source_tokens = source.shape[1]
                packed_source[:, cursor : cursor + source_tokens].copy_(source)
                cursor += source_tokens
            if self.first_call:
                print(
                    "UNIREC_VISION_ATLAS_FIRST_CALL_BEGIN "
                    f"cache_dir={self.cache_dir}",
                    flush=True,
                )
                started = time.perf_counter()
            result = self.compiled(*values)
            if self.first_call:
                elapsed = time.perf_counter() - started
                self.stats["first_call_wall_s"] = elapsed
                print(
                    f"UNIREC_VISION_ATLAS_FIRST_CALL_RETURN wall_s={elapsed:.3f}",
                    flush=True,
                )
                self.first_call = False
            cursor = 0
            for crop in pack.crops:
                end = cursor + crop.tokens
                outputs[crop.source_index] = result[:, cursor:end]
                cursor = end
        return outputs

    def prefill_images_packed_for_cohort(
        self,
        images: list[tuple[Image.Image, str]],
        *,
        profile_device_stages: bool = False,
    ) -> list[UniRecPrefilledItem]:
        if not images:
            raise ValueError("cannot prefill an empty UniRec vision atlas group")
        prepared = [
            self.runner.prepare_pil_image(image, image_source=image_source)
            for image, image_source in images
        ]
        text_runtime = self.runner._get_compiled_packed_text_prefill_runtime()
        cross_cache_len = self.runner._get_static_cross_cache_len()
        timeline = (
            PrefillDeviceTimeline(torch.device(self.runner.device))
            if profile_device_stages
            else None
        )
        measure = timeline.measure if timeline is not None else lambda _name, fn: fn()

        synchronize_device(self.runner.device)
        started = time.perf_counter()
        stem_states: dict[int, torch.Tensor] = {}
        stem_shapes: dict[int, tuple[int, int]] = {}
        shapes: list[CropShape] = []
        with torch.inference_mode():
            for source_index, (inputs, _prep) in enumerate(prepared):
                x, height, width = measure(
                    "vision_crop_patch_stem",
                    lambda inputs=inputs: self._run_patch_stem(
                        inputs["pixel_values"]
                    ),
                )
                if height % PREFIX_SCALE or width % PREFIX_SCALE:
                    raise ValueError(
                        "UniRec patch-stem output must be divisible by 4, got "
                        f"{height}x{width}"
                    )
                stem_states[source_index] = x
                stem_shapes[source_index] = (height, width)
                shapes.append(
                    CropShape(
                        source_index,
                        height // PREFIX_SCALE,
                        width // PREFIX_SCALE,
                    )
                )

            packs, overflow = _pack_shapes(shapes)
            atlas_outputs = measure(
                "compiled_vision_atlas_stages_0_2",
                lambda: self._run_atlas_packs(packs, stem_states),
            )
            for crop in overflow:
                stem_height, stem_width = stem_shapes[crop.source_index]
                prefix, prefix_height, prefix_width = measure(
                    "vision_stages_0_1_eager_overflow",
                    lambda crop=crop, stem_height=stem_height, stem_width=stem_width: (
                        self._run_stages_0_2_eager(
                            stem_states[crop.source_index],
                            stem_height,
                            stem_width,
                        )
                    ),
                )
                atlas_outputs[crop.source_index] = measure(
                    "vision_stage2_eager_overflow",
                    lambda prefix=prefix, prefix_height=prefix_height, prefix_width=prefix_width: self._run_stage2_eager(
                        prefix, prefix_height, prefix_width
                    ),
                )

            encoder_hidden_states: list[torch.Tensor] = []
            encoder_attention_masks: list[torch.Tensor] = []
            for crop in sorted(shapes, key=lambda item: item.source_index):
                hidden = measure(
                    "vision_crop_suffix_stage3_projection",
                    lambda crop=crop: self._run_suffix(
                        atlas_outputs[crop.source_index], crop.height, crop.width
                    ),
                )
                encoder_hidden_states.append(hidden)
                encoder_attention_masks.append(
                    self.runner.model.build_encoder_attention_mask(hidden)
                )

            packed_output = measure(
                "compiled_packed_text_prefill_s1024",
                lambda: text_runtime.run(
                    encoder_hidden_states=encoder_hidden_states,
                ),
            )
            caches = []
            for member, attention_mask in enumerate(encoder_attention_masks):
                decode_mask = self.runner.model.decoder.build_cross_attention_mask(
                    encoder_attention_mask=attention_mask,
                    target_length=1,
                )
                caches.append(
                    measure(
                        "static_cache_build_and_padding",
                        lambda member=member, decode_mask=decode_mask: (
                            LocalUniRecStaticCache.from_cross_prefill(
                                cross_key_cache=packed_output.cross_key_cache[member],
                                cross_value_cache=packed_output.cross_value_cache[member],
                                cross_attention_mask=decode_mask,
                                cache_len=LOCAL_UNIREC_STATIC_CACHE_LEN,
                                cross_cache_len=cross_cache_len,
                            )
                        ),
                    )
                )

        if timeline is None:
            synchronize_device(self.runner.device)
            stage_seconds = None
        else:
            stage_seconds = timeline.resolve()
        wall_s = time.perf_counter() - started

        members = len(images)
        real_text_tokens = packed_output.real_source_tokens
        physical_text_tokens = packed_output.physical_source_tokens
        padding_text_tokens = physical_text_tokens - real_text_tokens
        text_stats = self.runner._packed_text_prefill_stats
        text_stats["packs"] += 1
        text_stats["members"] += members
        text_stats["real_source_tokens"] += real_text_tokens
        text_stats["physical_source_tokens"] += physical_text_tokens
        histogram = text_stats["member_histogram"]
        histogram[str(members)] = histogram.get(str(members), 0) + 1

        self.stats["groups"] += 1
        self.stats["packs"] += len(packs)
        self.stats["packed_members"] += sum(len(pack.crops) for pack in packs)
        self.stats["overflow_members"] += len(overflow)
        self.stats["real_stage_tokens"] += sum(
            pack.stages[ATLAS_STAGE].real_tokens for pack in packs
        )
        self.stats["physical_stage_tokens"] += (
            len(packs) * ATLAS_HEIGHT * ATLAS_WIDTH
        )

        results = []
        for member, ((_inputs, prep), cache) in enumerate(zip(prepared, caches)):
            member_stages = None
            if stage_seconds is not None:
                member_stages = {
                    name: seconds / members for name, seconds in stage_seconds.items()
                }
            generated_ids = self.runner.model.decoder_start_ids(
                batch_size=1,
                device=cache.key_cache[0].device,
            )
            member_physical = packed_output.segment_lengths[member]
            if member == members - 1:
                member_physical += padding_text_tokens
            results.append(
                UniRecPrefilledItem(
                    prep=prep,
                    kv_cache=cache,
                    generated_ids=generated_ids,
                    next_token=generated_ids,
                    prefill_s=wall_s / members,
                    prefill_device_stage_s=member_stages,
                    text_prefill_execution="compiled_packed_s1024",
                    text_prefill_real_source_tokens=(
                        packed_output.segment_lengths[member]
                    ),
                    text_prefill_physical_source_tokens=member_physical,
                )
            )
        return results

    def summary(self) -> dict[str, Any]:
        result = dict(self.stats)
        physical = int(result["physical_stage_tokens"])
        result.update(
            {
                "execution": "eager_patch_stem_compiled_atlas_stages0_2",
                "stages": [0, 1, 2],
                "atlas_height": ATLAS_HEIGHT,
                "atlas_width": ATLAS_WIDTH,
                "guard": ATLAS_GUARD,
                "max_members": ATLAS_MAX_MEMBERS,
                "fill_fraction": (
                    int(result["real_stage_tokens"]) / physical if physical else None
                ),
                "compile_api": self.compile_api,
                "torchair_cache_dir": str(self.cache_dir),
            }
        )
        return result
