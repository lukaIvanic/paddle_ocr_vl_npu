"""Compiled guarded-atlas execution for UniRec FocalSVTR stage 2."""

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
ATLAS_HEIGHT = 64
ATLAS_WIDTH = 192
ATLAS_GUARD = 3
ATLAS_MAX_MEMBERS = 16
ATLAS_CHANNELS = 384
ATLAS_STAGE_FACTOR = 16


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

    @property
    def inner_y(self) -> int:
        return self.y + ATLAS_GUARD

    @property
    def inner_x(self) -> int:
        return self.x + ATLAS_GUARD


@dataclass
class Shelf:
    y: int
    height: int
    next_x: int = 0


@dataclass
class AtlasPack:
    shelves: list[Shelf]
    placements: list[Placement]

    @classmethod
    def empty(cls) -> "AtlasPack":
        return cls(shelves=[], placements=[])

    def try_add(self, crop: CropShape) -> bool:
        if len(self.placements) >= ATLAS_MAX_MEMBERS:
            return False
        physical_h = crop.height + 2 * ATLAS_GUARD
        physical_w = crop.width + 2 * ATLAS_GUARD
        if physical_h > ATLAS_HEIGHT or physical_w > ATLAS_WIDTH:
            return False
        selected = None
        for shelf in self.shelves:
            if physical_h <= shelf.height and shelf.next_x + physical_w <= ATLAS_WIDTH:
                selected = shelf
                break
        if selected is None:
            next_y = sum(shelf.height for shelf in self.shelves)
            if next_y + physical_h > ATLAS_HEIGHT:
                return False
            selected = Shelf(y=next_y, height=physical_h)
            self.shelves.append(selected)
        self.placements.append(
            Placement(
                crop=crop,
                member=len(self.placements),
                y=selected.y,
                x=selected.next_x,
            )
        )
        selected.next_x += physical_w
        return True

    @property
    def real_tokens(self) -> int:
        return sum(placement.crop.tokens for placement in self.placements)


class GuardedAtlasStage(nn.Module):
    """Run stage-2 focal blocks on independent masked atlas regions."""

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


class RoutedGuardedAtlasStage(nn.Module):
    """Route one fixed flat reservoir through a fixed stage-2 atlas."""

    def __init__(self, stage: nn.Module):
        super().__init__()
        self.atlas_stage = GuardedAtlasStage(stage)

    def forward(
        self,
        packed_source: torch.Tensor,
        atlas_to_source: torch.Tensor,
        source_to_atlas: torch.Tensor,
        valid_mask: torch.Tensor,
        membership: torch.Tensor,
        normalized_membership: torch.Tensor,
    ) -> torch.Tensor:
        cells = ATLAS_HEIGHT * ATLAS_WIDTH
        channels = packed_source.shape[-1]
        atlas = packed_source.index_select(1, atlas_to_source).reshape(
            1, ATLAS_HEIGHT, ATLAS_WIDTH, channels
        ).permute(0, 3, 1, 2).contiguous()
        output = self.atlas_stage(
            atlas,
            valid_mask,
            membership,
            normalized_membership,
        )
        output = output.permute(0, 2, 3, 1).reshape(1, cells, channels)
        return output.index_select(1, source_to_atlas)


def _source_hash() -> str:
    payload = Path(__file__).read_bytes()
    payload += Path(__file__).with_name("modeling_optimized_unirec.py").read_bytes()
    return hashlib.sha256(payload).hexdigest()[:12]


def _pack_shapes(shapes: list[CropShape]) -> tuple[list[AtlasPack], list[CropShape]]:
    ordered = sorted(
        shapes,
        key=lambda crop: (
            (crop.height + 2 * ATLAS_GUARD) * (crop.width + 2 * ATLAS_GUARD),
            crop.height,
            crop.width,
        ),
        reverse=True,
    )
    packs: list[AtlasPack] = []
    overflow: list[CropShape] = []
    for crop in ordered:
        if (
            crop.height + 2 * ATLAS_GUARD > ATLAS_HEIGHT
            or crop.width + 2 * ATLAS_GUARD > ATLAS_WIDTH
        ):
            overflow.append(crop)
            continue
        for pack in packs:
            if pack.try_add(crop):
                break
        else:
            pack = AtlasPack.empty()
            if not pack.try_add(crop):
                raise AssertionError("fresh atlas rejected a fitting crop")
            packs.append(pack)
    return packs, overflow


def _routing_inputs(
    pack: AtlasPack,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    cells = ATLAS_HEIGHT * ATLAS_WIDTH
    atlas_to_source = [-1] * cells
    source_cursor = 0
    valid = torch.zeros((1, 1, ATLAS_HEIGHT, ATLAS_WIDTH), dtype=dtype)
    membership = torch.zeros((ATLAS_MAX_MEMBERS, cells), dtype=dtype)
    for placement in pack.placements:
        crop = placement.crop
        for row in range(crop.height):
            atlas_row = placement.inner_y + row
            for column in range(crop.width):
                atlas_index = atlas_row * ATLAS_WIDTH + placement.inner_x + column
                atlas_to_source[atlas_index] = source_cursor + row * crop.width + column
        valid[
            :,
            :,
            placement.inner_y : placement.inner_y + crop.height,
            placement.inner_x : placement.inner_x + crop.width,
        ] = 1
        member_map = membership[placement.member].view(ATLAS_HEIGHT, ATLAS_WIDTH)
        member_map[
            placement.inner_y : placement.inner_y + crop.height,
            placement.inner_x : placement.inner_x + crop.width,
        ] = 1
        source_cursor += crop.tokens
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
        torch.zeros((1, cells, ATLAS_CHANNELS), dtype=dtype, device=device),
        torch.tensor(atlas_to_source, dtype=torch.long, device=device),
        torch.tensor(source_to_atlas, dtype=torch.long, device=device),
        valid.to(device),
        membership.to(device),
        normalized.to(device),
    )


class UniRecVisionAtlasRuntime:
    """Run stage 2 in atlases, then continue the stock crop-local pipeline."""

    def __init__(self, runner: OptimizedUniRecRunner) -> None:
        if not runner.device.startswith("npu"):
            raise ValueError("compiled UniRec vision atlas requires an NPU")
        if runner.compile_cache_dir is None:
            raise ValueError("compiled UniRec vision atlas requires compile_cache_dir")
        self.runner = runner
        self.stage = runner.model.encoder.vision_encoder.layers[ATLAS_STAGE]
        self.module = RoutedGuardedAtlasStage(self.stage).eval()
        self.cache_dir = runner.compile_cache_dir / (
            f"vision_atlas_stage2_{ATLAS_HEIGHT}x{ATLAS_WIDTH}_"
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

    def _run_prefix(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        vision = self.runner.model.encoder.vision_encoder
        x = vision.patch_embed(pixel_values)
        height, width = x.shape[2:]
        x = vision.pos_drop(x.flatten(2).transpose(1, 2))
        for layer in vision.layers[:ATLAS_STAGE]:
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
        packs: list[AtlasPack],
        prefix_states: dict[int, torch.Tensor],
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
            for placement in pack.placements:
                source = prefix_states[placement.crop.source_index]
                packed_source[:, cursor : cursor + placement.crop.tokens].copy_(source)
                cursor += placement.crop.tokens
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
            for placement in pack.placements:
                end = cursor + placement.crop.tokens
                outputs[placement.crop.source_index] = result[:, cursor:end]
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
        prefix_states: dict[int, torch.Tensor] = {}
        shapes: list[CropShape] = []
        with torch.inference_mode():
            for source_index, (inputs, _prep) in enumerate(prepared):
                x, height, width = measure(
                    "vision_crop_prefix_stages_0_1",
                    lambda inputs=inputs: self._run_prefix(inputs["pixel_values"]),
                )
                prefix_states[source_index] = x
                shapes.append(CropShape(source_index, height, width))

            packs, overflow = _pack_shapes(shapes)
            atlas_outputs = measure(
                "compiled_vision_atlas_stage2",
                lambda: self._run_atlas_packs(packs, prefix_states),
            )
            for crop in overflow:
                atlas_outputs[crop.source_index] = measure(
                    "vision_stage2_eager_overflow",
                    lambda crop=crop: self._run_stage2_eager(
                        prefix_states[crop.source_index], crop.height, crop.width
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
        self.stats["packed_members"] += sum(len(pack.placements) for pack in packs)
        self.stats["overflow_members"] += len(overflow)
        self.stats["real_stage_tokens"] += sum(pack.real_tokens for pack in packs)
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
                "execution": "compiled_atlas_stage2",
                "stage": ATLAS_STAGE,
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
