"""Masked fixed-canvas batching for the complete UniRec vision encoder."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn

from modeling_optimized_unirec import (
    OptimizedUniRecRunner,
    import_torchair_cache_compile,
    synchronize_device,
)
from vision_focal_depthwise import (
    VISION_FOCAL_DEPTHWISE_REWRITE_CHOICES,
    rewrite_vision_focal_depthwise_convs,
    vision_rewrite_source_hash,
)


@dataclass(frozen=True, order=True)
class VisionBucketSpec:
    width: int
    height: int
    batch_size: int

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1 or self.batch_size < 1:
            raise ValueError(f"invalid UniRec vision bucket: {self}")
        if self.width % 32 or self.height % 32:
            raise ValueError(
                "UniRec vision bucket dimensions must be divisible by 32: "
                f"{self.width}x{self.height}"
            )

    @property
    def key(self) -> str:
        return f"{self.width}x{self.height}_b{self.batch_size}"

    def accepts(self, width: int, height: int) -> bool:
        return width <= self.width and height <= self.height


# Five graphs cover 1,513/1,564 accepted crops in the first 32 hard pages.
DEFAULT_VISION_BUCKETS = (
    VisionBucketSpec(width=960, height=64, batch_size=16),
    VisionBucketSpec(width=512, height=256, batch_size=16),
    VisionBucketSpec(width=960, height=256, batch_size=4),
    VisionBucketSpec(width=512, height=512, batch_size=8),
    VisionBucketSpec(width=960, height=512, batch_size=4),
)


VISION_WEIGHT_FORMAT_CHOICES = (
    "native",
    "torchair_internal",
)


def _prepare_vision_weight_formats(
    vision_encoder: nn.Module,
    *,
    requested: str,
) -> dict[str, Any]:
    """Apply the proven TorchAir internal-weight pass to the vision encoder."""
    if requested not in VISION_WEIGHT_FORMAT_CHOICES:
        raise ValueError(f"unsupported vision weight format: {requested}")
    import torch_npu

    tracked = [
        (name, module)
        for name, module in vision_encoder.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]

    def histogram() -> dict[str, int]:
        result: dict[str, int] = {}
        for _name, module in tracked:
            code = str(int(torch_npu.get_npu_format(module.weight)))
            result[code] = result.get(code, 0) + 1
        return dict(sorted(result.items()))

    before = histogram()
    if requested == "torchair_internal":
        torch_npu.npu.config.allow_internal_format = True
        try:
            from torch_npu.dynamo.torchair import use_internal_format_weight
        except ImportError:
            from torchair import use_internal_format_weight
        use_internal_format_weight(vision_encoder)
    return {
        "requested": requested,
        "tracked_count": len(tracked),
        "before_histogram": before,
        "after_histogram": histogram(),
    }


def _vision_weight_format_source_hash(requested: str) -> str:
    if requested == "native":
        return ""
    return hashlib.sha256(
        inspect.getsource(_prepare_vision_weight_formats).encode("utf-8")
    ).hexdigest()[:12]


@dataclass(frozen=True)
class PreprocessedVisionInput:
    source_index: int
    pixel_values: np.ndarray
    original_image_size: tuple[int, int]
    image_source: str

    @property
    def input_contract(self) -> str:
        pixels = self.pixel_values
        if (
            pixels.dtype == np.uint8
            and pixels.ndim == 3
            and pixels.shape[2] == 3
        ):
            return "compact_uint8_hwc"
        if (
            pixels.dtype == np.float32
            and pixels.ndim == 4
            and pixels.shape[0] == 1
            and pixels.shape[1] == 3
        ):
            return "legacy_float32_bchw"
        raise ValueError(
            f"invalid processed vision input for {self.image_source}: "
            f"{pixels.dtype} {pixels.shape}"
        )

    @property
    def processed_height(self) -> int:
        if self.input_contract == "compact_uint8_hwc":
            return int(self.pixel_values.shape[0])
        return int(self.pixel_values.shape[2])

    @property
    def processed_width(self) -> int:
        if self.input_contract == "compact_uint8_hwc":
            return int(self.pixel_values.shape[1])
        return int(self.pixel_values.shape[3])


@dataclass
class EncodedVisionItem:
    source_index: int
    hidden_states: torch.Tensor
    prep: dict[str, Any]
    bucket_key: str | None


def _mask_nhwc(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    return x * valid_mask.permute(0, 2, 3, 1)


def _masked_per_row_global_context(
    ctx: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Match native mean(width), then mean(height), for each batch row."""
    row_counts = valid_mask.sum(dim=3, keepdim=True).clamp_min(1)
    row_means = (ctx * valid_mask).sum(dim=3, keepdim=True) / row_counts
    valid_rows = (valid_mask.sum(dim=3, keepdim=True) > 0).to(ctx.dtype)
    return (row_means * valid_rows).sum(dim=2, keepdim=True) / valid_rows.sum(
        dim=2,
        keepdim=True,
    ).clamp_min(1)


def _run_masked_focal_block(
    block: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Run one focal block while making padded cells mathematically inert."""
    _batch, channels, _height, _width = x.shape
    shortcut = x * valid_mask
    normalized = block.norm1(shortcut.permute(0, 2, 3, 1))
    modulation = block.modulation
    projected = modulation.f(normalized).permute(0, 3, 1, 2).contiguous()
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
        raise RuntimeError("UniRec focal modulation unexpectedly has no focal layers")
    global_context = modulation.act(
        _masked_per_row_global_context(ctx, valid_mask)
    )
    ctx_all = ctx_all + global_context * gates[:, modulation.focal_level :]
    modulator = modulation.h(ctx_all) * valid_mask
    modulated = q * modulator
    modulated = modulated.permute(0, 2, 3, 1).contiguous()
    modulated = _mask_nhwc(modulation.proj(modulated), valid_mask)
    residual = shortcut.permute(0, 2, 3, 1) + modulated
    output = residual + block.mlp(block.norm2(residual))
    return _mask_nhwc(output, valid_mask).permute(0, 3, 1, 2).contiguous()


class _MaskedFullVisionEncoder(nn.Module):
    """Complete UniRec vision encoder over independent masked batch rows."""

    def __init__(self, runner: OptimizedUniRecRunner) -> None:
        super().__init__()
        encoder = runner.model.encoder
        vision = encoder.vision_encoder
        self.stem0 = vision.patch_embed[0]
        self.stem1 = vision.patch_embed[1]
        self.pos_drop = vision.pos_drop
        self.stages = vision.layers
        self.projection = encoder.vision_fc
        if len(self.stages) != 4:
            raise ValueError(
                f"masked UniRec vision expects four stages, got {len(self.stages)}"
            )
        for index, stage in enumerate(self.stages[:3]):
            if stage.downsample is None:
                raise ValueError(f"UniRec vision stage {index} has no downsampler")
        if self.stages[3].downsample is not None:
            raise ValueError("UniRec vision stage 3 unexpectedly has a downsampler")

    @staticmethod
    def _tokens_to_chw(
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch = tokens.shape[0]
        height, width = mask.shape[2:]
        tokens = tokens * mask.flatten(2).transpose(1, 2)
        return tokens.transpose(1, 2).reshape(batch, -1, height, width)

    def _forward_fixed(
        self,
        pixel_values: torch.Tensor,
        mask2: torch.Tensor,
        mask4: torch.Tensor,
        mask8: torch.Tensor,
        mask16: torch.Tensor,
        mask32: torch.Tensor,
    ) -> torch.Tensor:
        x = self.stem0(pixel_values) * mask2
        x = self.stem1(x) * mask4
        x = self.pos_drop(x.flatten(2).transpose(1, 2))
        x = self._tokens_to_chw(x, mask4)
        stage_masks = (mask4, mask8, mask16, mask32)
        for stage_index, stage in enumerate(self.stages):
            stage_mask = stage_masks[stage_index]
            for block in stage.blocks:
                x = _run_masked_focal_block(block, x, stage_mask)
            if stage.downsample is not None:
                x = stage.downsample(x)[0]
                x = self._tokens_to_chw(x, stage_masks[stage_index + 1])
        # The stock stage-3 path returns a contiguous BTC tensor. Preserve
        # that contract explicitly: GE otherwise sees the transposed view's
        # token stride and can infer T, rather than C, as the linear K axis.
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        tokens = self.projection(tokens)
        return tokens * mask32.flatten(2).transpose(1, 2)

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask2: torch.Tensor,
        mask4: torch.Tensor,
        mask8: torch.Tensor,
        mask16: torch.Tensor,
        mask32: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_fixed(
            pixel_values,
            mask2,
            mask4,
            mask8,
            mask16,
            mask32,
        )


def _new_masked_full_encoder_module(
    runner: OptimizedUniRecRunner,
    spec: VisionBucketSpec,
) -> _MaskedFullVisionEncoder:
    """Give each static bucket a distinct deterministic forward identity."""
    filename = f"<unirec_full_vision_{spec.key}>"
    namespace: dict[str, Any] = {}
    exec(
        compile(
            "def forward(self, pixel_values, mask2, mask4, mask8, mask16, mask32):\n"
            "    return self._forward_fixed(pixel_values, mask2, mask4, mask8, mask16, mask32)\n",
            filename,
            "exec",
        ),
        namespace,
    )
    module_type = type(
        f"MaskedFullVisionEncoder_{spec.width}x{spec.height}_b{spec.batch_size}",
        (_MaskedFullVisionEncoder,),
        {"forward": namespace["forward"]},
    )
    return module_type(runner).eval()


def _source_hash() -> str:
    payload = inspect.getsource(_MaskedFullVisionEncoder).encode("utf-8")
    payload += inspect.getsource(_run_masked_focal_block).encode("utf-8")
    payload += inspect.getsource(_masked_per_row_global_context).encode("utf-8")
    payload += inspect.getsource(_new_masked_full_encoder_module).encode("utf-8")
    payload += Path(__file__).with_name("modeling_optimized_unirec.py").read_bytes()
    return hashlib.sha256(payload).hexdigest()[:12]


def _make_host_masks(
    dimensions: Sequence[tuple[int, int]],
    *,
    spec: VisionBucketSpec,
) -> tuple[np.ndarray, ...]:
    masks = []
    for factor in (2, 4, 8, 16, 32):
        mask = np.zeros(
            (
                spec.batch_size,
                1,
                spec.height // factor,
                spec.width // factor,
            ),
            dtype=np.float16,
        )
        for row, (width, height) in enumerate(dimensions):
            mask[row, :, : height // factor, : width // factor] = 1
        masks.append(mask)
    return tuple(masks)


def _compact_uint8_hwc_to_device(
    host_pixels: np.ndarray,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Transfer compact HWC pixels, then normalize and transpose on device."""
    pixels = torch.from_numpy(host_pixels).to(device)
    if pixels.ndim == 3:
        pixels = pixels.permute(2, 0, 1).unsqueeze(0)
    elif pixels.ndim == 4:
        pixels = pixels.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"compact pixels must be HWC or BHWC, got {pixels.shape}")
    output = pixels.to(torch.float32)
    output.mul_(np.float32(2.0 / 255.0))
    output.sub_(np.float32(1.0))
    return output.to(dtype).contiguous()


class BucketedFullVisionRuntime:
    """Dispatch real processed crops through a small full-encoder bucket set."""

    def __init__(
        self,
        runner: OptimizedUniRecRunner,
        *,
        specs: Sequence[VisionBucketSpec] = DEFAULT_VISION_BUCKETS,
        diagnostic_graph_log: bool = False,
        focal_depthwise_rewrite: str = "native",
        weight_format: str = "native",
    ) -> None:
        if not specs:
            raise ValueError("bucketed UniRec vision requires at least one bucket")
        if runner.compile_cache_dir is None:
            raise ValueError("bucketed UniRec vision requires compile_cache_dir")
        self.runner = runner
        self.specs = tuple(specs)
        self.diagnostic_graph_log = bool(diagnostic_graph_log)
        self.focal_depthwise_rewrite = str(focal_depthwise_rewrite)
        self.focal_depthwise_rewrite_summary = (
            rewrite_vision_focal_depthwise_convs(
                runner.model.encoder.vision_encoder,
                requested=self.focal_depthwise_rewrite,
            )
        )
        self.weight_format = str(weight_format)
        self.weight_format_summary = _prepare_vision_weight_formats(
            runner.model.encoder.vision_encoder,
            requested=self.weight_format,
        )
        required_specializations = len(self.specs) + 16
        torch._dynamo.config.cache_size_limit = max(
            int(torch._dynamo.config.cache_size_limit),
            required_specializations,
        )
        torch._dynamo.config.recompile_limit = max(
            int(torch._dynamo.config.recompile_limit),
            required_specializations,
        )
        torch._dynamo.config.accumulated_cache_size_limit = max(
            int(torch._dynamo.config.accumulated_cache_size_limit),
            required_specializations * 4,
        )
        torch._dynamo.config.accumulated_recompile_limit = max(
            int(torch._dynamo.config.accumulated_recompile_limit),
            required_specializations * 4,
        )

        from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

        config = CompilerConfig()
        config.mode.value = "max-autotune"
        cache_compile, import_path = import_torchair_cache_compile()
        source_hash = _source_hash()
        rewrite_hash = vision_rewrite_source_hash(
            self.focal_depthwise_rewrite
        )
        rewrite_cache_suffix = (
            ""
            if self.focal_depthwise_rewrite == "native"
            else f"_dw{self.focal_depthwise_rewrite}_src{rewrite_hash}"
        )
        weight_format_hash = _vision_weight_format_source_hash(
            self.weight_format
        )
        weight_format_cache_suffix = (
            ""
            if self.weight_format == "native"
            else f"_w{self.weight_format}_src{weight_format_hash}"
        )
        self.compile_api = import_path
        self.modules: dict[str, _MaskedFullVisionEncoder] = {}
        self.compiled: dict[str, Callable[..., torch.Tensor]] = {}
        self.cache_dirs: dict[str, Path] = {}
        self._diagnostic_bucket_calls = {spec.key: 0 for spec in self.specs}
        self._diagnostic_fallback_calls = 0
        self._diagnostic_log(
            "runtime_init_begin",
            graph_count=len(self.specs),
            graph_keys=[spec.key for spec in self.specs],
            compile_api=import_path,
        )
        for graph_index, spec in enumerate(self.specs):
            module = _new_masked_full_encoder_module(runner, spec)
            cache_dir = runner.compile_cache_dir / (
                f"vision_full_bucket_{spec.key}_{runner.dtype_name}_src{source_hash}"
                f"{rewrite_cache_suffix}"
                f"{weight_format_cache_suffix}"
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.modules[spec.key] = module
            self.cache_dirs[spec.key] = cache_dir
            self._diagnostic_log(
                "graph_registration_begin",
                graph_index=graph_index,
                graph_count=len(self.specs),
                bucket=spec.key,
                width=spec.width,
                height=spec.height,
                batch_size=spec.batch_size,
                cache=self._cache_snapshot(cache_dir),
            )
            registration_started = time.perf_counter()
            self.compiled[spec.key] = cache_compile(
                module.forward,
                config=config,
                dynamic=False,
                cache_dir=str(cache_dir),
                ge_cache=True,
                fullgraph=True,
            )
            self._diagnostic_log(
                "graph_registration_end",
                graph_index=graph_index,
                graph_count=len(self.specs),
                bucket=spec.key,
                registration_wall_s=time.perf_counter() - registration_started,
                cache=self._cache_snapshot(cache_dir),
            )
        self._diagnostic_log(
            "runtime_init_end",
            graph_count=len(self.specs),
            graph_keys=[spec.key for spec in self.specs],
        )
        self.stats: dict[str, Any] = {
            "bucket_calls": {spec.key: 0 for spec in self.specs},
            "bucket_real_rows": {spec.key: 0 for spec in self.specs},
            "fallback_rows": 0,
            "compact_input_rows": 0,
            "legacy_input_rows": 0,
            "batch_h2d_s": 0.0,
            "vision_wall_s": 0.0,
            "output_compact_s": 0.0,
            "first_call_wall_s": {},
        }

    @staticmethod
    def _cache_snapshot(cache_dir: Path) -> dict[str, Any]:
        try:
            om_files = sorted(cache_dir.rglob("*.om"))
            return {
                "directory": str(cache_dir),
                "om_count": len(om_files),
                "om_bytes": sum(path.stat().st_size for path in om_files),
            }
        except OSError as exception:
            return {
                "directory": str(cache_dir),
                "error": repr(exception),
            }

    @staticmethod
    def _cache_inventory_snapshot(cache_dir: Path) -> dict[str, Any]:
        snapshot = BucketedFullVisionRuntime._cache_snapshot(cache_dir)
        if "error" in snapshot:
            return snapshot
        try:
            snapshot["om_files"] = [
                {
                    "path": str(path.relative_to(cache_dir)),
                    "size_bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for path in sorted(cache_dir.rglob("*.om"))
            ]
        except OSError as exception:
            snapshot["inventory_error"] = repr(exception)
        return snapshot

    def cache_inventory(self) -> dict[str, dict[str, Any]]:
        """Return exact per-bucket OM identities for compile diagnostics."""
        return {
            spec.key: self._cache_inventory_snapshot(self.cache_dirs[spec.key])
            for spec in self.specs
        }

    def _memory_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        device = torch.device(self.runner.device)
        for name in (
            "memory_allocated",
            "memory_reserved",
            "max_memory_allocated",
            "max_memory_reserved",
        ):
            operation = getattr(torch.npu, name, None)
            if operation is None:
                continue
            try:
                snapshot[f"{name}_bytes"] = int(operation(device))
            except BaseException as exception:  # diagnostic must not mask inference
                snapshot[f"{name}_error"] = repr(exception)
        return snapshot

    def _diagnostic_log(self, event: str, **fields: Any) -> None:
        if not self.diagnostic_graph_log:
            return
        payload = {
            "event": event,
            "monotonic_s": time.monotonic(),
            "pid": os.getpid(),
            "npu_memory": self._memory_snapshot(),
            **fields,
        }
        print(
            "UNIREC_VISION_GRAPH_DIAGNOSTIC "
            + json.dumps(payload, sort_keys=True),
            flush=True,
        )

    def diagnostic_event(self, event: str, **fields: Any) -> None:
        """Emit one opt-in lab-level event with the runtime memory snapshot."""
        self._diagnostic_log(event, **fields)

    def select_bucket(self, width: int, height: int) -> VisionBucketSpec | None:
        candidates = [spec for spec in self.specs if spec.accepts(width, height)]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda spec: (
                spec.width * spec.height,
                spec.batch_size,
                spec.height,
                spec.width,
            ),
        )

    def _prep_metadata(self, item: PreprocessedVisionInput) -> dict[str, Any]:
        return {
            "image": item.image_source,
            "original_image_size": [
                int(item.original_image_size[0]),
                int(item.original_image_size[1]),
            ],
            "processed_image_size": [
                item.processed_width,
                item.processed_height,
            ],
            "encoder_seq_len_hint": int(
                self.runner.processor.estimate_encoder_token_count_from_processed_size(
                    processed_width=item.processed_width,
                    processed_height=item.processed_height,
                )
            ),
            "pixel_values_shape": list(item.pixel_values.shape),
            "pixel_values_contract": item.input_contract,
            "image_load_s": 0.0,
            "image_preprocess_s": 0.0,
            "move_to_device_s": 0.0,
            "prepare_total_s": 0.0,
            "worker_preprocessed": True,
        }

    def _run_bucket(
        self,
        spec: VisionBucketSpec,
        items: Sequence[PreprocessedVisionInput],
    ) -> list[EncodedVisionItem]:
        if not items or len(items) > spec.batch_size:
            raise ValueError(
                f"bucket {spec.key} received {len(items)} real rows"
            )
        self._diagnostic_bucket_calls[spec.key] += 1
        diagnostic_call = self._diagnostic_bucket_calls[spec.key]
        self._diagnostic_log(
            "bucket_call_begin",
            bucket=spec.key,
            bucket_call=diagnostic_call,
            real_rows=len(items),
            physical_rows=spec.batch_size,
            processed_shapes=[
                [item.processed_width, item.processed_height] for item in items
            ],
        )
        contracts = {item.input_contract for item in items}
        if len(contracts) != 1:
            raise ValueError(
                f"bucket {spec.key} cannot mix input contracts: {contracts}"
            )
        compact_input = contracts == {"compact_uint8_hwc"}
        if compact_input:
            host_pixels = np.zeros(
                (spec.batch_size, spec.height, spec.width, 3),
                dtype=np.uint8,
            )
            host_pixel_mask = np.zeros(
                (spec.batch_size, 1, spec.height, spec.width),
                dtype=np.uint8,
            )
        else:
            host_pixels = np.zeros(
                (spec.batch_size, 3, spec.height, spec.width),
                dtype=np.float32,
            )
            host_pixel_mask = None
        dimensions = []
        for row, item in enumerate(items):
            pixels = item.pixel_values
            if not spec.accepts(item.processed_width, item.processed_height):
                raise ValueError(
                    f"{item.processed_width}x{item.processed_height} does not fit "
                    f"bucket {spec.key}"
                )
            if compact_input:
                host_pixels[
                    row,
                    : item.processed_height,
                    : item.processed_width,
                    :,
                ] = pixels
                assert host_pixel_mask is not None
                host_pixel_mask[
                    row,
                    :,
                    : item.processed_height,
                    : item.processed_width,
                ] = 1
            else:
                host_pixels[
                    row,
                    :,
                    : item.processed_height,
                    : item.processed_width,
                ] = pixels[0]
            dimensions.append((item.processed_width, item.processed_height))
        host_masks = _make_host_masks(dimensions, spec=spec)

        transfer_started = time.perf_counter()
        with torch.inference_mode(False):
            if compact_input:
                pixel_values = _compact_uint8_hwc_to_device(
                    host_pixels,
                    device=self.runner.device,
                    dtype=self.runner.dtype,
                )
                assert host_pixel_mask is not None
                pixel_mask = torch.from_numpy(host_pixel_mask).to(
                    self.runner.device,
                    dtype=self.runner.dtype,
                )
                pixel_values.mul_(pixel_mask)
            else:
                pixel_values = torch.from_numpy(host_pixels).to(
                    self.runner.device,
                    dtype=self.runner.dtype,
                )
            masks = tuple(
                torch.from_numpy(mask).to(self.runner.device)
                for mask in host_masks
            )
        self.stats["batch_h2d_s"] += time.perf_counter() - transfer_started
        self._diagnostic_log(
            "bucket_inputs_ready",
            bucket=spec.key,
            bucket_call=diagnostic_call,
            h2d_wall_s=time.perf_counter() - transfer_started,
        )

        first_call = self.stats["bucket_calls"][spec.key] == 0
        started = time.perf_counter()
        # Warmup uses inference mode. Keep the production call under the same
        # Dynamo guard so the first real crop batch loads that graph instead of
        # compiling a second grad-enabled specialization in the timed window.
        self._diagnostic_log(
            "bucket_graph_call_begin",
            bucket=spec.key,
            bucket_call=diagnostic_call,
            first_workload_call=first_call,
            cache=self._cache_snapshot(self.cache_dirs[spec.key]),
        )
        with torch.inference_mode():
            output = self.compiled[spec.key](pixel_values, *masks)
        self._diagnostic_log(
            "bucket_graph_submit_end",
            bucket=spec.key,
            bucket_call=diagnostic_call,
            first_workload_call=first_call,
            submit_wall_s=time.perf_counter() - started,
        )
        if first_call:
            self._diagnostic_log(
                "bucket_first_call_sync_begin",
                bucket=spec.key,
                bucket_call=diagnostic_call,
            )
            synchronize_device(self.runner.device)
            self.stats["first_call_wall_s"][spec.key] = (
                time.perf_counter() - started
            )
            self._diagnostic_log(
                "bucket_first_call_sync_end",
                bucket=spec.key,
                bucket_call=diagnostic_call,
                synchronized_wall_s=time.perf_counter() - started,
                cache=self._cache_snapshot(self.cache_dirs[spec.key]),
            )
        self.stats["vision_wall_s"] += time.perf_counter() - started
        self.stats["bucket_calls"][spec.key] += 1
        self.stats["bucket_real_rows"][spec.key] += len(items)
        input_row_key = (
            "compact_input_rows" if compact_input else "legacy_input_rows"
        )
        self.stats[input_row_key] += len(items)

        compact_started = time.perf_counter()
        grid = output.reshape(
            spec.batch_size,
            spec.height // 32,
            spec.width // 32,
            output.shape[-1],
        )
        encoded = []
        for row, item in enumerate(items):
            hidden = grid[
                row : row + 1,
                : item.processed_height // 32,
                : item.processed_width // 32,
            ].reshape(1, -1, output.shape[-1]).contiguous()
            encoded.append(
                EncodedVisionItem(
                    source_index=item.source_index,
                    hidden_states=hidden,
                    prep=self._prep_metadata(item),
                    bucket_key=spec.key,
                )
            )
        self.stats["output_compact_s"] += time.perf_counter() - compact_started
        self._diagnostic_log(
            "bucket_call_end",
            bucket=spec.key,
            bucket_call=diagnostic_call,
            real_rows=len(items),
            output_rows=len(encoded),
            output_compact_wall_s=time.perf_counter() - compact_started,
        )
        return encoded

    def _run_fallback(
        self,
        item: PreprocessedVisionInput,
    ) -> EncodedVisionItem:
        self._diagnostic_fallback_calls += 1
        diagnostic_call = self._diagnostic_fallback_calls
        self._diagnostic_log(
            "fallback_call_begin",
            fallback_call=diagnostic_call,
            request_id=item.image_source,
            processed_shape=[item.processed_width, item.processed_height],
        )
        compact_input = item.input_contract == "compact_uint8_hwc"
        transfer_started = time.perf_counter()
        with torch.inference_mode(False):
            if compact_input:
                pixel_values = _compact_uint8_hwc_to_device(
                    item.pixel_values,
                    device=self.runner.device,
                    dtype=self.runner.dtype,
                )
            else:
                pixel_values = torch.from_numpy(item.pixel_values).to(
                    self.runner.device,
                    dtype=self.runner.dtype,
                )
        self.stats["batch_h2d_s"] += time.perf_counter() - transfer_started
        self._diagnostic_log(
            "fallback_inputs_ready",
            fallback_call=diagnostic_call,
            request_id=item.image_source,
            processed_shape=[item.processed_width, item.processed_height],
            h2d_wall_s=time.perf_counter() - transfer_started,
        )
        started = time.perf_counter()
        self._diagnostic_log(
            "fallback_eager_call_begin",
            fallback_call=diagnostic_call,
            request_id=item.image_source,
            processed_shape=[item.processed_width, item.processed_height],
        )
        with torch.inference_mode():
            hidden = self.runner.model.forward_encoder(pixel_values)
        self._diagnostic_log(
            "fallback_eager_submit_end",
            fallback_call=diagnostic_call,
            request_id=item.image_source,
            processed_shape=[item.processed_width, item.processed_height],
            submit_wall_s=time.perf_counter() - started,
        )
        self.stats["fallback_rows"] += 1
        input_row_key = (
            "compact_input_rows" if compact_input else "legacy_input_rows"
        )
        self.stats[input_row_key] += 1
        encoded = EncodedVisionItem(
            source_index=item.source_index,
            hidden_states=hidden,
            prep=self._prep_metadata(item),
            bucket_key=None,
        )
        self._diagnostic_log(
            "fallback_call_end",
            fallback_call=diagnostic_call,
            request_id=item.image_source,
            processed_shape=[item.processed_width, item.processed_height],
            output_shape=list(hidden.shape),
        )
        return encoded

    def encode(
        self,
        items: Sequence[PreprocessedVisionInput],
    ) -> list[EncodedVisionItem]:
        """Encode arbitrary cross-page inputs and preserve source-index order."""
        grouped: dict[str, list[PreprocessedVisionInput]] = {
            spec.key: [] for spec in self.specs
        }
        fallbacks = []
        specs_by_key = {spec.key: spec for spec in self.specs}
        for item in items:
            spec = self.select_bucket(item.processed_width, item.processed_height)
            if spec is None:
                fallbacks.append(item)
            else:
                grouped[spec.key].append(item)

        outputs: dict[int, EncodedVisionItem] = {}
        for spec in self.specs:
            pending = grouped[spec.key]
            for start in range(0, len(pending), spec.batch_size):
                for output in self._run_bucket(
                    specs_by_key[spec.key],
                    pending[start : start + spec.batch_size],
                ):
                    outputs[output.source_index] = output
        for item in fallbacks:
            output = self._run_fallback(item)
            outputs[output.source_index] = output
        if len(outputs) != len(items):
            raise RuntimeError(
                f"bucketed vision lost rows: {len(outputs)} != {len(items)}"
            )
        return [outputs[item.source_index] for item in items]

    def warmup_all(self, *, passes: int = 1) -> dict[str, Any]:
        if passes < 1:
            raise ValueError("bucketed vision warmup passes must be positive")
        report = {}
        device = torch.device(self.runner.device)
        self._diagnostic_log(
            "warmup_all_begin",
            graph_count=len(self.specs),
            passes=passes,
        )
        for graph_index, spec in enumerate(self.specs):
            self._diagnostic_log(
                "warmup_graph_inputs_begin",
                graph_index=graph_index,
                graph_count=len(self.specs),
                bucket=spec.key,
                passes=passes,
            )
            with torch.inference_mode(False):
                pixels = torch.zeros(
                    (spec.batch_size, 3, spec.height, spec.width),
                    dtype=self.runner.dtype,
                    device=device,
                )
                masks = tuple(
                    torch.ones(
                        (
                            spec.batch_size,
                            1,
                            spec.height // factor,
                            spec.width // factor,
                        ),
                        dtype=self.runner.dtype,
                        device=device,
                    )
                    for factor in (2, 4, 8, 16, 32)
                )
            self._diagnostic_log(
                "warmup_graph_inputs_ready",
                graph_index=graph_index,
                graph_count=len(self.specs),
                bucket=spec.key,
                pixel_shape=list(pixels.shape),
                mask_shapes=[list(mask.shape) for mask in masks],
            )
            pass_wall_s = []
            for pass_index in range(passes):
                self._diagnostic_log(
                    "warmup_graph_call_begin",
                    graph_index=graph_index,
                    graph_count=len(self.specs),
                    bucket=spec.key,
                    pass_index=pass_index,
                    pass_count=passes,
                    cache=self._cache_snapshot(self.cache_dirs[spec.key]),
                )
                started = time.perf_counter()
                with torch.inference_mode():
                    self.compiled[spec.key](pixels, *masks)
                self._diagnostic_log(
                    "warmup_graph_submit_end",
                    graph_index=graph_index,
                    graph_count=len(self.specs),
                    bucket=spec.key,
                    pass_index=pass_index,
                    submit_wall_s=time.perf_counter() - started,
                )
                self._diagnostic_log(
                    "warmup_graph_sync_begin",
                    graph_index=graph_index,
                    graph_count=len(self.specs),
                    bucket=spec.key,
                    pass_index=pass_index,
                )
                synchronize_device(self.runner.device)
                pass_wall_s.append(time.perf_counter() - started)
                self._diagnostic_log(
                    "warmup_graph_call_end",
                    graph_index=graph_index,
                    graph_count=len(self.specs),
                    bucket=spec.key,
                    pass_index=pass_index,
                    synchronized_wall_s=pass_wall_s[-1],
                    cache=self._cache_snapshot(self.cache_dirs[spec.key]),
                )
            report[spec.key] = {
                "pass_wall_s": pass_wall_s,
                "cache_dir": str(self.cache_dirs[spec.key]),
            }
        self._diagnostic_log(
            "warmup_all_end",
            graph_count=len(self.specs),
            passes=passes,
        )
        return report

    def summary(self) -> dict[str, Any]:
        return {
            "spatial_execution": "compiled_masked_full_encoder_buckets",
            "compile_api": self.compile_api,
            "diagnostic_graph_log": self.diagnostic_graph_log,
            "buckets": [
                {
                    "key": spec.key,
                    "width": spec.width,
                    "height": spec.height,
                    "batch_size": spec.batch_size,
                    "cache_dir": str(self.cache_dirs[spec.key]),
                }
                for spec in self.specs
            ],
            **self.stats,
        }
