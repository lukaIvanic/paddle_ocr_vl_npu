"""Static-shape compiled UniRec vision prefix and suffix execution."""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from modeling_optimized_unirec import (
    OptimizedUniRecRunner,
    import_torchair_cache_compile,
    synchronize_device,
)
from vision_atlas import ATLAS_STAGE, UniRecVisionAtlasRuntime


def _run_fixed_focal_block(
    block: nn.Module,
    x: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Run one focal block without mutating its shared ``H``/``W`` fields."""
    batch, _, channels = x.shape
    shortcut = x
    x = block.norm1(x)
    x = x.view(batch, height, width, channels)
    x = block.modulation(x).view(batch, height * width, channels)
    x = shortcut + x
    return x + block.mlp(block.norm2(x))


class _StaticVisionPrefix(nn.Module):
    def __init__(self, runner: OptimizedUniRecRunner, *, input_height: int, input_width: int) -> None:
        super().__init__()
        vision = runner.model.encoder.vision_encoder
        self.patch_embed = vision.patch_embed
        self.pos_drop = vision.pos_drop
        self.stage0_blocks = vision.layers[0].blocks
        self.stage0_downsample = vision.layers[0].downsample
        self.stage1_blocks = vision.layers[1].blocks
        self.stage1_downsample = vision.layers[1].downsample
        if self.stage0_downsample is None or self.stage1_downsample is None:
            raise ValueError("UniRec vision stages 0 and 1 must have downsamplers")
        self.patch_height = input_height // 4
        self.patch_width = input_width // 4

    def _forward_fixed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(pixel_values)
        x = self.pos_drop(x.flatten(2).transpose(1, 2))
        for block in self.stage0_blocks:
            x = _run_fixed_focal_block(
                block,
                x,
                height=self.patch_height,
                width=self.patch_width,
            )
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, self.patch_height, self.patch_width
        )
        x = self.stage0_downsample(x)[0]

        stage1_height = self.patch_height // 2
        stage1_width = self.patch_width // 2
        for block in self.stage1_blocks:
            x = _run_fixed_focal_block(
                block,
                x,
                height=stage1_height,
                width=stage1_width,
            )
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, stage1_height, stage1_width
        )
        x = self.stage1_downsample(x)[0]
        return x

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._forward_fixed(pixel_values)


def _new_static_prefix_module(
    runner: OptimizedUniRecRunner,
    *,
    input_height: int,
    input_width: int,
) -> _StaticVisionPrefix:
    """Give every static shape a deterministic, distinct forward code object.

    TorchAir's disk cache rejects several static specializations of one Python
    code object as recompilations.  A shape-specific forward identity lets all
    graphs coexist and load independently from their own cache directories.
    """
    filename = f"<unirec_static_prefix_{input_width}x{input_height}>"
    namespace: dict[str, Any] = {}
    exec(
        compile(
            "def forward(self, pixel_values):\n"
            "    return self._forward_fixed(pixel_values)\n",
            filename,
            "exec",
        ),
        namespace,
    )
    shape_class = type(
        f"StaticVisionPrefix_{input_width}x{input_height}",
        (_StaticVisionPrefix,),
        {"forward": namespace["forward"]},
    )
    return shape_class(
        runner,
        input_height=input_height,
        input_width=input_width,
    )


class _StaticVisionSuffix(nn.Module):
    def __init__(self, runner: OptimizedUniRecRunner, *, stage_height: int, stage_width: int) -> None:
        super().__init__()
        vision = runner.model.encoder.vision_encoder
        stage2 = vision.layers[ATLAS_STAGE]
        if stage2.downsample is None:
            raise ValueError("UniRec stage 2 must have a downsampler")
        self.downsample = stage2.downsample
        self.stage3 = vision.layers[ATLAS_STAGE + 1]
        self.projection = runner.model.encoder.vision_fc
        self.stage_height = stage_height
        self.stage_width = stage_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).reshape(1, -1, self.stage_height, self.stage_width)
        x, height, width = self.downsample(x)
        x, _, _ = self.stage3(x, height, width)
        return self.projection(x)


def _source_hash() -> str:
    payload = inspect.getsource(_StaticVisionPrefix).encode("utf-8")
    payload += inspect.getsource(_run_fixed_focal_block).encode("utf-8")
    payload += inspect.getsource(_new_static_prefix_module).encode("utf-8")
    payload += inspect.getsource(_StaticVisionSuffix).encode("utf-8")
    payload += Path(__file__).with_name("modeling_optimized_unirec.py").read_bytes()
    return hashlib.sha256(payload).hexdigest()[:12]


class StaticShapeUniRecVisionRuntime(UniRecVisionAtlasRuntime):
    """Keep stage 2 in the atlas and compile crop-local prefix/suffix graphs."""

    def __init__(
        self,
        runner: OptimizedUniRecRunner,
        *,
        input_width: int,
        input_height: int,
    ) -> None:
        if input_width % 32 or input_height % 32:
            raise ValueError("static UniRec vision dimensions must be divisible by 32")
        super().__init__(runner)
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.stage_width = self.input_width // 16
        self.stage_height = self.input_height // 16
        self.prefix_module = _new_static_prefix_module(
            runner,
            input_height=self.input_height,
            input_width=self.input_width,
        ).eval()
        self.suffix_module = _StaticVisionSuffix(
            runner,
            stage_height=self.stage_height,
            stage_width=self.stage_width,
        ).eval()

        from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

        config = CompilerConfig()
        config.mode.value = "max-autotune"
        cache_compile, import_path = import_torchair_cache_compile()
        source_hash = _source_hash()
        cache_root = runner.compile_cache_dir
        if cache_root is None:
            raise ValueError("static UniRec vision compilation requires compile_cache_dir")
        self.prefix_cache_dir = cache_root / (
            f"vision_static_prefix_{self.input_width}x{self.input_height}_"
            f"{runner.dtype_name}_src{source_hash}"
        )
        self.suffix_cache_dir = cache_root / (
            f"vision_static_suffix_{self.input_width}x{self.input_height}_"
            f"{runner.dtype_name}_src{source_hash}"
        )
        self.prefix_cache_dir.mkdir(parents=True, exist_ok=True)
        self.suffix_cache_dir.mkdir(parents=True, exist_ok=True)
        self.compiled_prefix = cache_compile(
            self.prefix_module.forward,
            config=config,
            dynamic=False,
            cache_dir=str(self.prefix_cache_dir),
            ge_cache=True,
            fullgraph=True,
        )
        self.compiled_suffix = cache_compile(
            self.suffix_module.forward,
            config=config,
            dynamic=False,
            cache_dir=str(self.suffix_cache_dir),
            ge_cache=True,
            fullgraph=True,
        )
        self.static_compile_api = import_path

    def _run_prefix(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        expected = (1, 3, self.input_height, self.input_width)
        if tuple(pixel_values.shape) != expected:
            raise ValueError(
                f"static vision prefix expected {expected}, got {tuple(pixel_values.shape)}"
            )
        return self.compiled_prefix(pixel_values), self.stage_height, self.stage_width

    def _run_suffix(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if (height, width) != (self.stage_height, self.stage_width):
            raise ValueError(
                "static vision suffix shape mismatch: "
                f"expected {(self.stage_height, self.stage_width)}, got {(height, width)}"
            )
        return self.compiled_suffix(x)

    def warmup_static_graphs(self, *, passes: int) -> dict[str, Any]:
        device = torch.device(self.runner.device)
        # Production image preprocessing creates pixel_values before entering
        # inference_mode. Match that tensor dispatch-key contract exactly.
        with torch.inference_mode(False):
            prefix_input = torch.zeros(
                (1, 3, self.input_height, self.input_width),
                dtype=self.runner.dtype,
                device=device,
            )
        suffix_input = torch.zeros(
            (1, self.stage_height * self.stage_width, 384),
            dtype=self.runner.dtype,
            device=device,
        )
        report: dict[str, Any] = {}
        for name, compiled, inputs, cache_dir in (
            ("vision_static_prefix", self.compiled_prefix, (prefix_input,), self.prefix_cache_dir),
            ("vision_static_suffix", self.compiled_suffix, (suffix_input,), self.suffix_cache_dir),
        ):
            pass_times = []
            for pass_index in range(passes):
                started = time.perf_counter()
                compiled(*inputs)
                synchronize_device(self.runner.device)
                elapsed = time.perf_counter() - started
                pass_times.append(elapsed)
                print(
                    "UNIREC_GRAPH_WARMUP_PASS "
                    f"graph={name} pass={pass_index + 1}/{passes} wall_s={elapsed:.3f}",
                    flush=True,
                )
            report[name] = {
                "pass_wall_s": pass_times,
                "cache_dir": str(cache_dir),
            }
        return report

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        result.update(
            {
                "spatial_execution": "compiled_static",
                "static_input_size": [self.input_width, self.input_height],
                "static_prefix_cache_dir": str(self.prefix_cache_dir),
                "static_suffix_cache_dir": str(self.suffix_cache_dir),
                "static_compile_api": self.static_compile_api,
            }
        )
        return result


def load_static_vision_shapes(path: Path) -> list[tuple[int, int]]:
    """Load unique ``(width, height)`` shapes from JSON or crop JSONL."""
    path = path.expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    shapes: set[tuple[int, int]] = set()
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        document = None
    if isinstance(document, list):
        values = document
        for value in values:
            if isinstance(value, dict):
                value = value.get("shape") or value.get("processed_image_size")
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"invalid static vision shape entry: {value!r}")
            shapes.add((int(value[0]), int(value[1])))
    else:
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                value = row["prefill"]["prep"]["processed_image_size"]
            except (KeyError, TypeError) as exception:
                raise ValueError(
                    f"{path}:{line_number} has no processed image size"
                ) from exception
            shapes.add((int(value[0]), int(value[1])))
    if not shapes:
        raise ValueError(f"static vision shape manifest is empty: {path}")
    for width, height in shapes:
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise ValueError(f"invalid static UniRec vision shape: {width}x{height}")
    return sorted(shapes, key=lambda shape: (shape[1] * shape[0], shape[1], shape[0]))


class PerShapeCompiledPrefixUniRecVisionRuntime(UniRecVisionAtlasRuntime):
    """Dispatch stages 0-1 to one immutable compiled graph per input shape."""

    def __init__(
        self,
        runner: OptimizedUniRecRunner,
        *,
        shapes: list[tuple[int, int]],
    ) -> None:
        super().__init__(runner)
        self.shapes = tuple(shapes)
        # TorchDynamo defaults to eight specializations for a shared nested
        # Python code object. This registry deliberately owns one static graph
        # per declared image shape, so size both compiler limits explicitly.
        required_cache_entries = len(self.shapes) + 8
        torch._dynamo.config.cache_size_limit = max(
            int(torch._dynamo.config.cache_size_limit),
            required_cache_entries,
        )
        torch._dynamo.config.accumulated_cache_size_limit = max(
            int(torch._dynamo.config.accumulated_cache_size_limit),
            required_cache_entries * 4,
        )
        self.prefix_modules: dict[tuple[int, int], _StaticVisionPrefix] = {}
        self.compiled_prefixes: dict[
            tuple[int, int], Callable[[torch.Tensor], torch.Tensor]
        ] = {}
        self.prefix_cache_dirs: dict[tuple[int, int], Path] = {}
        self.prefix_first_call_wall_s: dict[str, float] = {}
        self.prefix_call_count = 0

        from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

        config = CompilerConfig()
        config.mode.value = "max-autotune"
        cache_compile, import_path = import_torchair_cache_compile()
        self.static_prefix_compile_api = import_path
        source_hash = _source_hash()
        cache_root = runner.compile_cache_dir
        if cache_root is None:
            raise ValueError("per-shape vision compilation requires compile_cache_dir")
        for width, height in self.shapes:
            module = _new_static_prefix_module(
                runner,
                input_height=height,
                input_width=width,
            ).eval()
            cache_dir = cache_root / (
                f"vision_static_prefix_{width}x{height}_"
                f"{runner.dtype_name}_src{source_hash}"
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.prefix_modules[(width, height)] = module
            self.prefix_cache_dirs[(width, height)] = cache_dir
            self.compiled_prefixes[(width, height)] = cache_compile(
                module.forward,
                config=config,
                dynamic=False,
                cache_dir=str(cache_dir),
                ge_cache=True,
                fullgraph=True,
            )

    def _run_prefix(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        height = int(pixel_values.shape[2])
        width = int(pixel_values.shape[3])
        shape = (width, height)
        compiled = self.compiled_prefixes.get(shape)
        if compiled is None:
            raise ValueError(
                f"no compiled UniRec prefix graph for {width}x{height}; "
                f"registry contains {len(self.shapes)} shapes"
            )
        self.prefix_call_count += 1
        key = f"{width}x{height}"
        if key not in self.prefix_first_call_wall_s:
            started = time.perf_counter()
            output = compiled(pixel_values)
            synchronize_device(self.runner.device)
            self.prefix_first_call_wall_s[key] = time.perf_counter() - started
        else:
            output = compiled(pixel_values)
        return output, height // 16, width // 16

    def warmup_all_prefix_graphs(self, *, passes: int = 1) -> dict[str, Any]:
        if passes < 1:
            raise ValueError("static prefix warmup passes must be positive")
        device = torch.device(self.runner.device)
        report: dict[str, Any] = {
            f"{width}x{height}": {
                "pass_wall_s": [],
                "cache_dir": str(self.prefix_cache_dirs[(width, height)]),
            }
            for width, height in self.shapes
        }
        inputs: dict[tuple[int, int], torch.Tensor] = {}
        for width, height in self.shapes:
            # Production preprocessing creates the device input before the
            # compiled call enters inference mode. Preserve that dispatch-key
            # contract so a cache warmed here loads in the page worker.
            with torch.inference_mode(False):
                inputs[(width, height)] = torch.zeros(
                    (1, 3, height, width),
                    dtype=self.runner.dtype,
                    device=device,
                )
        # Sweep the complete registry once per pass. The second pass therefore
        # revisits each graph after every other shape has run.
        for _ in range(passes):
            for width, height in self.shapes:
                pixel_values = inputs[(width, height)]
                with torch.inference_mode():
                    started = time.perf_counter()
                    self.compiled_prefixes[(width, height)](pixel_values)
                    synchronize_device(self.runner.device)
                    report[f"{width}x{height}"]["pass_wall_s"].append(
                        time.perf_counter() - started
                    )
        return report

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        result.update(
            {
                "spatial_execution": "compiled_per_shape_prefix",
                "static_prefix_shape_count": len(self.shapes),
                "static_prefix_call_count": self.prefix_call_count,
                "static_prefix_first_call_wall_s": dict(
                    self.prefix_first_call_wall_s
                ),
                "static_prefix_compile_api": self.static_prefix_compile_api,
            }
        )
        return result

    def reset_stats(self) -> None:
        super().reset_stats()
        self.prefix_call_count = 0
