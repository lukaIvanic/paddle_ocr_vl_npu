"""Static-shape compiled UniRec vision prefix and suffix execution."""

from __future__ import annotations

import hashlib
import inspect
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from modeling_optimized_unirec import (
    OptimizedUniRecRunner,
    import_torchair_cache_compile,
    synchronize_device,
)
from vision_atlas import ATLAS_STAGE, UniRecVisionAtlasRuntime


class _StaticVisionPrefix(nn.Module):
    def __init__(self, runner: OptimizedUniRecRunner, *, input_height: int, input_width: int) -> None:
        super().__init__()
        vision = runner.model.encoder.vision_encoder
        self.patch_embed = vision.patch_embed
        self.pos_drop = vision.pos_drop
        self.stage0 = vision.layers[0]
        self.stage1 = vision.layers[1]
        self.patch_height = input_height // 4
        self.patch_width = input_width // 4

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(pixel_values)
        x = self.pos_drop(x.flatten(2).transpose(1, 2))
        x, height, width = self.stage0(x, self.patch_height, self.patch_width)
        x, _, _ = self.stage1(x, height, width)
        return x


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
        self.prefix_module = _StaticVisionPrefix(
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
