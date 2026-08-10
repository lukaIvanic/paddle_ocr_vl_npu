from __future__ import annotations

import os
from pathlib import Path

import torch_npu
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension

ROOT = Path(__file__).resolve().parent
TORCH_NPU_ROOT = Path(torch_npu.__file__).resolve().parent

setup(
    name="paddle_decode_kv_scatter_query_eager",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        NpuExtension(
            name="paddle_decode_kv_scatter_query_eager._C",
            sources=[str(ROOT / "csrc/paddle_decode_kv_scatter_query_eager.cpp")],
            extra_compile_args=[
                "-O3", "-std=c++17",
                f"-I{TORCH_NPU_ROOT / 'include/third_party/acl/inc'}",
                f"-I{TORCH_NPU_ROOT / 'include/third_party/op-plugin'}",
                f"-I{TORCH_NPU_ROOT / 'include/third_party/op-plugin/op_plugin/include'}",
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=os.environ.get("USE_NINJA") == "1"
        )
    },
)
