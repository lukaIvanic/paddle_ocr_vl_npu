from __future__ import annotations

import os
from pathlib import Path

import torch_npu
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension


ROOT = Path(__file__).resolve().parent
TORCH_NPU_ROOT = Path(torch_npu.__file__).resolve().parent
ASCEND_HOME = Path(
    os.environ.get("ASCEND_HOME_PATH")
    or os.environ.get("ASCEND_TOOLKIT_HOME")
    or "/usr/local/Ascend/ascend-toolkit/latest"
).resolve()
if not (ASCEND_HOME / "include/register/op_impl_registry.h").is_file():
    raise RuntimeError(
        "CANN host headers were not found under "
        f"{ASCEND_HOME}; source the matching CANN environment first"
    )

setup(
    name="unirec_layout_msda_aclnn",
    version="0.1.0",
    ext_modules=[
        NpuExtension(
            name="unirec_layout_msda_aclnn._C",
            sources=[str(ROOT / "csrc/layout_msda_aclnn.cpp")],
            extra_compile_args=[
                "-O3",
                "-std=c++17",
                f"-I{TORCH_NPU_ROOT / 'include/third_party/acl/inc'}",
                f"-I{TORCH_NPU_ROOT / 'include/third_party/op-plugin'}",
                f"-I{TORCH_NPU_ROOT / 'include/third_party/op-plugin/op_plugin/include'}",
                f"-I{ASCEND_HOME / 'include'}",
            ],
            extra_link_args=[
                f"-L{ASCEND_HOME / 'lib64'}",
                "-lopp_registry",
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=os.environ.get("USE_NINJA") == "1"
        )
    },
)
