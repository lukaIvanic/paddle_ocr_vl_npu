"""Build the minimal Ascend/vision DVPP surface used by the layout lab.

Run this file from an Ascend/vision source checkout.  It intentionally compiles
only the official JPEG decode, resize, image-to-tensor, and initialization
sources; it does not install or monkey-patch the full torchvision_npu package.
"""

from pathlib import Path

from setuptools import setup
from torch_npu.utils.cpp_extension import NpuExtension
from torch.utils.cpp_extension import BuildExtension


ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "torchvision_npu" / "csrc"
SOURCES = [
    CSRC / "dvpp_init.cpp",
    CSRC / "ops" / "npu_decode_jpeg.cpp",
    CSRC / "ops" / "npu_resize.cpp",
    CSRC / "ops" / "npu_img_to_tensor.cpp",
    CSRC / "ops" / "npu" / "npu_decode_jpeg_kernel.cpp",
    CSRC / "ops" / "npu" / "npu_resize_kernel.cpp",
    CSRC / "ops" / "npu" / "npu_img_to_tensor_kernel.cpp",
]
missing = [str(path) for path in SOURCES if not path.is_file()]
if missing:
    raise FileNotFoundError(
        "run setup_minimal.py from an Ascend/vision source checkout; "
        f"missing sources: {missing}"
    )

setup(
    name="paddle_layout_dvpp_probe",
    version="0.0.0",
    ext_modules=[
        NpuExtension(
            "paddle_layout_dvpp_probe_ops",
            sources=[str(path) for path in SOURCES],
            include_dirs=[str(CSRC)],
            extra_compile_args=[
                "-O3",
                "-fPIC",
                "-Wno-deprecated-declarations",
                "-Wno-sign-compare",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
