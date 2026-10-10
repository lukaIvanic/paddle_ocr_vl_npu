"""Direct Python call to the upstream AddLayerNormQuantV2 ACLNN API on 910B.

Source the locally built operator package's set_env.bash first, then run:
  python test_add_layer_norm_quant_v2.py --op-api /path/to/libcust_opapi.so --output /path/to/run
No PyTorch extension, TorchAir, or model integration is involved.
"""
import argparse
import ctypes as C
import csv
import hashlib
import json
import os
from pathlib import Path

import torch
import torch_npu


class V2:
    def __init__(self, library):
        self.base = C.CDLL("libnnopbase.so", mode=C.RTLD_GLOBAL)
        self.api = C.CDLL(str(library.resolve()), mode=C.RTLD_GLOBAL)
        ptr, i64, u64 = C.c_void_p, C.c_int64, C.c_uint64
        self.base.aclCreateTensor.argtypes = [C.POINTER(i64), u64, C.c_int, C.POINTER(i64),
                                             i64, C.c_int, C.POINTER(i64), u64, ptr]
        self.base.aclCreateTensor.restype = ptr
        self.base.aclDestroyTensor.argtypes = [ptr]
        self.prepare = self.api.aclnnAddLayerNormQuantV2GetWorkspaceSize
        self.prepare.argtypes = [ptr] * 9 + [C.c_char_p, C.c_double, C.c_bool, C.c_bool] + [ptr] * 6 + [C.POINTER(u64), C.POINTER(ptr)]
        self.prepare.restype = C.c_int
        self.launch = self.api.aclnnAddLayerNormQuantV2
        self.launch.argtypes = [ptr, u64, ptr, ptr]
        self.launch.restype = C.c_int

    def tensor(self, tensor):
        if tensor is None:
            return None
        assert tensor.device.type == "npu" and tensor.is_contiguous() and tensor.storage_offset() == 0
        dims = (C.c_int64 * tensor.ndim)(*tensor.shape)
        strides = (C.c_int64 * tensor.ndim)(*tensor.stride())
        dtype = {torch.float32: 0, torch.float16: 1, torch.int8: 2}[tensor.dtype]
        handle = self.base.aclCreateTensor(dims, tensor.ndim, dtype, strides, 0, 2,
                                           dims, tensor.ndim, tensor.data_ptr())  # ACL_FORMAT_ND
        if not handle:
            raise RuntimeError("aclCreateTensor failed")
        return handle

    def __call__(self, x1, x2, gamma, beta, scale, bias=None, eps=1e-5):
        # Static, one quantized output, no pre-normalization sum output.
        quant = torch.empty_like(x1, dtype=torch.int8)
        norm = torch.empty_like(x1)
        unused = [torch.empty(1, device=x1.device, dtype=torch.int8), torch.empty_like(x1),
                  torch.empty(1, device=x1.device), torch.empty(1, device=x1.device)]
        tensors = [x1, x2, gamma, beta, bias, scale, None, None, None,
                   quant, unused[0], unused[1], norm, unused[2], unused[3]]
        handles = [self.tensor(t) for t in tensors]
        workspace_bytes, executor = C.c_uint64(), C.c_void_p()
        try:
            status = self.prepare(*handles[:9], b"static", eps, False, False, *handles[9:],
                                  C.byref(workspace_bytes), C.byref(executor))
            if status:
                raise RuntimeError(f"V2 GetWorkspaceSize failed: {status}")
            workspace = torch.empty(workspace_bytes.value, device=x1.device, dtype=torch.uint8)
            status = self.launch(workspace.data_ptr(), workspace_bytes.value, executor,
                                 torch.npu.current_stream().npu_stream)
            if status:
                raise RuntimeError(f"V2 launch failed: {status}")
            torch.npu.synchronize()  # Keep buffers/ACL descriptors alive until completion.
        finally:
            for handle in handles:
                if handle:
                    self.base.aclDestroyTensor(handle)
        return norm, quant


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op-api", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.npu.set_device(0)
    chip = torch.npu.get_device_name(0)
    if "910B" not in chip.upper():
        raise RuntimeError(f"Expected 910B, got {chip}")
    op = V2(args.op_api)
    torch.manual_seed(17)
    shape = (args.rows, args.width)
    x1, x2 = [torch.randn(shape).half().npu() for _ in range(2)]
    gamma = (1 + 0.1 * torch.randn(1, args.width)).half().npu()
    beta, bias = [(.1 * torch.randn(1, args.width)).half().npu() for _ in range(2)]
    scale = torch.tensor([16.0], dtype=torch.float16, device="npu")  # V2 uses multiplication.
    result = {"device": chip, "physical_npu": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
              "library": str(args.op_api), "library_sha256": hashlib.sha256(args.op_api.read_bytes()).hexdigest(),
              "shape": shape, "cases": [], "passed": False}
    for enabled in (False, True):
        selected_bias = bias if enabled else None
        norm, quant = op(x1, x2, gamma, beta, scale, selected_bias)
        summed = x1.float().cpu() + x2.float().cpu()
        if enabled:
            summed += bias.float().cpu()
        reference = torch.nn.functional.layer_norm(summed, (args.width,), gamma.float().cpu().flatten(),
                                                    beta.float().cpu().flatten(), 1e-5)
        norm_cpu, quant_cpu = norm.float().cpu(), quant.cpu().int()
        expected_quant = (reference * float(scale.cpu())).round().clamp(-128, 127).int()
        error = (quant_cpu - expected_quant).abs()
        torch.testing.assert_close(norm_cpu, reference, atol=0.004, rtol=0.002)
        assert int(error.max()) <= 1, "Quantized output differs by more than one INT8 level"
        row = {"bias": enabled, "norm_max_abs": float((norm_cpu-reference).abs().max()),
               "int8_max_abs": int(error.max()), "int8_exact_fraction": float((error == 0).float().mean())}
        result["cases"].append(row)
        print(json.dumps(row), flush=True)

    import torch_npu.profiler as prof
    with prof.profile(activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
                      on_trace_ready=prof.tensorboard_trace_handler(str(args.output / "profile")),
                      record_shapes=True,
                      experimental_config=prof._ExperimentalConfig(profiler_level=prof.ProfilerLevel.Level1)):
        for _ in range(3):
            op(x1, x2, gamma, beta, scale, bias)
    kernels = list((args.output / "profile").glob("**/kernel_details.csv"))
    if not kernels:
        raise RuntimeError("Profiler did not produce device kernel evidence")
    with kernels[0].open() as handle:
        rows = list(csv.DictReader(handle))
    result["kernels"] = [{k: r[k] for k in ("Name", "Type", "Duration(us)", "Input Shapes")} for r in rows]
    result["passed"] = True
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "kernel_rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
