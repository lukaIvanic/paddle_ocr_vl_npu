"""Build/probe the imported WKV-7 NPU operator before using it in a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch


class WkvStep(torch.nn.Module):
    def forward(self, k, v, w, r, a, b, hi):
        return torch.ops.rwkv_reference.wkv7.default(k, v, w, r, a, b, hi)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_bridge(root, build):
    import torch
    import torch_npu
    from torch.utils.cpp_extension import load
    package = Path(torch_npu.__file__).parent
    extension = build / "bridge"
    extension.mkdir(exist_ok=True)
    return load(name="rwkv_reference_wkv7_bridge", sources=[str(root / "bridge.cpp")],
                build_directory=str(extension), is_python_module=False,
                extra_include_paths=[str(package / "include"),
                    str(package / "include/third_party/acl/inc"),
                    str(package / "include/third_party/op-plugin"),
                    str(package / "include/third_party/op-plugin/op_plugin/include")],
                extra_cflags=["-O3", "-std=c++17"],
                extra_ldflags=[f"-L{package / 'lib'}", "-ltorch_npu"], verbose=True)


def reference(inputs):
    """Independent CPU FP64 recurrence in the C anchor's [key,value] orientation."""
    import torch
    k, v, w, r, a, b, hi = [x.double() for x in inputs]
    state = hi.transpose(-1, -2).contiguous().clone()
    rows = []
    for t in range(k.shape[2]):
        sa = (state * a[:, :, t, :, None]).sum(-2)
        state = (state * w[:, :, t].exp().unsqueeze(-1)
                 + b[:, :, t, :, None] * sa.unsqueeze(-2)
                 + k[:, :, t, :, None] * v[:, :, t, None, :])
        rows.append((state * r[:, :, t, :, None]).sum(-2))
    return torch.stack(rows, dim=2).float(), state.transpose(-1, -2).contiguous().float()


def make_inputs(batch, length, nonzero, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    shape = (batch, 12, length, 64)
    k, v, z, r, kk, rate = [torch.randn(shape, generator=g) for _ in range(6)]
    kk = torch.nn.functional.normalize(kk, dim=-1)
    state = torch.randn((batch, 12, 64, 64), generator=g) * .01 if nonzero else torch.zeros((batch, 12, 64, 64))
    # w is log decay: the NPU kernel applies exp(w), unlike exp(-exp(w_raw)).
    return [k * .1, v * .1, -torch.sigmoid(z) * .6065306597126334,
            r * .1, -kk, kk * torch.sigmoid(rate), state]


def compare(actual, expected):
    import torch
    delta = (actual - expected).abs()
    return {"allclose": bool(torch.allclose(actual, expected, atol=2e-5, rtol=2e-4)),
            "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
            "sha256": hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--allow-shared-device", action="store_true",
                   help="Explicitly authorized shared-device correctness run; skip timing")
    p.add_argument("--lengths", default="1,47,48,49,64,128,512,976")
    p.add_argument("--batch-sizes", default="1,2")
    p.add_argument("--backend", choices=("raw_eager", "torchair"), default="raw_eager")
    p.add_argument("--repeats", type=int, default=20)
    args = p.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    batches = [int(x) for x in args.batch_sizes.split(",")]
    if (not 1 <= args.repeats <= 100 or any(not 1 <= x <= 2048 for x in lengths)
            or any(x not in (1, 2) for x in batches)):
        p.error("Use lengths 1..2048 and repeats 1..100")
    if args.backend == "torchair" and (len(lengths) != 1 or len(batches) != 1):
        p.error("TorchAir probes use exactly one batch/length shape per process")
    root = Path(__file__).parent / "wkv7_npu"
    build = args.build_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"hostname": platform.node(), "machine": platform.machine(),
              "source_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
              "script_sha256": digest(__file__), "physical_npu": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
              "backend": args.backend, "torchair_used": args.backend == "torchair", "python_stock_fallback": False,
              "same_name_override": False, "shared_device": args.allow_shared_device,
              "atol": 2e-5, "rtol": 2e-4, "cases": [],
              "all_checks_passed": False, "inference_executed": False}
    try:
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        os.environ["MAX_JOBS"] = "2"
        import torch
        import torch_npu
        torch.set_num_threads(4)
        report.update(torch=torch.__version__, torch_npu=torch_npu.__version__)
        load_bridge(root, build)
        report["bridge_sha256"] = digest(build / "bridge/rwkv_reference_wkv7_bridge.so")
        table = torch._C._dispatch_dump_table("rwkv_reference::wkv7")
        assert "PrivateUse1" in table and "Meta" in table
        report["dispatcher"] = table
        if args.build_only:
            report["status"] = "bridge_built_no_inference"
            return
        if not report["physical_npu"]:
            raise RuntimeError("source npu-setup first; no physical NPU was selected")
        status = subprocess.check_output(["/usr/local/bin/npu-status"], text=True)
        report["device_status_before_inference"] = status
        selected = next((line for line in status.splitlines()
                         if line.startswith(f"NPU {report['physical_npu']}: ")), "")
        if "Health=OK" not in selected:
            raise RuntimeError("Selected NPU does not report OK health")
        if not args.allow_shared_device and ": free " not in selected:
            raise RuntimeError("Selected device is no longer healthy and free; rerun npu-setup")
        torch.npu.set_device(0)
        free, total = torch.npu.mem_get_info()
        report["hbm_before_inference"] = {"free_bytes": free, "total_bytes": total}
        if free < 1024 ** 3:
            raise RuntimeError("Less than 1 GiB free HBM for the recurrence probe")
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = False
        report["device"] = torch.npu.get_device_name(0)
        # Stock control establishes that this selected device can execute work.
        control = torch.ones((64, 64), device="npu")
        assert torch.equal((control @ control).cpu(), torch.full((64, 64), 64.))
        eager_op = torch.ops.rwkv_reference.wkv7.default
        op = eager_op
        if args.backend == "torchair":
            import torchair
            from torchair.configs.compiler_config import CompilerConfig
            from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
            from torchair import ge

            @register_fx_node_ge_converter(eager_op)
            def convert(k, v, w, r, a, b, hi, meta_outputs=None):
                return ge.custom_op("RwkvReferenceWkv7",
                    inputs={"k": k, "v": v, "w": w, "r": r, "a": a, "b": b, "hi": hi},
                    attrs={}, outputs=["o", "ho"])

            report["graph_cache"] = str(output / "graph_cache")
            op = torchair.inference.cache_compile(WkvStep().forward, config=CompilerConfig(),
                    dynamic=False, cache_dir=report["graph_cache"], ge_cache=True)
        for batch in batches:
            for length in lengths:
                for nonzero in (False, True):
                    cpu = make_inputs(batch, length, nonzero, 20261006 + length)
                    expected = reference(cpu)
                    device = [x.to("npu") for x in cpu]
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    out, state = op(*device)
                    torch.npu.synchronize()
                    first_seconds = time.perf_counter() - start
                    report["inference_executed"] = True
                    got = (out.cpu(), state.cpu())
                    comparisons = [compare(x, y) for x, y in zip(got, expected)]
                    assert all(x["allclose"] for x in comparisons), comparisons
                    if args.backend == "torchair":
                        direct = tuple(x.cpu() for x in eager_op(*device))
                        assert all(compare(x, y)["allclose"] for x, y in zip(got, direct)), "Compiled vs eager"
                    assert all(torch.equal(x.cpu(), y) for x, y in zip(device, cpu)), "Input mutation"
                    repeated = tuple(x.cpu() for x in op(*device))
                    assert all(torch.equal(x, y) for x, y in zip(got, repeated)), "Repeat-call mismatch"
                    if length > 1:
                        split = length // 2
                        left = [x[:, :, :split].contiguous() for x in device[:6]] + [device[6]]
                        lo, ls = eager_op(*left)
                        right = [x[:, :, split:].contiguous() for x in device[:6]] + [ls]
                        ro, rs = eager_op(*right)
                        assert compare(torch.cat([lo, ro], dim=2).cpu(), got[0])["allclose"], "Split output"
                        assert compare(rs.cpu(), got[1])["allclose"], "Split final state"
                    steady = None
                    if not args.allow_shared_device:
                        torch.npu.synchronize()
                        start = time.perf_counter()
                        for _ in range(args.repeats):
                            op(*device)
                        torch.npu.synchronize()
                        steady = (time.perf_counter() - start) / args.repeats
                    row = {"batch": batch, "heads": 12, "length": length, "nonzero_initial_state": nonzero,
                           "output": comparisons[0], "state": comparisons[1],
                           "first_call_seconds": None if args.allow_shared_device else first_seconds,
                           "steady_host_seconds": steady, "repeat_bitwise_equal": True,
                           "split_continuation_passed": True if length > 1 else None, "inputs_unchanged": True}
                    report["cases"].append(row)
                    print(json.dumps(row), flush=True)
        report["all_checks_passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
