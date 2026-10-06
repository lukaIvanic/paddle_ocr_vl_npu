#!/usr/bin/env python3
"""Standalone FP16 MinerU-shaped KV layout probe; only workers import torch/NPU.

No model, vLLM, TorchAir, or repository imports. Every operation/layout/format
case runs in a fresh process, with a timeout, so a rejected contract or a 310P
masked-GQA stall cannot contaminate the remaining cases.
"""
import argparse
import hashlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback


SOURCE_COMMIT = "80610e4438dba05011b05f89fc45d91e96992671"
LAYOUTS = {"increfa": ("dense",), "fia": ("pages", "blocked5"),
           "fia2": ("pages", "blocked5"), "paged": ("pages4", "blocked4")}


def block_mapping(batch, capacity, block_size, seed):
    pages_per_row = capacity // block_size
    order = list(range(batch * pages_per_row))
    random.Random(seed).shuffle(order)
    return [order[b * pages_per_row:(b + 1) * pages_per_row] for b in range(batch)]


def lengths_for(batch, context, pattern):
    # Include full context, a short row, and a non-block-aligned tail.
    choices = [context, max(1, context - 1), max(1, context // 2 + 3), 1]
    return [context if pattern == "uniform" else choices[b % 4] for b in range(batch)]


def layout_shape(layout, batch, capacity, heads, dim, block):
    n = batch * capacity // block
    return {"dense": (batch, heads, capacity, dim),
            "pages": (n, block, heads * dim),
            "pages4": (n, block, heads, dim),
            "blocked4": (n, heads * dim // 16, block, 16),
            "blocked5": (n, heads, dim // 16, block, 16)}[layout]


def pack_cache(t, torch, mapping, layout, block):
    if layout == "dense":
        return t.contiguous()
    batch, heads, capacity, dim = t.shape
    source = t.permute(0, 2, 1, 3).contiguous().view(-1, block, heads, dim)
    pages = torch.empty_like(source)
    indices = torch.tensor(mapping, dtype=torch.long).flatten()
    pages.index_copy_(0, indices, source)
    if layout in ("pages", "pages4"):
        return pages.view(layout_shape(layout, batch, capacity, heads, dim, block))
    return pages.view(-1, block, heads, dim // 16, 16).permute(
        0, 2, 3, 1, 4).contiguous().view(
            layout_shape(layout, batch, capacity, heads, dim, block))


def unpack_cache(t, torch, mapping, layout, batch, capacity, heads, dim, block):
    if layout == "dense":
        return t.contiguous()
    if layout in ("pages", "pages4"):
        pages = t.reshape(-1, block, heads, dim)
    else:
        pages = t.reshape(-1, heads, dim // 16, block, 16).permute(
            0, 3, 1, 2, 4).contiguous().reshape(-1, block, heads, dim)
    indices = torch.tensor(mapping, dtype=torch.long).flatten()
    return pages.index_select(0, indices).view(batch, capacity, heads, dim).permute(
        0, 2, 1, 3).contiguous()


def digest(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chip", choices=("910B", "310P"))
    p.add_argument("--operators", default="increfa,fia,fia2")
    p.add_argument("--batches", default="1,16")
    p.add_argument("--contexts", default="768")
    p.add_argument("--patterns", default="ragged")
    p.add_argument("--formats", default="2,29", help="requested ACL descriptors; select 2 for native-only timing controls")
    p.add_argument("--paged-length-device", choices=("npu", "cpu"), default="npu",
                   help="npu reproduces pinned 310P dispatch; cpu is a labeled 910B ATB compatibility control")
    p.add_argument("--block-size", type=int, choices=(64, 128), default=128)
    p.add_argument("--cache-length", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--calls-per-sample", type=int, default=10)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--seed", type=int, default=310)
    p.add_argument("--output", type=Path)
    p.add_argument("--atol", type=float, default=0.003)
    p.add_argument("--rtol", type=float, default=0.003)
    p.add_argument("--cpu-self-test", action="store_true")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    return p


def case_matrix(args):
    operators = args.operators.split(",")
    if any(op not in LAYOUTS for op in operators):
        raise ValueError(f"operators must be selected from {tuple(LAYOUTS)}")
    patterns = args.patterns.split(",")
    if any(p not in ("uniform", "ragged") for p in patterns):
        raise ValueError("patterns must be uniform or ragged")
    formats = list(map(int, args.formats.split(",")))
    if not formats or len(set(formats)) != len(formats) or any(f not in (2, 29) for f in formats):
        raise ValueError("formats must be a unique subset of 2,29")
    for op, b, s, pattern in itertools.product(operators,
            map(int, args.batches.split(",")), map(int, args.contexts.split(",")), patterns):
        if min(b, s) < 1:
            raise ValueError("batch and context must be positive")
        capacity = args.cache_length
        if capacity % args.block_size or s > capacity:
            raise ValueError("cache length must be block-aligned and at least the largest context")
        for layout, fmt in itertools.product(LAYOUTS[op], formats):
            fills = ("packed", "writer") if op == "paged" else ("packed",)
            for fill in fills:
                case = dict(operator=op, batch=b, context=s, pattern=pattern,
                            capacity=capacity, layout=layout, format=fmt, fill=fill)
                if op == "paged":
                    case["length_device"] = args.paged_length_device
                yield case


def tensor_info(t, torch_npu):
    return dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                format=int(torch_npu.get_npu_format(t)), logical_bytes=t.numel() * t.element_size())


def callable_info(fn):
    try:
        signature = str(inspect.signature(fn))
    except (TypeError, ValueError):
        signature = "unavailable"
    try:
        file = inspect.getfile(fn)
    except TypeError:
        file = "native binding"
    return dict(module=getattr(fn, "__module__", None), file=file, signature=signature)


def run_worker(args, case):
    import torch
    import torch_npu

    torch.npu.config.allow_internal_format = True
    if not torch.npu.is_available():
        raise RuntimeError("NPU unavailable; this probe has no CPU inference fallback")
    torch.npu.set_device("npu:0")  # npu-setup chooses the physical device externally.
    device = "npu:0"
    b, capacity, s = case["batch"], case["capacity"], case["context"]
    hq, hk, d, block = 14, 2, 64, args.block_size
    op, layout, fmt = case["operator"], case["layout"], case["format"]
    mapping = block_mapping(b, capacity, block, args.seed)
    lengths = lengths_for(b, s, case["pattern"])
    result = dict(case=case, status="initializing", source_commit=SOURCE_COMMIT,
                  requested_chip=args.chip, device_name=torch.npu.get_device_name(0),
                  torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
                  visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  cann_home=os.environ.get("ASCEND_HOME_PATH"), hostname=os.uname().nodename,
                  lengths=lengths, block_table=mapping, heads=dict(query=hq, kv=hk, dim=d),
                  internal_format_requested=True)
    # Save metadata even if an op hangs before returning to Python.
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    actual_chip = result["device_name"].upper()
    if ((args.chip == "310P" and "310P" not in actual_chip) or
            (args.chip == "910B" and "910B" not in actual_chip)):
        raise RuntimeError(f"requested {args.chip}, observed {actual_chip}; refusing mislabeled run")
    g = torch.Generator().manual_seed(args.seed)
    qcpu = torch.randn((b, hq, 1, d), generator=g).half()
    kcpu = torch.randn((b, hk, capacity, d), generator=g).half()
    vcpu = torch.randn((b, hk, capacity, d), generator=g).half()
    # Poison masked capacity to catch using physical capacity as logical length.
    for row, length in enumerate(lengths):
        kcpu[row, :, length:] = 7
        vcpu[row, :, length:] = 13
    result["fixture_sha256"] = {"q": digest(qcpu), "k": digest(kcpu), "v": digest(vcpu)}
    refs = []
    for row, length in enumerate(lengths):
        kr = kcpu[row, :, :length].float().repeat_interleave(hq // hk, dim=0)
        vr = vcpu[row, :, :length].float().repeat_interleave(hq // hk, dim=0)
        probs = torch.softmax(qcpu[row].float() @ kr.transpose(-1, -2) / math.sqrt(d), dim=-1)
        refs.append(probs @ vr)
    ref = torch.stack(refs)
    table = torch.tensor(mapping, dtype=torch.int32, device=device)
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    q = qcpu.to(device)

    started = time.perf_counter()
    caches = []
    for host in (kcpu, vcpu):
        packed = pack_cache(host, torch, mapping, layout, block)
        cache = torch_npu.empty_with_format(size=tuple(packed.shape), dtype=torch.float16,
                                           device=device, acl_format=fmt)
        # Both variants use the same allocation API; format 29 is never simulated.
        if case["fill"] == "packed":
            cache.copy_(packed.to(device))
        caches.append(cache)
    torch.npu.synchronize()
    result["cache_allocation_and_fill_s"] = time.perf_counter() - started
    kc, vc = caches
    result["cache_before_attention"] = {"key": tensor_info(kc, torch_npu),
                                         "value": tensor_info(vc, torch_npu)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    observed_formats = [int(torch_npu.get_npu_format(t)) for t in caches]
    # torch-npu normalizes ordinary rank-4 ND allocations to native NCHW
    # descriptor 0, preserving contiguous element order. Record that descriptor
    # and verify every value in the subsequent KV roundtrip. NZ stays strict.
    acceptable_formats = (0, 2) if fmt == 2 else (29,)
    if any(value not in acceptable_formats for value in observed_formats):
        result["status"] = "format_unavailable"
        result["reason"] = "requested descriptor was not retained; no ND-as-NZ fallback"
        return result
    if case["fill"] == "packed":
        restored = [unpack_cache(torch_npu.npu_format_cast(t, 2).cpu(), torch, mapping, layout,
                               b, capacity, hk, d, block) for t in caches]
        result["packed_roundtrip_exact"] = all(torch.equal(a, z) for a, z in zip(restored, (kcpu, vcpu)))
        if not result["packed_roundtrip_exact"]:
            result["status"] = "validation_failed"
            result["reason"] = "allocation/copy/format conversion changed logical KV values"
            return result

    if case["fill"] == "writer":
        result["writer_binding"] = callable_info(torch_npu._npu_reshape_and_cache)
        result["stage"] = "cache_writer"
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        # The exact BaseDeviceAdaptor cache writer used on 310P. Populate every
        # physical slot, including poisoned tail, rather than trusting our packer.
        keys = kcpu.permute(0, 2, 1, 3).reshape(-1, hk, d).contiguous().to(device)
        values = vcpu.permute(0, 2, 1, 3).reshape(-1, hk, d).contiguous().to(device)
        slots = [mapping[row][pos // block] * block + pos % block
                 for row in range(b) for pos in range(capacity)]
        slots = torch.tensor(slots, dtype=torch.int32, device=device)
        kc.zero_()
        vc.zero_()
        torch.npu.synchronize()
        started = time.perf_counter()
        torch_npu._npu_reshape_and_cache(key=keys, value=values, key_cache=kc,
                                        value_cache=vc, slot_indices=slots)
        torch.npu.synchronize()
        result["writer_fill_s"] = time.perf_counter() - started
        # Some private kernels interpret the physical NZ storage directly.
        # Report logical roundtrip separately; attention must still match FP32.
        restored = [unpack_cache(torch_npu.npu_format_cast(t, 2).cpu(), torch, mapping,
                                layout, b, capacity, hk, d, block) for t in caches]
        result["writer_logical_roundtrip_exact"] = all(
            torch.equal(a, z) for a, z in zip(restored, (kcpu, vcpu)))

    result["cache_for_attention"] = {"key": tensor_info(kc, torch_npu),
                                    "value": tensor_info(vc, torch_npu)}
    if any(int(torch_npu.get_npu_format(t)) not in acceptable_formats for t in caches):
        result["status"] = "format_unavailable"
        result["reason"] = "cache writer changed the requested storage descriptor; no timing"
        return result

    if op == "increfa":
        result["operation_binding"] = callable_info(torch_npu.npu_incre_flash_attention)
        mask = torch.arange(capacity, device=device).view(1, 1, 1, -1) >= lens.view(b, 1, 1, 1)
        tile = (40960 // (hq // hk) // 4 // 128) * 128  # 1408, production workaround.
        sentinel = (lens.view(b, 1, 1, 1).remainder(tile) == 0) & (
            torch.arange(capacity, device=device).view(1, 1, 1, -1) == lens.view(b, 1, 1, 1))
        mask = mask & ~sentinel
        pse = torch.zeros((b, hq, 1, capacity), dtype=torch.float16, device=device).masked_fill(
            sentinel.expand(b, hq, 1, capacity), torch.finfo(torch.float16).min)
        result["contract"] = "BNSD, bool future mask, actual_seq_lengths=None, production PSE sentinel"
        def call():
            return torch_npu.npu_incre_flash_attention(q, kc, vc, atten_mask=mask,
                actual_seq_lengths=None, num_heads=hq, num_key_value_heads=hk,
                input_layout="BNSD", scale_value=1 / math.sqrt(d), pse_shift=pse)
    elif op in ("fia", "fia2"):
        qbsh = q.transpose(1, 2).contiguous().view(b, 1, hq * d)
        result["contract"] = "BSH one-token query, paged cache, lengths exclude poisoned tail, no causal mask"
        if op == "fia":
            result["operation_binding"] = callable_info(torch_npu.npu_fused_infer_attention_score)
            def call():
                return torch_npu.npu_fused_infer_attention_score(qbsh, kc, vc,
                    num_heads=hq, num_key_value_heads=hk, input_layout="BSH",
                    scale=1 / math.sqrt(d), actual_seq_lengths=[1] * b,
                    actual_seq_lengths_kv=lengths, block_table=table,
                    block_size=block, sparse_mode=0, inner_precise=1)[0]
        else:
            result["operation_binding"] = callable_info(torch_npu.npu_fused_infer_attention_score_v2)
            def call():
                return torch_npu.npu_fused_infer_attention_score_v2(qbsh, kc, vc,
                    num_query_heads=hq, num_key_value_heads=hk, input_layout="BSH",
                    softmax_scale=1 / math.sqrt(d), actual_seq_qlen=[1] * b,
                    actual_seq_kvlen=lengths, block_table=table,
                    block_size=block, sparse_mode=0, inner_precise=1)[0]
    else:
        result["operation_binding"] = callable_info(torch_npu._npu_paged_attention)
        q3 = q.squeeze(2).contiguous()
        output = torch.empty_like(q3)
        length_device = case.get("length_device", "npu")
        attention_lens = lens if length_device == "npu" else torch.tensor(lengths, dtype=torch.int32)
        result["context_lens_device"] = str(attention_lens.device)
        result["contract"] = ("exact pinned 310P forward_paged_attention call with NPU lengths"
                              if length_device == "npu" else
                              "910B ATB compatibility control: same private op, CPU int32 lengths")
        def call():
            torch_npu._npu_paged_attention(query=q3, key_cache=kc, value_cache=vc,
                num_kv_heads=hk, num_heads=hq, scale_value=1 / math.sqrt(d),
                block_table=table, context_lens=attention_lens, out=output)
            return output

    result["stage"] = "attention_validation"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    observed = call()
    torch.npu.synchronize()
    result["cache_after_validation_call"] = {"key": tensor_info(kc, torch_npu),
                                             "value": tensor_info(vc, torch_npu)}
    if op == "increfa":
        observed = observed.cpu().float()
    else:
        observed = observed.cpu().float().reshape(b, 1, hq, d).transpose(1, 2)
    diff = observed - ref
    result["correctness"] = dict(allclose=bool(torch.allclose(observed, ref, atol=args.atol, rtol=args.rtol)),
        finite=bool(torch.isfinite(observed).all()), max_abs=float(diff.abs().max()),
        relative_l2=float(diff.norm() / ref.norm().clamp_min(1e-12)), atol=args.atol, rtol=args.rtol)
    if not result["correctness"]["allclose"] or not result["correctness"]["finite"]:
        result["status"] = "validation_failed"
        return result
    result["stage"] = "attention_timing"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for _ in range(args.warmup):
        call()
    torch.npu.synchronize()
    wall_ms, device_ms = [], []
    for _ in range(args.samples):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        started = time.perf_counter()
        start.record()
        for _ in range(args.calls_per_sample):
            call()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - started) * 1000 / args.calls_per_sample)
        device_ms.append(start.elapsed_time(end) / args.calls_per_sample)
    result["timing_ms_per_call"] = dict(wall_median=statistics.median(wall_ms),
        device_median=statistics.median(device_ms), wall_samples=wall_ms, device_samples=device_ms,
        warmup=args.warmup, calls_per_sample=args.calls_per_sample,
        scope="attention only; eager dispatch included; fixture/packing/writer/validation excluded")
    result["cache_after_attention"] = {"key": tensor_info(kc, torch_npu), "value": tensor_info(vc, torch_npu)}
    result["status"] = "passed"
    result["stage"] = "complete"
    return result


def cpu_self_test():
    import torch
    # Deliberately unique coordinates, shuffled physical pages, multiple KV
    # heads, and D > 16 catch head/sequence/lane interchange errors.
    batch, heads, capacity, dim, block = 3, 2, 24, 32, 8
    t = torch.arange(batch * heads * capacity * dim).view(batch, heads, capacity, dim)
    mapping = block_mapping(batch, capacity, block, 99)
    for layout in ("dense", "pages", "pages4", "blocked4", "blocked5"):
        packed = pack_cache(t, torch, mapping, layout, block)
        restored = unpack_cache(packed, torch, mapping, layout, batch, capacity, heads, dim, block)
        assert torch.equal(t, restored), layout
        # Independent scalar indexing establishes physical placement, not only
        # a pack/unpack pair that might share the same permutation mistake.
        for row, head, pos, lane in itertools.product(range(batch), range(heads),
                                                     range(capacity), range(dim)):
            page, offset = mapping[row][pos // block], pos % block
            coord = {"dense": (row, head, pos, lane), "pages": (page, offset, head * dim + lane),
                     "pages4": (page, offset, head, lane),
                     "blocked4": (page, head * dim // 16 + lane // 16, offset, lane % 16),
                     "blocked5": (page, head, lane // 16, offset, lane % 16)}[layout]
            assert packed[coord] == t[row, head, pos, lane], (layout, coord)
    print("CPU_SELF_TEST: five layouts pass roundtrip and independent coordinate checks; no NPU validation")


def main():
    args = parser().parse_args()
    if args.cpu_self_test:
        cpu_self_test()
        return 0
    if not args.chip or not args.output:
        parser().error("--chip and --output are required for a matrix or worker")
    if min(args.samples, args.calls_per_sample, args.cache_length) < 1 or args.warmup < 0 or args.timeout <= 0:
        parser().error("samples/calls/timeout must be positive and warmup nonnegative")
    if args.worker:
        case = json.loads(args.worker)
        try:
            result = run_worker(args, case)
        except Exception as exc:
            result = json.loads(args.output.read_text()) if args.output.exists() else {"case": case}
            result.update(status="operation_error", error=f"{type(exc).__name__}: {exc}",
                          traceback=traceback.format_exc())
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return 0 if result["status"] == "passed" else 1
    cases = list(case_matrix(args))
    if args.plan:
        print(json.dumps(cases, indent=2))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    details = args.output.parent / (args.output.stem + "_cases")
    details.mkdir(exist_ok=False)  # Never overwrite another run's evidence.
    results = []
    for i, case in enumerate(cases):
        path = details / f"{i:04d}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(case),
            "--chip", args.chip, "--output", str(path), "--block-size", str(args.block_size),
            "--warmup", str(args.warmup), "--samples", str(args.samples),
            "--calls-per-sample", str(args.calls_per_sample), "--seed", str(args.seed),
            "--atol", str(args.atol), "--rtol", str(args.rtol)]
        with path.with_suffix(".log").open("w") as log:
            case_started_unix_s = time.time()
            try:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                status = "operation_error" if completed.returncode else "passed"
            except subprocess.TimeoutExpired:
                status = "timeout"
        result = json.loads(path.read_text()) if path.exists() else {"case": case, "status": status}
        if status == "timeout":
            result["status"] = status
        result["command"] = command
        result["case_started_unix_s"] = case_started_unix_s
        result["case_finished_unix_s"] = time.time()
        result["log"] = str(path.with_suffix(".log"))
        path.write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        error_excerpt = result.get("error", "")
        if error_excerpt:
            error_excerpt = error_excerpt.splitlines()[0][:500]
        print(json.dumps({"case": case, "status": result["status"],
                          "timing": result.get("timing_ms_per_call", {}).get("device_median"),
                          "error": error_excerpt or None}), flush=True)
        args.output.write_text(json.dumps({"source_commit": SOURCE_COMMIT, "results": results}, indent=2) + "\n")
        if result["status"] == "timeout":
            # A device-side hang can survive terminating the worker. Do not
            # submit more work to the same shared device after a timeout.
            print("STOPPED_AFTER_TIMEOUT: inspect device health before another run", flush=True)
            break
    pairs = []
    for nd in results:
        if nd["status"] != "passed" or nd["case"]["format"] != 2:
            continue
        target = dict(nd["case"], format=29)
        nz = next((r for r in results if r["case"] == target and r["status"] == "passed"), None)
        if nz and nd["fixture_sha256"] == nz["fixture_sha256"]:
            pairs.append(dict(case=nd["case"], nd_over_nz_device_latency=
                nd["timing_ms_per_call"]["device_median"] / nz["timing_ms_per_call"]["device_median"]))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    summary = {"source_commit": SOURCE_COMMIT, "repo_commit": commit,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "requested_chip": args.chip,
               "planned_case_count": len(cases), "completed_case_count": len(results),
               "command": sys.argv, "results": results, "same_layout_format_comparisons": pairs,
               "note": "No NPU result transfers across chips; this is one-layer synthetic decode, not MinerU e2e."}
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print("FORMAT_COMPARISONS", json.dumps(pairs), flush=True)
    if any(r["status"] == "validation_failed" for r in results):
        return 2
    if len(results) != len(cases) or any(r["status"] == "timeout" for r in results):
        return 1
    return 0 if any(r["status"] == "passed" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
