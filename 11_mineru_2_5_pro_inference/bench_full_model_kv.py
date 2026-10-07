#!/usr/bin/env python3
"""Real-crop MinerU generation with complete static full-graph decoder steps.

No synthetic weights, tokens or KV. Vision/text prefill is real but outside the
decode interval. Every timed step contains all 24 text layers, KV writes,
LM head, greedy sampling and position advancement. No eager fallback.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import types
import warnings

VARIANTS = {
    "increfa_nd": ("increfa", "dense", 2),
    "increfa_nz": ("increfa", "dense", 29),
    "fia_nd": ("fia", "pages4", 2),
    "fia_blocked_nd": ("fia", "blocked4", 2),
    "fia_nz": ("fia", "blocked4", 29),
    "paged_nd": ("paged", "pages4", 2),
    "paged_nz": ("paged", "blocked4", 29),
}


def token_metrics(first_tokens, histories, eos, max_tokens):
    """Count useful outputs once; exclude EOS and all inactive-slot work."""
    rows = [[int(t)] for t in first_tokens]
    for sampled in histories:
        for row, t in zip(rows, sampled):
            if row[-1] != eos and len(row) < max_tokens:
                row.append(int(t))
    lengths = [len(row) - (row[-1] == eos) for row in rows]
    return {
        "token_ids": rows, "output_tokens_excluding_eos": sum(lengths),
        "decode_tokens_excluding_prefill_and_eos": sum(
            length - (first != eos) for length, first in zip(lengths, first_tokens)),
        "eos_count": sum(row[-1] == eos for row in rows),
        "length_cap_hit_count": sum(row[-1] != eos for row in rows),
        "per_item_output_lengths": lengths,
    }


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path(__file__).parents[1]/"crops/hotswap_100_manifest.json")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--cache-length", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--variants", default="increfa_nd,fia_nd,fia_blocked_nd,increfa_nz,fia_nz")
    p.add_argument("--chip", choices=("910B", "310P"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--max-pixels", type=int, default=602112)
    p.add_argument("--profile", action="store_true")
    return p


def run(args, report):
    import torch
    import torch.nn.functional as F
    import torch_npu
    from PIL import Image
    from transformers import AutoProcessor
    import local_modeling_mineru as lm
    from run_local_model_two_step_extract import (
        DEFAULT_SYSTEM_PROMPT, collect_model_identity, get_rgb_image, select_prompt, import_torchair,
    )

    torch.set_num_threads(8)
    torch.npu.set_device("npu:0")
    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    chip = torch.npu.get_device_name(0)
    if args.chip.upper() not in chip.upper():
        raise RuntimeError(f"Requested {args.chip}, observed {chip}; no mislabeled run")
    report.update(device_name=chip, visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
                  cann_home=os.environ.get("ASCEND_HOME_PATH"), hostname=os.uname().nodename)
    save(args.output, report)
    tng, CompilerConfig = import_torchair()
    setup = time.perf_counter()
    model = lm.LocalMinerU2_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16, device="npu:0").eval()
    lm.configure_decode_packed_projections(model)
    report["weight_format"] = lm.configure_decode_weight_format(model, "decode_nz")
    lm.configure_decode_rotary_impl(model, "npu_apply")
    lm.configure_decode_attention_impl(model, "increfa")
    lm.configure_decode_increfa_length_mode(model, "pse_sentinel_310p")
    processor = AutoProcessor.from_pretrained(args.model, use_fast=False, local_files_only=True)
    processor.image_processor.min_pixels = 25088
    processor.image_processor.max_pixels = args.max_pixels
    if isinstance(getattr(processor.image_processor,"size",None), dict):
        processor.image_processor.size["shortest_edge"] = 25088
        processor.image_processor.size["longest_edge"] = args.max_pixels
    report["processor"] = {"class":type(processor.image_processor).__name__,
        "min_pixels":processor.image_processor.min_pixels,"max_pixels":processor.image_processor.max_pixels,
        "size":getattr(processor.image_processor,"size",None)}
    report["model_identity"] = collect_model_identity(args.model, hash_model_files=True)
    report["model_layers"] = model.config.text_config.num_hidden_layers
    if report["model_layers"] != 24:
        raise RuntimeError("Benchmark requires the checkpoint's complete 24-layer decoder")
    eos = int(model.config.eos_token_id)
    manifest = json.loads(args.manifest.read_text())[:args.limit]
    if len(manifest) != args.limit:
        raise ValueError("Manifest has fewer real crops than --limit")
    if len({item["file"] for item in manifest}) != args.limit:
        raise ValueError("Real crop inputs must be distinct")
    prepared, items = [], []
    for entry in manifest:
        path = (args.manifest.parent / entry["file"]).resolve()
        kind = entry["category_type"]
        block_type = "table" if "table" in kind else "equation" if ("equation" in kind or "formula" in kind) else "text"
        prompt = select_prompt(block_type)
        with Image.open(path) as image:
            image = get_rgb_image(image).copy()
        chat = processor.apply_chat_template([
            {"role":"system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role":"user", "content":[{"type":"image"}, {"type":"text", "text":prompt}]},
        ], tokenize=False, add_generation_prompt=True)
        inp = processor(text=[chat], images=[image], return_tensors="pt", padding=True)
        raw_vision_tokens = sum(int(t)*int(h)*int(w) for t,h,w in inp.image_grid_thw.tolist())
        if raw_vision_tokens * 196 > args.max_pixels:
            raise RuntimeError(f"{path.name}: processor ignored the requested pixel cap")
        n = int(inp.input_ids.shape[1])
        if n + args.max_new_tokens > args.cache_length:
            raise ValueError(f"{path.name}: {n} prompt + generation cap exceeds KV capacity")
        items.append({"id":entry["id"], "file":str(path), "image_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                      "source_image":entry.get("source_image"), "category":kind, "prompt":prompt,
                      "input_tokens":n, "input_ids":inp.input_ids[0].tolist(),
                      "raw_vision_tokens":raw_vision_tokens,
                      "image_grid_thw":inp.image_grid_thw.tolist()})
        prepared.append(inp.to(device="npu:0", dtype=torch.float16))
    report["inputs"] = items
    report["setup_s"] = time.perf_counter() - setup
    save(args.output, report)

    class FullStep(torch.nn.Module):
        def __init__(self, variant, batch):
            super().__init__()
            self.model = model
            self.op, self.layout, self.fmt = VARIANTS[variant]
            self.eos = eos
            self.block = args.block_size
            self.capacity = args.cache_length
            self.register_buffer("block_table", torch.arange(batch*self.capacity//self.block, device="npu:0", dtype=torch.int32).view(batch,-1))

        def forward(self, input_ids, positions, rope, *caches):
            text = self.model.model
            hidden = text.embed_tokens(input_ids)
            batch = input_ids.shape[0]
            mask = lm.build_static_decode_mask(hidden, positions, self.capacity)
            mask, pse, _ = lm.apply_increfa_310p_pse_sentinel(mask, hidden, positions,
                cache_length=self.capacity, num_attention_heads=14, num_key_value_heads=2)
            position_ids = (positions.view(batch,1) + rope).unsqueeze(0).expand(3,-1,-1)
            factors = lm.prepare_multimodal_rotary_factors(*text.rotary_emb(hidden, position_ids), text.layers[0].self_attn.mrope_section)
            physical = torch.gather(self.block_table.to(torch.int64), 1,
                                    (positions // self.block).view(-1,1)).reshape(-1)
            offsets = positions.remainder(self.block)
            if self.layout == "blocked4":
                tiles = torch.arange(8, device=input_ids.device, dtype=torch.int64).view(1,-1).expand(batch,-1)
                write_indices = torch.stack((physical[:,None].expand(-1,8), tiles,
                                              offsets[:,None].expand(-1,8)), dim=-1).reshape(-1,3)
            else:
                write_indices = torch.stack((physical, offsets), dim=-1)
            residual = None
            for i, layer in enumerate(text.layers):
                if residual is None:
                    normed = lm._decode_rms_norm(layer.input_layernorm, hidden)
                    residual = hidden
                else:
                    normed, residual = lm._decode_add_rms_norm(hidden, residual, layer.input_layernorm)
                att = layer.self_attn
                kc, vc = caches[i], caches[i+24]
                if self.op == "increfa":
                    attended = att.forward_decode_static(normed, mask, factors, kc, vc, positions, pse)
                else:
                    q, k, v = att.project_qkv_decode_static(normed)
                    q, k = att.apply_decode_rotary(q, k, factors)
                    if self.op == "paged":
                        torch_npu._npu_reshape_and_cache(key=k.squeeze(2).contiguous(), value=v.squeeze(2).contiguous(),
                            key_cache=kc, value_cache=vc, slot_indices=(physical*self.block+offsets).to(torch.int32))
                        out = torch.empty_like(q.squeeze(2))
                        torch_npu._npu_paged_attention(query=q.squeeze(2).contiguous(), key_cache=kc, value_cache=vc,
                            num_heads=14, num_kv_heads=2, scale_value=att.scaling,
                            block_table=self.block_table, context_lens=(positions+1).to(torch.int32), out=out)
                    else:
                        if self.layout == "blocked4":
                            ku, vu = k.squeeze(2).contiguous().view(-1,16), v.squeeze(2).contiguous().view(-1,16)
                            k_read = kc.view(kc.shape[0],2,4,self.block,16)
                            v_read = vc.view_as(k_read)
                        else:
                            ku, vu = k.squeeze(2).contiguous(), v.squeeze(2).contiguous()
                            k_read = kc.view(kc.shape[0],self.block,128)
                            v_read = vc.view_as(k_read)
                        torch_npu.npu_scatter_nd_update_(kc, write_indices, ku)
                        torch_npu.npu_scatter_nd_update_(vc, write_indices, vu)
                        out = tng.ops.npu_fused_infer_attention_score(
                            q.transpose(1,2).contiguous().view(batch,1,896), k_read, v_read,
                            num_heads=14, num_key_value_heads=2, input_layout="BSH", scale=att.scaling,
                            actual_seq_lengths=torch.ones_like(positions), actual_seq_lengths_kv=positions+1,
                            block_table=self.block_table, block_size=self.block, sparse_mode=0, inner_precise=1)[0]
                    attended = lm.linear_last_dim(att.o_proj, out.reshape(batch,1,896))
                mlp_in, residual = lm._decode_add_rms_norm(attended, residual, layer.post_attention_layernorm)
                hidden = layer.mlp.forward_decode_static(mlp_in)
            hidden, _ = lm._decode_add_rms_norm(hidden, residual, text.norm)
            weight = self.model.decode_lm_head_weight
            logits = F.linear(hidden[:,-1,:], weight if weight is not None else text.embed_tokens.weight)
            sampled = logits.float().argmax(-1, keepdim=True)
            alive = input_ids != self.eos
            return torch.where(alive, sampled, input_ids), positions + alive.squeeze(1).to(torch.int64)

    def allocate(dense, variant):
        _, layout, fmt = VARIANTS[variant]
        batch, _, capacity, _ = dense[0].shape
        result = []
        for source in dense:
            if layout == "dense":
                packed = source
            else:
                pages = source.permute(0,2,1,3).contiguous().view(-1,args.block_size,2,64)
                packed = pages if layout == "pages4" else pages.view(-1,args.block_size,8,16).permute(0,2,1,3).contiguous()
            cache = torch_npu.empty_with_format(size=tuple(packed.shape), dtype=torch.float16,
                                                device="npu:0", acl_format=fmt)
            cache.copy_(packed)
            actual = int(torch_npu.get_npu_format(cache))
            if actual not in ((0,2) if fmt == 2 else (29,)):
                raise RuntimeError(f"Cache descriptor {fmt} not retained: {actual}; no substitute")
            if not torch.equal(torch_npu.npu_format_cast(cache,2).cpu(), packed.cpu()):
                raise RuntimeError("Real prefill KV changed during format conversion")
            result.append(cache)
        return tuple(result)

    def generate(fn, initial, caches):
        ids, positions, rope = (t.clone() for t in initial)
        first = ids.flatten().cpu().tolist()
        history = []
        torch.npu.synchronize()
        before_graphs = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        with warnings.catch_warnings(record=True) as caught:
            began = time.perf_counter()
            for _ in range(args.max_new_tokens-1):
                if all(t == eos for t in (history[-1] if history else first)):
                    break
                ids, positions = fn(ids, positions, rope, *caches)
                history.append(ids.flatten().cpu().tolist())
            torch.npu.synchronize()
            duration = time.perf_counter() - began
        after_graphs = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        return {**token_metrics(first, history, eos, args.max_new_tokens),
                "decode_s":duration, "forward_calls":len(history),
                "raw_batch_slots":len(history)*len(first),
                "new_graphs_during_generation":after_graphs-before_graphs,
                "recompile_warning_count":sum("recompiled" in str(w.message) for w in caught)}

    variants = args.variants.split(",")
    report["batches"] = []
    for start in range(0,len(prepared),args.batch_size):
        inputs = prepared[start:start+args.batch_size]
        batch = len(inputs)
        report["stage"] = f"real_prefill_batch_{start}"
        save(args.output,report)
        prefill = []
        torch.npu.synchronize()
        began = time.perf_counter()
        for inp in inputs:
            prefill.append(model.forward_static_prefill(input_ids=inp.input_ids, attention_mask=inp.attention_mask,
                pixel_values=inp.pixel_values, image_grid_thw=inp.image_grid_thw,
                cache_length=args.cache_length, logits_to_keep=1))
        torch.npu.synchronize()
        prefill_s = time.perf_counter()-began
        dense = tuple(torch.cat([p.cache.key_caches[i] for p in prefill],0).contiguous() for i in range(24)) + tuple(
            torch.cat([p.cache.value_caches[i] for p in prefill],0).contiguous() for i in range(24))
        initial = (torch.cat([p.logits[:,-1,:].float().argmax(-1,keepdim=True) for p in prefill]),
                   torch.cat([p.next_cache_position for p in prefill]), torch.cat([p.rope_deltas for p in prefill]))
        del prefill
        # Independent production flat decoder, raw eager, with the same real
        # prefill. It is a fidelity control, never a competing throughput lane.
        prod = model.make_flat_static_decode_module(cache_length=args.cache_length).eval()
        def reference(ids, positions, rope, *caches):
            logits = prod(ids,positions,rope,*caches)
            sampled = logits[:,-1,:].float().argmax(-1,keepdim=True)
            alive = ids != eos
            return torch.where(alive,sampled,ids), positions + alive.squeeze(1).to(torch.int64)
        report["stage"] = f"eager_production_reference_batch_{start}"
        save(args.output,report)
        ref = generate(reference, initial, tuple(t.clone() for t in dense))
        batch_report = {"input_indices":list(range(start,start+batch)), "prefill_s":prefill_s,
                        "reference":ref, "variants":{}, "repeat_order":[]}
        report["batches"].append(batch_report)
        functions = {}
        for variant in variants:
            state = {"status":"initializing", "samples":[], "scope":"complete compiled decoder with real advancing KV and EOS"}
            batch_report["variants"][variant] = state
            report["stage"] = f"compile_{variant}_batch_{start}"
            save(args.output,report)
            try:
                cache = allocate(dense,variant)
                state["cache_before"] = [{"shape":list(t.shape), "format":int(torch_npu.get_npu_format(t))} for t in cache]
                cfg = CompilerConfig()
                root = args.cache_dir / variant / f"b{batch}_kv{args.cache_length}"
                root.mkdir(parents=True,exist_ok=True)
                module = FullStep(variant,batch).eval()
                # Dynamo keys by Python code object, not just cache directory.
                # Distinct paths must not share this forward's specialization
                # cache, or TorchAir can repeatedly invalidate cached functions.
                method = module.forward
                isolated = types.FunctionType(method.__func__.__code__.replace(
                    co_name=f"forward_{variant}_b{batch}_{start}"),
                    method.__func__.__globals__, name=f"forward_{variant}_b{batch}_{start}",
                    argdefs=method.__func__.__defaults__, closure=method.__func__.__closure__)
                isolated.__qualname__ = f"FullStep.forward_{variant}_b{batch}_{start}"
                module.forward = types.MethodType(isolated,module)
                fn = tng.inference.cache_compile(module.forward, config=cfg, dynamic=False,
                    cache_dir=str(root), ge_cache=True, fullgraph=True)
                state["compile"] = {"api":"torchair.inference.cache_compile", "fullgraph":True,
                                    "dynamic":False,"cache_dir":str(root),"fallback":False}
                warm = tuple(t.clone() for t in initial)
                began = time.perf_counter()
                warm_ids, warm_positions = fn(*warm,*cache)
                torch.npu.synchronize()
                state["compiled_first_call_s"] = time.perf_counter()-began
                for _ in range(max(0,args.warmup_steps-1)):
                    warm_ids,warm_positions = fn(warm_ids,warm_positions,warm[2],*cache)
                torch.npu.synchronize()
                check = generate(fn,initial,allocate(dense,variant))
                state["validation"] = {"token_match":check["token_ids"]==ref["token_ids"],
                                        "token_ids":check["token_ids"]}
                if not state["validation"]["token_match"]:
                    state["status"] = "validation_failed"
                else:
                    functions[variant] = fn
                    state["status"] = "validated"
                state["cache_after_warmup"] = [int(torch_npu.get_npu_format(t)) for t in cache]
                del cache
            except Exception as exc:
                state.update(status="operation_error",error=repr(exc),traceback=traceback.format_exc())
            save(args.output,report)
            if variant == "increfa_nd" and state["status"] != "validated":
                raise RuntimeError("Compiled native full-model control failed; no benchmark is validated")
        # AB/BA ordering within one loaded checkpoint and one real-prefill bank.
        for repeat in range(args.repeats):
            order = list(functions) if repeat%2 == 0 else list(reversed(functions))
            batch_report["repeat_order"].append(order)
            for variant in order:
                report["stage"] = f"measure_{variant}_batch_{start}_repeat_{repeat}"
                save(args.output,report)
                caches = allocate(dense,variant)  # reset OUTSIDE timed generation
                measured = generate(functions[variant],initial,caches)
                measured["token_match"] = measured["token_ids"] == ref["token_ids"]
                measured["cache_formats_after"] = [int(torch_npu.get_npu_format(t)) for t in caches]
                acceptable = (0,2) if VARIANTS[variant][2] == 2 else (29,)
                measured["format_match"] = all(f in acceptable for f in measured["cache_formats_after"])
                measured["no_compile_in_timing"] = (measured["new_graphs_during_generation"] == 0 and
                                                       measured["recompile_warning_count"] == 0)
                state = batch_report["variants"][variant]
                state["samples"].append(measured)
                state["status"] = "passed" if all(s["token_match"] and s["format_match"] and s["no_compile_in_timing"]
                                                     for s in state["samples"]) else "validation_failed"
                del caches
                save(args.output,report)
        if args.profile and functions:
            batch_report["profiles"] = []
            for variant in functions:
                caches = allocate(dense,variant)
                ids, pos, rope = (t.clone() for t in initial)
                profile_dir = args.output.parent / f"profile_{variant}_batch{start}"
                with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
                      torch_npu.profiler.ProfilerActivity.NPU], record_shapes=True,
                      experimental_config=torch_npu.profiler._ExperimentalConfig(
                          profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
                      on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir))) as prof:
                    for _ in range(min(8,args.max_new_tokens-1)):
                        ids,pos = functions[variant](ids,pos,rope,*caches)
                        ids.cpu()
                        prof.step()
                    torch.npu.synchronize()
                batch_report["profiles"].append({"variant":variant,"directory":str(profile_dir),"outside_throughput":True})
                del caches
        del dense, initial, functions, prod
        torch.npu.empty_cache()
    aggregate = {}
    for variant in variants:
        states = [b["variants"][variant] for b in report["batches"]]
        if not all(s["status"] == "passed" for s in states):
            aggregate[variant] = {"status":"not_validated", "timing_valid":False}
            continue
        trials = []
        for r in range(args.repeats):
            samples = [s["samples"][r] for s in states]
            duration = sum(s["decode_s"] for s in samples)
            useful = sum(s["decode_tokens_excluding_prefill_and_eos"] for s in samples)
            trials.append({"decode_s":duration, "useful_decode_tokens":useful,
                           "useful_decode_tok_s":useful/duration if duration else None,
                           "raw_token_slots":sum(s["raw_batch_slots"] for s in samples),
                           "length_cap_hit_count":sum(s["length_cap_hit_count"] for s in samples)})
        aggregate[variant] = {"status":"passed", "trials":trials,
            "median_useful_decode_tok_s":statistics.median(t["useful_decode_tok_s"] for t in trials),
            "min_useful_decode_tok_s":min(t["useful_decode_tok_s"] for t in trials),
            "max_useful_decode_tok_s":max(t["useful_decode_tok_s"] for t in trials)}
    report["throughput"] = aggregate
    comparisons = []
    for baseline, candidate, scope in (
        ("increfa_nd", "increfa_nz", "same dense IncreFA contract; storage format differs"),
        ("fia_blocked_nd", "fia_nz", "same blocked FIA contract; storage format differs"),
        ("increfa_nd", "fia_nd", "full decoder path: attention and cache writer/layout differ"),
        ("increfa_nd", "fia_blocked_nd", "full decoder path: attention and cache writer/layout differ"),
    ):
        if (aggregate.get(baseline,{}).get("status") != "passed" or
                aggregate.get(candidate,{}).get("status") != "passed"):
            continue
        ratios = [c["useful_decode_tok_s"]/b["useful_decode_tok_s"]
                  for b,c in zip(aggregate[baseline]["trials"],aggregate[candidate]["trials"])]
        comparisons.append({"baseline":baseline,"candidate":candidate,"scope":scope,
            "paired_throughput_ratios":ratios,"median_ratio":statistics.median(ratios),
            "min_ratio":min(ratios),"max_ratio":max(ratios)})
    report["comparisons"] = comparisons
    report["status"] = "completed"
    report["stage"] = "complete"


def save(path, report):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")


def main():
    args = args_parser().parse_args()
    if min(args.limit,args.batch_size,args.max_new_tokens,args.cache_length,args.repeats,args.warmup_steps) <= 0:
        raise ValueError("Counts and capacities must be positive")
    if args.max_new_tokens < 2 or args.cache_length % args.block_size:
        raise ValueError("Need at least two output tokens and block-aligned capacity")
    variants = args.variants.split(",")
    if not variants or variants[0] != "increfa_nd" or any(v not in VARIANTS for v in variants):
        raise ValueError("First variant must be increfa_nd; use documented variant names")
    report = {"status":"initializing", "args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              "repo_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "measurement":"Useful post-prefill non-EOS tokens / synchronized full-model generation wall time; host EOS/token copies included; compile/prefill/reset excluded",
              "reference":"Production full MinerU static decoder, raw eager, identical real vision/text prefill",
              "no_eos_fill_tokens_counted":True}
    try:
        import torch
        with torch.inference_mode():
            run(args,report)
    except Exception as exc:
        report.update(status="failed",error=repr(exc),traceback=traceback.format_exc())
    save(args.output,report)
    print(json.dumps({"status":report["status"],"stage":report.get("stage"),
                      "throughput":report.get("throughput"),"error":report.get("error"),
                      "output":str(args.output)},indent=2),flush=True)
    return 0 if report["status"] == "completed" and all(v["status"]=="passed" for v in report.get("throughput",{}).values()) else 2


if __name__ == "__main__":
    sys.exit(main())
