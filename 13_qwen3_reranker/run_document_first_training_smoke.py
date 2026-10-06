"""One-NPU full fine-tuning smoke; no model/optimizer checkpoints are written."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

from training_smoke_data import body, ranking_metrics, select_groups
from transformers_rerank import PREFIX, SUFFIX, DEFAULT_TASK


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial.json")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["step_timing", "train"], default="train")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--microbatch", type=int, default=4)
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=731)
    p.add_argument("--wall-time-limit", type=float, default=2400)
    args = p.parse_args()
    import torch
    import torch.nn.functional as F
    import torch_npu
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    from local_modeling_qwen3_reranker import (
        LocalQwen3RerankerConfig, LocalQwen3RerankerForCausalLM,
        build_left_padded_causal_bool_mask,
    )

    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.npu.set_device("npu:0")
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                              padding_side="left")
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prefix, suffix = encode(PREFIX), encode(SUFFIX)
    yes, no = (tokenizer.convert_tokens_to_ids(t) for t in ("yes", "no"))
    assert encode("yes") == [yes] and encode("no") == [no]

    def ids(query, document, order):
        text = body(DEFAULT_TASK, query, document, order)
        return prefix + encode(text) + suffix

    fits = lambda q, d: all(len(ids(q, d, order)) <= args.max_length
                           for order in ("query_first", "document_first"))
    source = json.loads(args.source.read_text())
    dataset = select_groups(source, fits, seed=args.seed)
    save(args.output / "inputs.json", dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "scope": "BGE-M3 subset smoke, not benchmark parity",
              "environment": {"host": platform.node(), "chip": "Ascend 910B2",
                              "physical_npu": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
                              "torch": torch.__version__, "torch_npu": torch_npu.__version__,
                              "transformers": transformers.__version__},
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in [Path(__file__).name, "training_smoke_data.py",
                                             "local_modeling_qwen3_reranker.py"]},
              "data_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "training": {"attention": "npu_fusion_attention", "weights": "float32",
                           "autocast": "bfloat16", "optimizer": "NpuFusedAdamW",
                           "loss": "final-position yes/no cross entropy", "weight_decay": 0.0,
                           "gradient_clip": 1.0, "checkpoints_saved": False},
              "split_checks": dataset["checks"], "evaluations": {}, "updates": []}
    save(args.output / "result.json", result)
    print("DATA_READY", json.dumps({"train_queries": len(dataset["train"]),
          "eval_queries": len(dataset["eval"]), "checks": dataset["checks"]}), flush=True)

    def batches(groups, order):
        records = [(g["query"], d, label, g["id"]) for g in groups
                   for d, label in zip(g["documents"], g["labels"])]
        output = []
        for start in range(0, len(records), args.microbatch):
            rows = records[start:start + args.microbatch]
            encoded = [ids(q, d, order) for q, d, _, _ in rows]
            assert all(len(row) <= args.max_length for row in encoded)
            x = tokenizer.pad({"input_ids": encoded}, padding="max_length",
                              max_length=args.max_length, return_tensors="pt")
            x = {k: v.to(device) for k, v in x.items()}
            positions = (x["attention_mask"].long().cumsum(-1) - 1).clamp_min(0)
            mask = build_left_padded_causal_bool_mask(x["attention_mask"])
            output.append((x, positions, mask,
                           torch.tensor([r[2] for r in rows], device=device),
                           sum(len(row) for row in encoded)))
        return output

    eval_batches = {order: batches(dataset["eval"], order)
                    for order in ("query_first", "document_first")}
    train_groups = list(dataset["train"])
    random.Random(args.seed + 1).shuffle(train_groups)
    train_batches = batches(train_groups, "document_first")
    answer_ids = torch.tensor([no, yes], device=device)

    @torch.no_grad()
    def evaluate(model, prepared, hf=False):
        model.eval()
        values = []
        t = time.monotonic()
        for x, positions, mask, _, _ in prepared:
            with torch.autocast("npu", dtype=torch.bfloat16):
                if hf:
                    z = model(**x, position_ids=positions, use_cache=False,
                              logits_to_keep=1).logits[:, -1, answer_ids]
                else:
                    h = model.forward_hidden_states_prepared(x["input_ids"], positions, mask)
                    z = F.linear(h[:, -1], model.lm_head.weight[answer_ids])
            values.extend(z.float().cpu().tolist())
        return {"metrics": ranking_metrics(dataset["eval"], values),
                "logits": values, "seconds": time.monotonic() - t}

    if args.mode == "train":
        hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                  attn_implementation="eager", local_files_only=True).to(device)
        result["evaluations"]["original_transformers_query_first"] = evaluate(hf, eval_batches["query_first"], hf=True)
        del hf
        gc.collect()
        torch.npu.empty_cache()
        save(args.output / "result.json", result)
        print("HF_BASELINE", json.dumps(result["evaluations"]["original_transformers_query_first"]["metrics"]), flush=True)

    config = LocalQwen3RerankerConfig.from_model_dir(args.model)
    model = LocalQwen3RerankerForCausalLM(config, attention_impl="fusion_attention")
    state = {}
    for path in sorted(args.model.glob("*.safetensors")):
        state.update({k.removeprefix("model."): v for k, v in load_file(str(path)).items()})
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and (not missing or (config.tie_word_embeddings and missing == ["lm_head.weight"]))
    del state
    model.to(device=device, dtype=torch.float32)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    optimizer = torch_npu.optim.NpuFusedAdamW(model.parameters(), lr=args.learning_rate,
                                            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    if args.mode == "train":
        for order in ("query_first", "document_first"):
            result["evaluations"]["step_0_" + order] = evaluate(model, eval_batches[order])
            print("BASELINE", order, json.dumps(result["evaluations"]["step_0_" + order]["metrics"]), flush=True)
        save(args.output / "result.json", result)
    torch.npu.reset_peak_memory_stats()
    for step in range(1, args.steps + 1):
        if time.monotonic() - started > args.wall_time_limit:
            raise RuntimeError("Smoke wall-time limit reached")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.npu.synchronize()
        begin = time.monotonic()
        total_loss = torch.zeros((), device=device)
        tokens = 0
        tracked = model.layers[0].self_attn.q_proj.weight
        before = tracked[:4, :16].detach().clone()
        for micro in range(args.accumulation):
            index = ((step - 1) * args.accumulation + micro) % len(train_batches)
            x, positions, mask, labels, n = train_batches[index]
            with torch.autocast("npu", dtype=torch.bfloat16):
                h = model.forward_hidden_states_prepared(x["input_ids"], positions, mask)
                z = F.linear(h[:, -1], model.lm_head.weight[answer_ids])
                loss = F.cross_entropy(z.float(), labels) / args.accumulation
            loss.backward()
            total_loss += loss.detach()
            tokens += n
        torch.npu.synchronize()
        backward_s = time.monotonic() - begin
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                   error_if_nonfinite=True)
        optimizer_start = time.monotonic()
        optimizer.step()
        torch.npu.synchronize()
        optimizer_s = time.monotonic() - optimizer_start
        seconds = time.monotonic() - begin
        changed = bool((tracked[:4, :16].detach() != before).any().item())
        row = {"step": step, "loss": float(total_loss.cpu()), "gradient_norm": float(grad_norm.cpu()),
               "parameter_changed": changed, "forward_backward_seconds": backward_s,
               "optimizer_seconds": optimizer_s, "total_update_seconds": seconds,
               "pairs": args.microbatch * args.accumulation, "real_tokens": tokens,
               "peak_allocated_gib": torch.npu.max_memory_allocated() / 1024**3,
               "peak_reserved_gib": torch.npu.max_memory_reserved() / 1024**3}
        assert changed, "Optimizer did not update sampled Q projection weights"
        if step == 1:
            result["training"]["optimizer_state_dtypes"] = sorted({str(v.dtype) for s in optimizer.state.values()
                                  for v in s.values() if isinstance(v, torch.Tensor)})
        result["updates"].append(row)
        print("UPDATE", json.dumps(row), flush=True)
        if args.mode == "train" and step in {1, 5, args.steps}:
            result["evaluations"][f"step_{step}_document_first"] = evaluate(model, eval_batches["document_first"])
            print("EVAL", step, json.dumps(result["evaluations"][f"step_{step}_document_first"]["metrics"]), flush=True)
        save(args.output / "result.json", result)
    if args.mode == "train":
        result["evaluations"][f"step_{args.steps}_query_first"] = evaluate(model, eval_batches["query_first"])
    result["status"] = "completed"
    result["total_seconds"] = time.monotonic() - started
    save(args.output / "result.json", result)
    print("COMPLETED", json.dumps({"seconds": result["total_seconds"], "updates": len(result["updates"])}), flush=True)


if __name__ == "__main__":
    main()
