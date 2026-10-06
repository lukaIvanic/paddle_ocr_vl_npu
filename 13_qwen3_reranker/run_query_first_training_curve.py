"""Bounded query-first NPU training curve; log weights' effects without saving weights."""
import argparse
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

from curve_metrics import ndcg10, validation_metrics
from mixture_data import EXCLUDED, quotas, text_hash
from training_smoke_data import body


def save(path, data):
    temp = path.with_suffix(".partial.json")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plans(records, max_batch, max_tokens):
    """Length-sorted CPU records; never preallocate all quadratic NPU masks."""
    ordered = sorted(records, key=lambda r: (len(r["ids"]), r["index"]))
    batch = []
    for record in ordered:
        length = math.ceil(len(record["ids"]) / 128) * 128
        if batch and (len(batch) == max_batch or (len(batch) + 1) * length > max_tokens):
            yield batch
            batch = []
        batch.append(record)
    if batch:
        yield batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["pilot", "curve"], default="curve")
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--eval-steps", default="0,10,50,100,250")
    p.add_argument("--validation-queries", type=int, default=384)
    p.add_argument("--touche-queries", type=int, default=49)
    p.add_argument("--pairs-per-update", type=int, default=32)
    p.add_argument("--microbatch", type=int, default=4)
    p.add_argument("--train-batch-tokens", type=int, default=8192)
    p.add_argument("--eval-batch-tokens", type=int, default=16384)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=731)
    p.add_argument("--wall-time-limit", type=float, default=2400)
    args = p.parse_args()
    assert args.max_length % 128 == 0 and args.pairs_per_update % 2 == 0
    assert args.train_batch_tokens >= args.max_length
    import torch
    import torch.nn.functional as F
    import torch_npu
    import transformers
    from transformers_rerank import DEFAULT_TASK, PREFIX, SUFFIX
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from safetensors.torch import load_file
    from local_modeling_qwen3_reranker import (
        LocalQwen3RerankerConfig, LocalQwen3RerankerForCausalLM,
        build_left_padded_causal_bool_mask,
    )
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(gzip.decompress(args.dataset.read_bytes()))
    fixture = json.loads(args.fixture.read_text())
    assert fixture["task"] == "Touche2020Retrieval.v3"
    assert dataset["provenance"]["excluded_families"] == EXCLUDED
    assert digest(args.fixture) == dataset["provenance"]["touche_exclusion"]["fixture_sha256"]
    blocked_queries = {text_hash(x["text"]) for x in fixture["queries"].values()}
    blocked_docs = {text_hash(x["text"]) for x in fixture["documents"].values()}
    val_queries = {text_hash(r["query"]) for r in dataset["validation"]}
    val_docs = {text_hash(d) for r in dataset["validation"] for d in r["documents"]}
    assert not val_queries & {text_hash(r["query"]) for r in dataset["train"]}
    assert not val_docs & {text_hash(d) for r in dataset["train"] for d in r["documents"]}
    for g in dataset["train"] + dataset["validation"]:
        assert g["source"] not in EXCLUDED and text_hash(g["query"]) not in blocked_queries
        assert not {text_hash(d) for d in g["documents"]} & blocked_docs

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="left")
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prefix, suffix = encode(PREFIX), encode(SUFFIX)
    limit = args.max_length - len(prefix) - len(suffix)
    no, yes = (tokenizer.convert_tokens_to_ids(x) for x in ("no", "yes"))
    assert encode("no") == [no] and encode("yes") == [yes]
    answer_ids = torch.tensor([no, yes], device=device)
    length_stats = {}

    def records(groups, instruction, section):
        output = []
        original_lengths, truncated = [], 0
        for g in groups:
            for j, document in enumerate(g["documents"]):
                raw = encode(body(instruction, g["query"], document, "query_first"))
                original_lengths.append(len(raw) + len(prefix) + len(suffix))
                truncated += len(raw) > limit
                output.append({"ids": prefix + raw[:limit] + suffix,
                               "index": len(output), "group_id": g["id"],
                               "source": g.get("source", "touche"),
                               "label": g.get("labels", [0] * len(g["documents"]))[j],
                               "document_id": g.get("document_ids", [None] * len(g["documents"]))[j]})
        ordered = sorted(original_lengths)
        length_stats[section] = {"pairs": len(output), "truncated_pairs": truncated,
             "original_length_percentiles": {str(q): ordered[round(q * (len(ordered) - 1))] for q in (0, .5, .9, .99, 1)},
             "actual_tokens": sum(len(r["ids"]) for r in output)}
        return output

    validation = dataset["validation"]
    if args.validation_queries < len(validation):
        sizes = {s: sum(g["source"] == s for g in validation) for s in {g["source"] for g in validation}}
        q = quotas(args.validation_queries, sizes, sizes, minimum=1)
        selected = []
        for g in validation:
            if q[g["source"]]:
                selected.append(g)
                q[g["source"]] -= 1
        validation = selected
    tqids = sorted(fixture["queries"], key=lambda q: hashlib.sha256(("qwen-curve/" + q).encode()).hexdigest())[:args.touche_queries]
    touche = []
    for qid in tqids:
        q = fixture["queries"][qid]
        dids = sorted(q["candidates"], key=lambda d: (-q["candidates"][d], d))
        assert len(dids) == 100
        touche.append({"id": qid, "query": q["text"], "document_ids": dids,
                       "documents": [fixture["documents"][d]["text"] for d in dids]})
    train_groups = list(dataset["train"])
    random.Random(args.seed + 1).shuffle(train_groups)
    train = records(train_groups, DEFAULT_TASK, "train")
    val = records(validation, DEFAULT_TASK, "validation")
    touch = records(touche, fixture["instruction"], "touche")
    selected_qrels = {q: fixture["queries"][q]["qrels"] for q in tqids}
    reference = ndcg10({q: fixture["queries"][q]["qwen_scores"] for q in tqids},
                       selected_qrels, fixture["ignore_identical_ids"])
    if len(tqids) == 49:
        assert abs(reference["ndcg10"] * 100 - 72.9999) < 0.001, "Frozen Qwen metric reproduction failed"
    result = {"status": "running", "scope": "query-first experimental training curve",
        "environment": {"host": platform.node(), "chip": "Ascend 910B2",
            "physical_npu": os.getenv("ASCEND_RT_VISIBLE_DEVICES"), "torch": torch.__version__,
            "torch_npu": torch_npu.__version__, "transformers": transformers.__version__},
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_sha256": {n: digest(Path(__file__).with_name(n)) for n in
            [Path(__file__).name, "mixture_data.py", "curve_metrics.py", "local_modeling_qwen3_reranker.py", "training_smoke_data.py"]},
        "dataset_sha256": digest(args.dataset), "fixture_sha256": digest(args.fixture),
        "lengths": length_stats, "distribution": dataset["distribution"],
        "split_checks": dataset["checks"], "validation_ids": [g["id"] for g in validation],
        "touche_query_ids": tqids, "frozen_qwen4b_reference": reference,
        "training": {"prompt_order": "query_first", "attention": "npu_fusion_attention",
            "optimizer": "NpuFusedAdamW", "master_weights": "float32", "autocast": "bfloat16",
            "weight_decay": 0.0, "gradient_clip": 1.0, "loss": "pointwise no/yes cross entropy",
            "checkpoints_saved": False, "truncation": f"right-truncate body at {args.max_length} total, preserve prefix/suffix",
            "scoring": "yes-minus-no logit margin, monotonic with binary yes probability"},
        "training_order_ids": [g["id"] for g in train_groups],
        "updates": [], "evaluations": {}}
    save(args.output / "result.json", result)
    print("PREPARED", json.dumps({"seconds": time.monotonic() - started, "lengths": length_stats,
          "sources": len(dataset["distribution"]), "validation_queries": len(validation),
          "touche_queries": len(touche)}), flush=True)

    def batch(rows):
        padded = math.ceil(max(len(r["ids"]) for r in rows) / 128) * 128
        x = tokenizer.pad({"input_ids": [r["ids"] for r in rows]}, padding="max_length",
                          max_length=padded, return_tensors="pt")
        x = {k: v.to(device) for k, v in x.items()}
        positions = (x["attention_mask"].cumsum(-1) - 1).clamp_min(0)
        mask = build_left_padded_causal_bool_mask(x["attention_mask"])
        return x, positions, mask

    def logits(model, rows, hf=False):
        x, pos, mask = batch(rows)
        with torch.autocast("npu", dtype=torch.bfloat16):
            if hf:
                return model(**x, position_ids=pos, use_cache=False, logits_to_keep=1).logits[:, -1, answer_ids]
            h = model.forward_hidden_states_prepared(x["input_ids"], pos, mask)
            return F.linear(h[:, -1], model.lm_head.weight[answer_ids])

    @torch.no_grad()
    def score(model, rows, section):
        model.eval()
        values = [None] * len(rows)
        t = time.monotonic()
        done = 0
        for i, micro in enumerate(plans(rows, 16, args.eval_batch_tokens), 1):
            z = logits(model, micro).float().cpu().tolist()
            for r, pair in zip(micro, z):
                assert all(math.isfinite(v) for v in pair)
                values[r["index"]] = pair
            done += len(micro)
            if i % 40 == 0:
                print("EVAL_PROGRESS", json.dumps({"section": section, "pairs": done,
                      "of": len(rows), "seconds": time.monotonic() - t}), flush=True)
        assert all(v is not None for v in values)
        return values, time.monotonic() - t

    config = LocalQwen3RerankerConfig.from_model_dir(args.model)
    model = LocalQwen3RerankerForCausalLM(config, attention_impl="fusion_attention")
    state = {}
    for path in sorted(args.model.glob("*.safetensors")):
        state.update({k.removeprefix("model."): v for k, v in load_file(str(path)).items()})
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and (not missing or (config.tie_word_embeddings and missing == ["lm_head.weight"]))
    del state
    model.to(device=device, dtype=torch.float32)

    if args.mode == "curve":
        short = [r for r in val + touch if len(r["ids"]) <= 1024][:32]
        # Run HF and local controls sequentially to avoid two model copies during training.
        model.to("cpu")
        torch.npu.empty_cache()
        hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                  attn_implementation="eager", local_files_only=True).to(device).eval()
        with torch.no_grad():
            ref = [p for rows in plans(short, 4, 4096) for p in logits(hf, rows, hf=True).float().cpu().tolist()]
        del hf
        gc.collect()
        torch.npu.empty_cache()
        model.to(device)
        with torch.no_grad():
            own = [p for rows in plans(short, 4, 4096) for p in logits(model, rows).float().cpu().tolist()]
        result["hf_control"] = {"pairs": len(short), "max_logit_difference": max(abs(x-y) for a,b in zip(ref,own) for x,y in zip(a,b)),
            "max_margin_difference": max(abs((a[1]-a[0])-(b[1]-b[0])) for a,b in zip(ref,own))}
        print("HF_CONTROL", json.dumps(result["hf_control"]), flush=True)

    optimizer = torch_npu.optim.NpuFusedAdamW(model.parameters(), lr=args.learning_rate,
                                            betas=(.9, .999), eps=1e-8, weight_decay=0.0)

    def evaluate(step, pilot=False):
        t = time.monotonic()
        if pilot:
            vg = validation[:48]
            tg = touche[:8]
            vr = records(vg, DEFAULT_TASK, "pilot_validation")
            tr = records(tg, fixture["instruction"], "pilot_touche")
        else:
            vg, tg, vr, tr = validation, touche, val, touch
        v, vs = score(model, vr, "validation")
        z, ts = score(model, tr, "touche")
        predictions = {}
        for r, (no_logit, yes_logit) in zip(tr, z):
            predictions.setdefault(r["group_id"], {})[r["document_id"]] = yes_logit - no_logit
        metric = ndcg10(predictions, {g["id"]: selected_qrels[g["id"]] for g in tg}, fixture["ignore_identical_ids"])
        item = {"validation": validation_metrics(vg, v), "validation_logits": v,
                "touche": metric, "touche_margins": predictions,
                "seconds": {"validation": vs, "touche": ts, "round": time.monotonic() - t},
                "validation_queries": len(vg), "touche_queries": len(tg)}
        result["evaluations"][str(step)] = item
        save(args.output / "result.json", result)
        print("EVALUATION", json.dumps({"step": step, "validation_ordering": item["validation"]["overall"]["ordering_accuracy"],
              "touche_ndcg10_percent": 100 * metric["ndcg10"], "seconds": item["seconds"]}), flush=True)

    evaluate(0, pilot=args.mode == "pilot")
    torch.npu.reset_peak_memory_stats()
    eval_steps = {int(s) for s in args.eval_steps.split(",")}
    try:
        for step in range(1, args.steps + 1):
            if time.monotonic() - started > args.wall_time_limit:
                result["status"] = "time_limit"
                break
            model.train()
            torch.npu.synchronize()
            begin = time.monotonic()
            optimizer.zero_grad(set_to_none=False)
            count = args.pairs_per_update
            window = [train[((step-1) * count + j) % len(train)] for j in range(count)]
            total_loss = torch.zeros((), device=device)
            tracked = model.layers[0].self_attn.q_proj.weight
            before = tracked[:4, :16].detach().clone()
            micros, padded_tokens = 0, 0
            for rows in plans(window, args.microbatch, args.train_batch_tokens):
                target = torch.tensor([r["label"] for r in rows], device=device)
                z = logits(model, rows)
                loss = F.cross_entropy(z.float(), target, reduction="sum") / count
                loss.backward()
                total_loss += loss.detach()
                micros += 1
                padded_tokens += len(rows) * math.ceil(max(len(r["ids"]) for r in rows) / 128) * 128
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.npu.synchronize()
            row = {"step": step, "seconds": time.monotonic() - begin, "loss": float(total_loss.cpu()),
                   "gradient_norm": float(grad_norm.cpu()), "pairs": count, "microbatches": micros,
                   "real_tokens": sum(len(r["ids"]) for r in window), "padded_tokens": padded_tokens,
                   "peak_allocated_gib": torch.npu.max_memory_allocated() / 1024**3,
                   "peak_reserved_gib": torch.npu.max_memory_reserved() / 1024**3,
                   "parameter_changed": bool((tracked[:4, :16].detach() != before).any().item())}
            row["pairs_by_source"] = {s: sum(r["source"] == s for r in window) for s in {r["source"] for r in window}}
            assert row["parameter_changed"]
            result["updates"].append(row)
            print("UPDATE", json.dumps(row), flush=True)
            if step == 1:
                result["training"]["optimizer_state_dtypes"] = sorted({str(v.dtype) for s in optimizer.state.values() for v in s.values() if isinstance(v, torch.Tensor)})
            if args.mode == "pilot" and step == args.steps:
                evaluate(step, pilot=True)
            elif args.mode == "curve" and (step in eval_steps or step == args.steps):
                evaluate(step)
            save(args.output / "result.json", result)
        else:
            result["status"] = "completed"
    except Exception as e:
        result.update(status="failed", error=repr(e))
        raise
    finally:
        result["total_seconds"] = time.monotonic() - started
        save(args.output / "result.json", result)
    print("FINISHED", json.dumps({"status": result["status"], "seconds": result["total_seconds"],
          "updates": len(result["updates"])}), flush=True)


if __name__ == "__main__":
    main()
