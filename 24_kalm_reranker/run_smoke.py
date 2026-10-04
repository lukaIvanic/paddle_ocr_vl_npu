"""Real-NPU BF16 Nano-R2 smoke; uncached versus reused encoder outputs.

No CPU inference fallback, no vLLM, no compilation. Timings are synchronized
smoke observations, not a warmed throughput benchmark. Source: pinned KaLM
release's kalm_reranker_utils.py prompt, masked mean pooling, and yes/no readout.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time
import traceback

REVISION = "32b242bb7fd7ad6fd404154c09560d249ffe017f"
REPO = "KaLM-Embedding/KaLM-Reranker-V1-Nano-R2"
SHA = "c1d8ed4eee064e9c594e8332896c7ebe3104f892b6ff3f321b443b8868ed01b9"
DOCS = ["The capital of China is Beijing. Beijing is in northern China.",
        "植物通过光合作用利用阳光，将水和二氧化碳转化为糖，并释放氧气。"]
QUERIES = ["What is the capital of China?", "Where is Beijing located?",
           "植物通过什么过程利用阳光制造糖？", "What process lets plants use sunlight to make sugar?"]


def emit(event, **fields):
    print(json.dumps(dict(event=event, unix_time=time.time(), **fields), ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dtype", choices=["bf16"], required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    phase = {"name": "imports"}
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(5):
            emit("heartbeat", phase=phase["name"])
    threading.Thread(target=heartbeat, daemon=True).start()
    result = dict(status="running", repository=REPO, revision=REVISION,
                  hostname=platform.node(), command=os.sys.argv,
                  physical_device=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  rows=[], timing_scope="synchronized smoke, includes cold calls; not throughput benchmark")
    try:
        import torch
        import torch_npu  # noqa: F401
        import torch.nn.functional as F
        from transformers import AutoTokenizer, T5Gemma2ForConditionalGeneration
        from transformers.modeling_outputs import BaseModelOutput
        if not torch.npu.is_available():
            raise RuntimeError("NPU unavailable; no CPU fallback")
        torch.npu.set_device(0)
        torch.set_num_threads(8)
        result["versions"] = {k: importlib.metadata.version(k) for k in ["torch", "torch-npu", "transformers"]}
        result["device_name"] = torch.npu.get_device_name(0)
        free, total = torch.npu.mem_get_info()
        result["initial_memory"] = dict(free=free, total=total)
        if free < 12 * 1024**3:
            raise RuntimeError("Less than 12 GiB free; refusing run")
        phase["name"] = "verify_weights"
        h = hashlib.sha256()
        with (args.model / "model.safetensors").open("rb") as f:
            for block in iter(lambda: f.read(8 << 20), b""):
                h.update(block)
        result["weights_sha256"] = h.hexdigest()
        if h.hexdigest() != SHA:
            raise ValueError("Weights do not match pinned release")
        phase["name"] = "load_model"
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = T5Gemma2ForConditionalGeneration.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="eager").eval().to("npu:0")
        torch.npu.synchronize()
        result["load_s"] = time.perf_counter() - started
        result["parameters"] = sum(x.numel() for x in model.parameters())
        result["parameter_dtypes"] = sorted({str(x.dtype) for x in model.parameters()})
        if any(x.device.type != "npu" or x.dtype != torch.bfloat16 for x in model.parameters()):
            raise ValueError("Expected all parameters BF16 on NPU")
        activations = {}
        def hook(name):
            def record(module, inputs, output):
                if name not in activations and torch.is_tensor(output):
                    activations[name] = dict(dtype=str(output.dtype), device=str(output.device), shape=list(output.shape))
            return record
        handles = []
        for name, module in model.named_modules():
            if name.endswith("q_proj") and ("layers.0." in name):
                handles.append(module.register_forward_hook(hook(name)))
        ids = [tokenizer(x, add_special_tokens=False)["input_ids"] for x in ["yes", "no"]]
        if any(len(x) != 1 for x in ids):
            raise ValueError("yes/no must be single tokens")
        result["answer_token_ids"] = ids
        def timed(name, function):
            phase["name"] = name
            torch.npu.synchronize()
            start = time.perf_counter()
            output = function()
            torch.npu.synchronize()
            return output, time.perf_counter() - start
        def encode(batch):
            hidden = model.get_encoder()(**batch, return_dict=True).last_hidden_state
            mask = batch["attention_mask"]
            padding = (-hidden.shape[1]) % 4
            hidden = F.pad(hidden, (0, 0, 0, padding))
            mask = F.pad(mask, (0, padding))
            hidden = hidden.reshape(1, -1, 4, hidden.shape[-1])
            mask = mask.reshape(1, -1, 4)
            pooled = (hidden * mask.unsqueeze(-1).to(hidden.dtype)).sum(2)
            pooled = pooled / mask.sum(2).clamp(min=1).unsqueeze(-1)
            return pooled, (mask.sum(2) > 0).long()
        def score(query, encoded):
            hidden, mask = encoded
            prompt = ('<bos><start_of_turn>user\n'
                      'Judge whether the Document meets the requirements based on the Query and '
                      'the Instruct provided. Note that the answer can only be "yes" or "no".\n\n'
                      '<Instruct>: Given a query, retrieve documents that answer the query.\n'
                      f'<Query>: {query}<end_of_turn>\n<start_of_turn>model\n\n\n\n')
            batch = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("npu:0")
            out = model(encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                        attention_mask=mask, decoder_input_ids=batch.input_ids,
                        decoder_attention_mask=batch.attention_mask, use_cache=False,
                        logits_to_keep=1, return_dict=True)
            logits = out.logits[0, -1, [ids[0][0], ids[1][0]]].float()
            return logits, logits.softmax(-1)[0], batch.input_ids.shape[1]
        with torch.inference_mode():
            for di, document in enumerate(DOCS):
                batch = tokenizer('<Document>: ' + document, add_special_tokens=False,
                                  return_tensors="pt").to("npu:0")
                cached, encoder_s = timed(f"encode_document_{di}", lambda: encode(batch))
                for qi, query in enumerate(QUERIES):
                    reference, full_s = timed(f"full_{di}_{qi}", lambda: score(query, encode(batch)))
                    reused, reuse_s = timed(f"cached_{di}_{qi}", lambda: score(query, cached))
                    a, b = reference[0], reused[0]
                    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
                        raise ValueError("Nonfinite scores")
                    row = dict(document=di, query=qi, document_text=document, query_text=query,
                               document_tokens=batch.input_ids.shape[1], query_prompt_tokens=reused[2],
                               cached_shape=list(cached[0].shape), encoder_s=encoder_s,
                               full_s=full_s, cached_scoring_s=reuse_s,
                               full_score=float(reference[1]), cached_score=float(reused[1]),
                               max_logit_delta=float((a-b).abs().max()),
                               cache_parity=bool(torch.allclose(a, b, atol=1e-4, rtol=1e-4)))
                    result["rows"].append(row)
                    emit("item_finished", **row)
            result["expected_ranking"] = all(
                result["rows"][qi if qi < 2 else 4+qi]["cached_score"] >
                result["rows"][4+qi if qi < 2 else qi]["cached_score"] for qi in range(4))
        for handle in handles:
            handle.remove()
        result["activation_samples"] = activations
        result["cache_parity"] = all(r["cache_parity"] for r in result["rows"])
        result["peak_allocated_bytes"] = torch.npu.max_memory_allocated()
        if not result["cache_parity"] or not result["expected_ranking"]:
            raise AssertionError("Cache parity or simple relevance ranking failed")
        result["status"] = "passed"
    except Exception:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        stop.set()
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        emit("complete", status=result["status"], output=str(args.output))


if __name__ == "__main__":
    main()
