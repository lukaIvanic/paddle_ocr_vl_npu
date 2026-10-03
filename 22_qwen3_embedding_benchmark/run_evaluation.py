#!/usr/bin/env python3
"""Pinned MTEB evaluator -> localhost vLLM-Ascend /embeddings, real full corpora.

Use --prepare-only to fetch pinned data without starting inference. Run output
directories must be new: never silently mix partial runs or model revisions.
All ranking/evaluation remains the original MTEB implementation. No gold injection.
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
import urllib.request

from protocol import DIMENSIONS, MAX_LENGTH, MODEL_ID, MODEL_REVISION, TASKS, format_text, validate_task


def emit(event, **fields):
    print(json.dumps({"event": event, "time": time.time(), **fields}, ensure_ascii=False), flush=True)


class Observer:
    def __init__(self):
        self.state = {"section": "initializing"}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)

    def heartbeat(self):
        while not self.stop.wait(5):
            emit("heartbeat", **self.state)


class EndpointEncoder:
    def __init__(self, args, observer):
        import numpy as np
        from transformers import AutoTokenizer
        from mteb.model_meta import ModelMeta
        self.np = np
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        self.tokenizer.padding_side = "left"
        self.args, self.observer = args, observer
        self.mteb_model_meta = ModelMeta(
            name="local/Qwen3-Embedding-0.6B-vllm-ascend-fp16", revision=MODEL_REVISION,
            release_date=None, languages=None, n_parameters=None, memory_usage_mb=None,
            max_tokens=MAX_LENGTH, embed_dim=DIMENSIONS, license="apache-2.0",
            open_weights=True, public_training_code=False, public_training_data=False,
            framework=["vLLM"], similarity_fn_name="cosine", use_instructions=True,
            training_datasets=None,
        )
        self.totals = {"texts": 0, "tokens": 0, "truncated_texts": 0, "request_wall_s": 0.0}
        self.log = (args.output / "encoding_batches.jsonl").open("w")

    def encode(self, sentences, *, task_name, prompt_type=None, batch_size=128, **kwargs):
        role = getattr(prompt_type, "value", prompt_type)
        rows = []
        for offset in range(0, len(sentences), self.args.batch_size):
            texts = [format_text(t, task_name, role)
                     for t in sentences[offset:offset + self.args.batch_size]]
            self.observer.state = {"task": task_name, "section": "tokenizing", "role": role,
                                   "offset": offset, "total": len(sentences)}
            # No chat template, automatic instruction, or extra EOS. Match HF tokenizer call.
            tokens = self.tokenizer(texts, add_special_tokens=True, padding=False,
                                    truncation=False)["input_ids"]
            truncated = sum(len(t) > MAX_LENGTH for t in tokens)
            tokens = [t[:MAX_LENGTH] for t in tokens]
            if any(not t for t in tokens):
                raise ValueError("Empty tokenized input")
            payload = {"model": "qwen3-embedding-0.6b", "input": tokens,
                       "encoding_format": "float"}
            req = urllib.request.Request(self.args.endpoint + "/embeddings",
                data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            self.observer.state["section"] = "vllm_request"
            start = time.monotonic()
            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.load(response)
            seconds = time.monotonic() - start
            records = sorted(result["data"], key=lambda x: x["index"])
            if [r["index"] for r in records] != list(range(len(tokens))):
                raise ValueError("Missing or reordered output indices")
            embeddings = self.np.asarray([r["embedding"] for r in records], dtype=self.np.float32)
            if embeddings.shape != (len(tokens), DIMENSIONS) or not self.np.isfinite(embeddings).all():
                raise ValueError("Invalid embedding dimensions or nonfinite values")
            norm_error = float(self.np.abs(self.np.linalg.norm(embeddings, axis=1) - 1).max())
            if norm_error > 0.005:
                raise ValueError(f"Embeddings were not unit-normalized: {norm_error}")
            lengths = [len(t) for t in tokens]
            record = {"task": task_name, "role": role, "offset": offset,
                      "completed": offset + len(tokens), "total": len(sentences),
                      "texts": len(tokens), "tokens": sum(lengths), "token_lengths": lengths,
                      "truncated_texts": truncated, "request_wall_s": seconds,
                      "request_tok_s": sum(lengths) / seconds, "max_norm_error": norm_error}
            self.log.write(json.dumps(record) + "\n")
            self.log.flush()
            emit("encoding_batch_finished", **{k:v for k,v in record.items() if k != "token_lengths"})
            for key in self.totals:
                self.totals[key] += record[key]
            rows.append(embeddings)
        self.observer.state = {"task": task_name, "section": "retrieval_or_scoring"}
        return self.np.concatenate(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/workspace/models/Qwen3-Embedding-0.6B")
    p.add_argument("--endpoint", default="http://127.0.0.1:18222/v1")
    p.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output directory exists; choose a fresh run directory")
    args.output.mkdir(parents=True)
    observer = Observer()
    observer.thread.start()
    started = time.monotonic()
    try:
        import mteb
        import torch
        if importlib.metadata.version("mteb") != "1.38.9":
            raise RuntimeError("Use the pinned evaluator environment")
        torch.set_num_threads(8)
        manifest = {"chip": "Ascend 910B2", "hostname": platform.node(),
                    "command": __import__("sys").argv,
                    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "packages": {n: importlib.metadata.version(n) for n in
                                 ["mteb", "datasets", "transformers", "torch", "sentence-transformers"]},
                    "model_id": MODEL_ID, "reference_revision": MODEL_REVISION,
                    "max_length": MAX_LENGTH, "dimensions": DIMENSIONS,
                    "tasks": {n:TASKS[n] for n in args.tasks}, "split": "dev",
                    "query_template": "Instruct: {task_instruction}\nQuery:{text}",
                    "document_template": "{text}", "search": "exact cosine", "top_k": 100,
                    "inference_backend": "vLLM-Ascend localhost endpoint", "dtype": "fp16",
                    "endpoint": args.endpoint, "prepare_only": args.prepare_only}
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        encoder = None if args.prepare_only else EndpointEncoder(args, observer)
        summaries = []
        for name in args.tasks:
            observer.state = {"section": "loading_dataset", "task": name}
            task = mteb.get_tasks(tasks=[name])[0]
            validate_task(task)
            task.load_data()
            counts = {"documents": len(task.corpus["dev"]), "queries": len(task.queries["dev"]),
                      "judged_queries": len(task.relevant_docs["dev"])}
            emit("dataset_ready", task=name, **counts)
            if args.prepare_only:
                summaries.append({"task": name, **counts})
                continue
            begin = time.monotonic()
            observer.state = {"section": "evaluation", "task": name}
            result = mteb.MTEB(tasks=[task]).run(
                encoder, output_folder=str(args.output / "mteb"), eval_splits=["dev"],
                encode_kwargs={"batch_size": args.batch_size}, corpus_chunk_size=4096,
                top_k=100, save_predictions=True, raise_error=True,
            )
            values = result[0].scores["dev"]
            if len(values) != 1:
                raise ValueError("Unexpected Chinese subset count")
            score = values[0]
            row = {"task": name, **counts, "wall_s": time.monotonic() - begin,
                   "ndcg_at_10": score["ndcg_at_10"], "recall_at_100": score["recall_at_100"],
                   "reference_ndcg_at_10": TASKS[name][3],
                   "delta_points": 100 * (score["ndcg_at_10"] - TASKS[name][3])}
            summaries.append(row)
            (args.output / "completed_tasks.json").write_text(json.dumps(summaries, indent=2) + "\n")
            emit("task_finished", **row)
            # Sanity gate is a discrepancy report, not arbitrary permission to change the protocol.
            if abs(row["delta_points"]) > 1.0:
                raise RuntimeError(f"Investigate >1-point reference discrepancy before continuing: {row}")
        report = {"complete": True, "full_cmteb_r": set(args.tasks) == set(TASKS),
                  "prepare_only": args.prepare_only, "tasks": summaries,
                  "wall_s": time.monotonic() - started}
        if encoder:
            report["mean_ndcg_at_10"] = sum(r["ndcg_at_10"] for r in summaries) / len(summaries)
            report["encoding_totals"] = encoder.totals
            encoder.log.close()
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        emit("run_finished", **report)
    except BaseException as exc:
        (args.output / "failure.json").write_text(json.dumps({"error": repr(exc), "state": observer.state}, indent=2))
        raise
    finally:
        observer.stop.set()


if __name__ == "__main__":
    main()

