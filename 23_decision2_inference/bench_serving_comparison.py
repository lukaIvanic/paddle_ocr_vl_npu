"""Real retrieval pairs -> native prompts -> HTTP serving -> scores, both models.

Uses intact saved CMTEB query/document pairs, no repeated-token synthetic input,
no isolated forwards, no gold-label filtering. Eos Noul and Qwen yes/no are NOT
assumed equivalent in quality. Paired content and length bands, not identical
token sequences: different model prompts/tokenizers are deliberately preserved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


def quantiles(values):
    s = sorted(values)
    def at(p):
        x = (len(s)-1)*p
        lo = int(x)
        return s[lo] + (s[min(lo+1,len(s)-1)]-s[lo])*(x-lo)
    return {"mean": statistics.mean(s), "p50": at(.5), "p95": at(.95), "p99": at(.99), "max": max(s)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workloads", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--reranker-model", default="/workspace/models/Qwen3-Reranker-4B")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--eos-url", default="http://127.0.0.1:18423")
    p.add_argument("--reranker-url", default="http://127.0.0.1:18330")
    p.add_argument("--pairs-per-band", type=int, default=16)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--contention", choices=["T2_active", "dedicated_NPU7"], required=True)
    p.add_argument("--models", nargs='+', choices=["eos","reranker"], default=["eos"],
                   help="Eos-only by default; reuse saved reranker speed measurements")
    p.add_argument("--concurrencies", nargs='+', type=int, default=[1,4])
    p.add_argument("--dataset-workloads", nargs='+', choices=["EcomRetrieval", "CmedqaRetrieval"],
                   help="Use all saved pairs to compare with historical 4B timings, instead of length-band samples")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    manifest = {"command": sys.argv, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "chip": "910B2", "physical_device": 7, "contention": args.contention,
                "workload_sha256": hashlib.sha256(args.workloads.read_bytes()).hexdigest(),
                "scope": "Native relevance prompts on paired intact documents, not an accuracy evaluation",
                "timing": "client tokenization + HTTP queue/inference + response readout, wall clock; warmup excluded",
                "config": {"eos": {"dtype":"bf16", "max_num_seqs":4, "max_num_batched_tokens":2048},
                           "reranker": {"dtype":"fp16", "max_num_seqs":32, "max_num_batched_tokens":16384}}}
    save("manifest.json", manifest)
    state = {"phase": "imports"}
    stop = threading.Event()
    def emit(event, **values):
        print(json.dumps({"event": event, "time":time.time(), **values}), flush=True)
    def heartbeat():
        while not stop.wait(5):
            emit("heartbeat", **state)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    sys.path.insert(0, str(args.bundle))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"22_qwen3_embedding_benchmark"))
    from transformers import AutoTokenizer
    from decision2._vendor.dev2model.decision_model import encode
    from decision2._vendor.dev2model.infer import question_to_row, product_answer
    from reranker_protocol import tokenize_pairs
    from protocol import TASKS
    eos_tok = AutoTokenizer.from_pretrained(str(args.bundle), local_files_only=True)
    rr_tok = AutoTokenizer.from_pretrained(args.reranker_model, local_files_only=True)

    def prepare(model, pair):
        if model == "reranker":
            ids, truncated = tokenize_pairs(rr_tok, pair["task"], [pair])
            assert not truncated
            return {"model":"reranker-diagnostic", "input": ids[0], "task":"classify",
                    "use_activation":True, "encoding_format":"float", "add_special_tokens":False}, None
        item = {"id":pair["qid"]+"/"+pair["did"], "state":{"query":pair["query"], "document":pair["document"]}}
        question = {"type":"noul", "instructions": "Does the document satisfy this retrieval instruction for the query? " + TASKS[pair["task"]][2]}
        row = question_to_row(item, "relevance", question)
        encoded = encode(row, eos_tok, 2048)
        return {"model":"eos-0.8b", "input": encoded["ids"], "task":"classify", "use_activation":False,
                "encoding_format":"float", "add_special_tokens":False,
                "decision2":{"candidate_positions":encoded["candidate_positions"],
                             "query_position":encoded["query_position"], "token_count":len(encoded["ids"])}}, (row,encoded)

    state["phase"] = "select_paired_real_workload"
    pool = [{**pair,"task":w["task"]} for w in json.loads(args.workloads.read_text()) for pair in w["pairs"]]
    random.Random(20261003).shuffle(pool)
    bands = {"short_64_256": (64,256), "medium_256_768": (256,768), "long_768_1536": (768,1536)}
    selected = {name:[] for name in bands}
    for pair in pool:
        try:
            lengths = {m:len(prepare(m,pair)[0]["input"]) for m in ("eos","reranker")}
        except ValueError as e:
            if "exceeds max_length" in str(e):
                continue
            raise
        for name,(lo,hi) in bands.items():
            if len(selected[name]) < args.pairs_per_band and all(lo <= n < hi for n in lengths.values()):
                selected[name].append({**pair,"native_lengths":lengths})
        if all(len(rows)==args.pairs_per_band for rows in selected.values()):
            break
    assert all(len(rows)==args.pairs_per_band for rows in selected.values()), {k:len(v) for k,v in selected.items()}
    if args.dataset_workloads:
        selected = {w['task']:[{**pair,'task':w['task']} for pair in w['pairs']]
                    for w in json.loads(args.workloads.read_text()) if w['task'] in args.dataset_workloads}
    save("selected_pairs.json", selected)
    emit("workload_ready", counts={k:len(v) for k,v in selected.items()})
    log = (args.output/"requests.jsonl").open("w")
    trials = []
    try:
        for concurrency in args.concurrencies:
            for band, pairs in selected.items():
                # Full matched group warmup at the tested concurrency; record
                # separately so first-use kernel compilation cannot inflate speed.
                for repeat in range(-1,args.repeats):
                    order = args.models if repeat % 2 else list(reversed(args.models))
                    for model in order:
                        state.update(phase="warmup" if repeat<0 else "measured",model=model,band=band,concurrency=concurrency,repeat=repeat,completed=0)
                        context = dict(state)
                        endpoint = (args.eos_url if model=="eos" else args.reranker_url)+"/pooling"
                        def one(pair):
                            start = time.perf_counter()
                            payload, native = prepare(model,pair)
                            prepared = time.perf_counter()
                            request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
                            try:
                                with urllib.request.urlopen(request, timeout=120) as response:
                                    data = json.load(response)
                            except urllib.error.HTTPError as e:
                                raise RuntimeError(e.read().decode()) from e
                            received = time.perf_counter()
                            tokens = len(payload["input"])
                            assert data["usage"]["prompt_tokens"] == tokens
                            logits = data["data"][0]["data"]
                            if model=="eos":
                                row, encoded = native
                                score = product_answer("noul",encoded["keys"],logits,1.0,[o["description"] for o in row["options"]])["noul"]
                            else:
                                assert len(logits)==1
                                score=logits[0]
                            assert math.isfinite(score) and 0<=score<=1
                            return {**context,"task":pair["task"],"qid":pair["qid"],"did":pair["did"],"tokens":tokens,"score":score,
                                    "tokenize_s":prepared-start,"http_s":received-prepared,"wall_s":time.perf_counter()-start}
                        started = time.perf_counter()
                        records=[]
                        with ThreadPoolExecutor(max_workers=concurrency) as executor:
                            trial_pairs = pairs[:max(16,concurrency)] if repeat<0 else pairs
                            for future in as_completed([executor.submit(one,pair) for pair in trial_pairs]):
                                record=future.result()
                                records.append(record)
                                log.write(json.dumps(record)+"\n")
                                log.flush()
                                state["completed"]=len(records)
                                emit("item_finished",**record)
                        wall = time.perf_counter()-started
                        tokens=sum(r["tokens"] for r in records)
                        summary={**context,"pairs":len(records),"tokens":tokens,"mean_tokens":tokens/len(records),"wall_s":wall,
                                 "pairs_s":len(records)/wall,"input_tok_s":tokens/wall,
                                 "item_latency_s":quantiles([r["wall_s"] for r in records]),"tokenize_s":sum(r["tokenize_s"] for r in records)}
                        trials.append(summary)
                        save("results.json",trials)
                        emit("trial_finished",**summary)
        (args.output/"exit_code.txt").write_text("0\n")
    finally:
        log.close()
        stop.set()
        thread.join()


if __name__ == "__main__":
    main()
