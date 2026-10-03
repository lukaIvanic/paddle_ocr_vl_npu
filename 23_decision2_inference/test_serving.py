"""Use actual HTTP serving, original Eos encoding/readout, and saved HF answers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import time
import urllib.request
import urllib.error


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:18423")
    p.add_argument("--served-model", default="eos-0.8b")
    p.add_argument("--cases", type=Path, default=Path(__file__).with_name('smoke_cases.json'))
    p.add_argument("--max-length", type=int, default=2048)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    sys.path.insert(0, str(a.bundle))
    from transformers import AutoTokenizer
    from decision2._vendor.dev2model.decision_model import encode
    from decision2._vendor.dev2model.infer import question_to_row, product_answer
    tokenizer = AutoTokenizer.from_pretrained(str(a.bundle), local_files_only=True)
    cases = json.loads(a.cases.read_text())
    reference = {r["id"]: r["response"]["answers"] for r in json.loads(a.reference.read_text())["rows"] if not r["warmup"]}
    requests = []
    for case in cases:
        for key, question in case["questions"].items():
            row = question_to_row(case, key, question)
            encoded = encode(row, tokenizer, a.max_length)
            requests.append((case["id"], key, row, encoded))

    def run(item):
        cid, key, row, encoded = item
        payload = {"model": a.served_model, "task": "classify", "input": encoded["ids"],
                   "add_special_tokens": False, "use_activation": False,
                   "encoding_format": "float", "decision2": {
                       "candidate_positions": encoded["candidate_positions"],
                       "query_position": encoded["query_position"], "token_count": len(encoded["ids"])}}
        started = time.perf_counter()
        request = urllib.request.Request(a.url + "/pooling", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.load(response)
        logits = data["data"][0]["data"]
        assert len(logits) == len(encoded["keys"]) and all(math.isfinite(x) for x in logits)
        assert data["usage"]["prompt_tokens"] == len(encoded["ids"])
        answer = product_answer(row["task_type"], encoded["keys"], logits, 1.0,
                                [o["description"] for o in row["options"]])
        old = reference[cid][key]
        if row["task_type"] == "noul":
            delta = abs(answer["noul"] - old["noul"])
        else:
            delta = max(abs(answer["probabilities"][k] - old["probabilities"][k]) for k in encoded["keys"])
        result = {"id": cid, "question": key, "tokens": len(encoded["ids"]),
                  "token_ids_sha256": encoded["token_ids_sha256"], "wall_s": time.perf_counter()-started,
                  "logits": logits, "answer": answer, "reference_answer": old,
                  "max_probability_delta": delta}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result

    sequential = [run(r) for r in requests]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(run, list(reversed(requests)) * 2))
    by_key = {(r["id"], r["question"]): r for r in sequential}
    drift = max(max(abs(x-y) for x,y in zip(r["logits"], by_key[r["id"],r["question"]]["logits"])) for r in concurrent)
    observations = {
        "english_returns": by_key["english_returns", "route"]["answer"]["choice"] == "returns",
        "chinese_returns": by_key["chinese_returns", "route"]["answer"]["choice"] == "returns",
        "relevance_order": by_key["relevance_positive", "relevance"]["answer"]["score"] > by_key["relevance_negative", "relevance"]["answer"]["score"],
        "concurrent_choices_stable": all(r["answer"].get("choice") == by_key[r["id"],r["question"]]["answer"].get("choice") for r in concurrent),
    }
    invalid_responses = []
    for endpoint, payload in [
        ("/pooling", {"model": "eos-0.8b", "task": "classify", "input": [1,2,3], "use_activation": False}),
        ("/pooling", {"model": "eos-0.8b", "task": "classify", "input": [1,2,3], "use_activation": False,
                      "decision2": {"candidate_positions": [0,99], "query_position": 2, "token_count": 3}}),
        ("/classify", {"model": "eos-0.8b", "input": "This endpoint does not carry Eos option positions."}),
    ]:
        payload['model'] = a.served_model
        request = urllib.request.Request(a.url + endpoint, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                invalid_responses.append({"endpoint": endpoint, "status": response.status})
        except urllib.error.HTTPError as e:
            invalid_responses.append({"endpoint": endpoint, "status": e.code, "body": e.read().decode()})
    with urllib.request.urlopen(a.url + "/health", timeout=10) as response:
        observations["healthy_after_invalid_requests"] = response.status == 200
    observations["invalid_requests_rejected_before_worker"] = all(r["status"] == 400 for r in invalid_responses)
    result = {"chip": "910B2", "mode": "vllm-ascend-http", "concurrent_workload": "T2",
              "sequential": sequential, "concurrent": concurrent,
              "sanity_observations": observations,
              "invalid_responses": invalid_responses,
              "concurrency_max_logit_delta": drift,
              "max_reference_probability_delta": max(r["max_probability_delta"] for r in sequential)}
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("sequential", "concurrent")}), flush=True)
    assert all(observations.values()), "Inspect unexpected decisions; numerical deltas alone are not an accuracy benchmark"


if __name__ == "__main__":
    main()
