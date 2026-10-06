"""Acquire a bounded, source/length-stratified BGE-M3 sample from a pinned mirror."""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path
import random
import threading
import time
import urllib.parse
import urllib.request
import urllib.error

from mixture_data import MIRROR, REVISION, EXCLUDED, clean_row, quotas, select_split, source_name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--blocked", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-queries", type=int, default=6000)
    p.add_argument("--val-queries", type=int, default=384)
    p.add_argument("--seed", type=int, default=731)
    p.add_argument("--cache", type=Path, default=Path("/tmp/qwen-mixture-http-cache"))
    p.add_argument("--prefer-parquet", action="store_true")
    p.add_argument("--hub-endpoint", default="https://huggingface.co")
    args = p.parse_args()
    started = time.monotonic()
    info = json.loads(args.metadata.read_text())
    assert info["sha"] == REVISION
    counts = {c["config_name"]: sum(s["num_examples"] for s in c["splits"])
              for c in info["cardData"]["dataset_info"]
              if source_name(c["config_name"]) not in EXCLUDED}
    blocked_meta = json.loads(gzip.decompress(args.blocked.read_bytes()))
    blocked = {k: set(blocked_meta[k]) for k in ("query_hashes", "document_hashes")}
    # Sampling contiguous 100-row blocks avoids downloading entire large shards.
    # Blocks are sampled uniformly without replacement, then rows are shuffled.
    families = sorted({source_name(c) for c in counts})
    family_target = quotas(2 * (args.train_queries + args.val_queries),
                           {f: sum(n for c, n in counts.items() if source_name(c) == f) for f in families},
                           minimum=16)
    target = {}
    for f in families:
        weights = {c: n for c, n in counts.items() if source_name(c) == f}
        target.update(quotas(family_target[f], weights, weights))
    jobs = []
    rng = random.Random(args.seed)
    for config, count in sorted(counts.items()):
        if not target[config]:
            continue
        pages = list(range(0, count, 100))
        rng.shuffle(pages)
        pages = pages[:min(len(pages), max(1, (target[config] + 99) // 100))]
        remaining = target[config]
        for base in pages:
            block_size = min(100, count - base)
            length = min(remaining, block_size)
            if not length:
                break
            # A circular random window gives every row in the sampled block
            # equal inclusion probability, including its beginning and end.
            window_rng = random.Random(f"{args.seed}/{config}/{base}")
            start = window_rng.randrange(block_size) if length < block_size else 0
            first = min(length, block_size - start)
            jobs.append((config, base + start, first, base, block_size))
            if first < length:
                jobs.append((config, base, length - first, base, block_size))
            remaining -= length
    print("ACQUISITION_PLAN", json.dumps({"requests": len(jobs), "configs": len(counts),
          "source_rows": sum(counts.values()), "excluded": EXCLUDED}), flush=True)
    responses, pool = [], []
    args.cache.mkdir(parents=True, exist_ok=True)
    request_lock = threading.Lock()
    last_request = [0.0]
    parquet_lock = threading.Lock()
    parquet_files = {}

    def parquet_rows(config, offset, length):
        import fsspec
        import pyarrow.parquet as pq
        with parquet_lock:
            if config not in parquet_files:
                files = [s["rfilename"] for s in info["siblings"]
                         if s["rfilename"].startswith(config + "/") and s["rfilename"].endswith(".parquet")]
                assert len(files) == 1, "Expected one mirror Parquet shard per config"
                url = f"{args.hub_endpoint.rstrip('/')}/datasets/{MIRROR}/resolve/{REVISION}/{files[0]}?download=true"
                stream = fsspec.open(url, mode="rb", block_size=1024**2, client_kwargs={"trust_env": True}).open()
                parquet_files[config] = (pq.ParquetFile(stream), url)
            reader, url = parquet_files[config]
            rows, base = [], 0
            for rg in range(reader.num_row_groups):
                n = reader.metadata.row_group(rg).num_rows
                lo, hi = max(offset, base), min(offset + length, base + n)
                if lo < hi:
                    rows.extend(reader.read_row_group(rg).slice(lo - base, hi - lo).to_pylist())
                base += n
                if base >= offset + length:
                    break
        return {"rows": [{"row_idx": offset + i, "row": r} for i, r in enumerate(rows)]}, url

    def fetch(job):
        config, offset, length, base, block_size = job
        url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
            dict(dataset=MIRROR, config=config, split="train", offset=offset, length=length))
        t = time.monotonic()
        cached = args.cache / (hashlib.sha256(url.encode()).hexdigest() + ".json.gz")
        whole_url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
            dict(dataset=MIRROR, config=config, split="train", offset=base, length=block_size))
        whole_cached = args.cache / (hashlib.sha256(whole_url.encode()).hexdigest() + ".json.gz")
        source_url = url
        for attempt in range(8):
            try:
                if cached.exists():
                    raw = gzip.decompress(cached.read_bytes())
                elif whole_cached.exists():
                    raw = gzip.decompress(whole_cached.read_bytes())
                    source_url = whole_url
                elif args.prefer_parquet:
                    print("PARQUET_READ", json.dumps({"config": config, "offset": offset, "rows": length}), flush=True)
                    data, parquet_url = parquet_rows(config, offset, length)
                    data["acquisition_parquet_url"] = parquet_url
                    raw = json.dumps(data, ensure_ascii=False).encode()
                    cached.write_bytes(gzip.compress(raw))
                else:
                    with request_lock:
                        time.sleep(max(0, 1.2 - (time.monotonic() - last_request[0])))
                        last_request[0] = time.monotonic()
                    raw = urllib.request.urlopen(url, timeout=40).read()
                    cached.write_bytes(gzip.compress(raw))
                data = json.loads(raw)
                assert not data.get("partial")
                break
            except Exception as e:
                if isinstance(e, urllib.error.HTTPError) and (e.code == 501 or (e.code == 429 and attempt >= 2)):
                    print("PARQUET_FALLBACK", json.dumps({"config": config, "offset": offset}), flush=True)
                    data, parquet_url = parquet_rows(config, offset, length)
                    raw = json.dumps(data, ensure_ascii=False).encode()
                    data["acquisition_parquet_url"] = parquet_url
                    raw = json.dumps(data, ensure_ascii=False).encode()
                    cached.write_bytes(gzip.compress(raw))
                    break
                if attempt == 7:
                    raise
                delay = min(60, 5 * 2**attempt)
                print("FETCH_RETRY", json.dumps({"config": config, "offset": offset,
                      "error": str(e), "delay_seconds": delay}), flush=True)
                time.sleep(delay)
        valid = [clean_row(r["row"], config, r["row_idx"], blocked)
                 for r in data["rows"] if offset <= r["row_idx"] < offset + length and not r.get("truncated_cells")]
        return {"config": config, "offset": offset, "length": length, "url": source_url,
                "parquet_fallback": data.get("acquisition_parquet_url"),
                "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                "seconds": time.monotonic() - t}, [r for r in valid if r]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Ordered map makes the resulting data independent of response timing.
        for i, (meta, rows) in enumerate(executor.map(fetch, jobs), 1):
            responses.append(meta)
            pool.extend(rows)
            if i % 10 == 0 or i == len(jobs):
                print("FETCH_PROGRESS", json.dumps({"requests": i, "of": len(jobs),
                      "eligible_rows": len(pool), "seconds": time.monotonic() - started}), flush=True)
    fresh = json.load(urllib.request.urlopen(args.hub_endpoint.rstrip('/') + "/api/datasets/" + MIRROR, timeout=30))
    assert fresh["sha"] == REVISION, "Mirror revision changed during acquisition"
    dataset = select_split(pool, counts, args.train_queries, args.val_queries, args.seed)
    dataset["provenance"] = {
        "original_dataset": "Shitao/bge-m3-data", "mirror_dataset": MIRROR,
        "mirror_revision": REVISION, "mirror_claim": "format-only repack, not independently verified",
        "benchmark_contracts": ["MTEB(eng, v2) retrieval", "MTEB(cmn, v1) retrieval"],
        "benchmark_mteb_version": "1.38.9", "excluded_families": EXCLUDED,
        "touche_exclusion": blocked_meta,
        "filter_scope": "whole dataset-family exclusion plus exact normalized Touché text exclusion; not exhaustive cross-benchmark corpus-text decontamination",
        "sampling": "proportional released-row source and length quotas; source floors 4 train/2 validation; random blocks without replacement with uniform circular windows inside partial blocks; quota caps reflect eligible unique rows",
        "acquisition_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prefer_parquet": args.prefer_parquet,
        "hub_endpoint": args.hub_endpoint,
        "counts_by_config": counts, "responses": responses, "seed": args.seed,
        "seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress((json.dumps(dataset, ensure_ascii=False) + "\n").encode(), mtime=0))
    print("ACQUIRED", json.dumps({"train_queries": len(dataset["train"]),
          "validation_queries": len(dataset["validation"]), "sources": len(dataset["distribution"]),
          "absent_sources": dataset["absent_eligible_sources"], "bytes": args.output.stat().st_size,
          "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
          "seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
