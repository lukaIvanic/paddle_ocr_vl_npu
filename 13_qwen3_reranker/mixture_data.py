"""Source-family filtering, stratified quotas and held-out text checks."""
import hashlib
import math
import random

from training_smoke_data import normalize

MIRROR = "hotchpotch/bge-m3-data-finetune-unified"
REVISION = "b51dd24cbce7d89255911410ef74a36e07bfbab9"
EXCLUDED = {
    "hotpotqa": "HotpotQAHardNegatives, pinned MTEB English v2",
    "msmarco": "Parent source of CMTEB MMarcoRetrieval",
    "mmarco_chinese": "CMTEB MMarcoRetrieval",
    "dureader": "CMTEB DuRetrieval",
    "t2ranking": "CMTEB T2Retrieval",
    "cMedQAv2": "CMTEB CmedqaRetrieval/MedicalRetrieval family",
}


def source_name(config):
    return config.split("_len-")[0]


def text_hash(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def quotas(total, weights, capacities=None, minimum=0):
    """Proportional integer allocation with explicit finite capacities."""
    capacities = capacities or {k: total for k in weights}
    result = {k: min(minimum, capacities[k]) for k in sorted(weights)}
    if sum(result.values()) > total or sum(capacities.values()) < total:
        raise ValueError("Infeasible allocation")
    while sum(result.values()) < total:
        remaining = total - sum(result.values())
        active = [k for k in result if result[k] < capacities[k]]
        denominator = sum(weights[k] for k in active)
        shares = {k: remaining * weights[k] / denominator for k in active}
        assigned = {k: min(capacities[k] - result[k], math.floor(shares[k])) for k in active}
        if not any(assigned.values()):
            for k in sorted(active, key=lambda k: (-(shares[k] % 1), k))[:remaining]:
                assigned[k] = 1
        for k, n in assigned.items():
            result[k] += n
    return result


def clean_row(row, config, row_index, blocked):
    if source_name(config) in EXCLUDED:
        raise ValueError("Excluded source family")
    q = row["query"]
    if not normalize(q) or text_hash(q) in blocked["query_hashes"]:
        return None
    positive_hashes = {text_hash(d) for d in row["pos"]}
    seen, positives, negatives = set(), [], []
    for label, docs in ((1, row["pos"]), (0, row["neg"])):
        for d in docs:
            h = text_hash(d)
            if not normalize(d) or h in seen or h in blocked["document_hashes"]:
                continue
            if not label and h in positive_hashes:
                continue
            seen.add(h)
            (positives if label else negatives).append(d)
    if not positives or not negatives:
        return None
    return {"id": f"{config}/{row_index}", "source": source_name(config),
            "config": config, "query": q, "positives": positives, "negatives": negatives}


def select_split(pool, counts, train_count, val_count, seed):
    rng = random.Random(seed)
    pool = list(pool)
    rng.shuffle(pool)
    unique, seen = [], set()
    for r in pool:
        h = text_hash(r["query"])
        if h not in seen:
            unique.append(r)
            seen.add(h)
    families = sorted({r["source"] for r in unique})
    by_family = {f: [r for r in unique if r["source"] == f] for f in families}
    weights = {f: sum(n for c, n in counts.items() if source_name(c) == f) for f in families}
    val_quota = quotas(val_count, weights, {f: len(by_family[f]) for f in families}, minimum=2)

    def choose(rows, n):
        configs = sorted({r["config"] for r in rows})
        q = quotas(n, {c: counts[c] for c in configs},
                   {c: sum(r["config"] == c for r in rows) for c in configs})
        out = []
        for r in rows:
            if q[r["config"]]:
                out.append(r)
                q[r["config"]] -= 1
        return out

    val = []
    for f in families:
        for r in choose(by_family[f], val_quota[f]):
            docs = [r["positives"][0]] + r["negatives"][:7]
            val.append({k: r[k] for k in ("id", "source", "config", "query")} |
                       {"documents": docs, "labels": [1] + [0] * (len(docs) - 1)})
    val_queries = {text_hash(r["query"]) for r in val}
    val_docs = {text_hash(d) for r in val for d in r["documents"]}
    available = {f: [] for f in families}
    for r in unique:
        if text_hash(r["query"]) in val_queries:
            continue
        pos = [d for d in r["positives"] if text_hash(d) not in val_docs]
        neg = [d for d in r["negatives"] if text_hash(d) not in val_docs]
        if pos and neg:
            available[r["source"]].append(r | {"positives": pos, "negatives": neg})
    tq = quotas(train_count, weights, {f: len(available[f]) for f in families}, minimum=4)
    train = []
    for f in families:
        for r in choose(available[f], tq[f]):
            train.append({k: r[k] for k in ("id", "source", "config", "query")} |
                         {"documents": [r["positives"][0], r["negatives"][0]], "labels": [1, 0]})
    rng.shuffle(train)
    rng.shuffle(val)
    assert not val_queries & {text_hash(r["query"]) for r in train}
    assert not val_docs & {text_hash(d) for r in train for d in r["documents"]}
    return {"train": train, "validation": val,
            "distribution": {f: {"released_rows": weights[f], "train_queries": tq[f],
                                  "validation_queries": val_quota[f]} for f in families},
            "checks": {"normalized_queries_disjoint": True, "normalized_documents_disjoint": True},
            "absent_eligible_sources": sorted({source_name(c) for c in counts} - set(families))}
