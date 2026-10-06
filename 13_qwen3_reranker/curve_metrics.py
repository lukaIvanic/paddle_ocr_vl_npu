"""Fixed-candidate Touché metrics and mixed-source validation summaries."""
import math

from training_smoke_data import ranking_metrics


def ndcg10(scores, qrels, ignore_self=False):
    per_query = {}
    for qid, documents in scores.items():
        # trec_eval breaks equal-score ties by descending document ID.
        ranked = sorted((d for d in documents if not (ignore_self and d == qid)),
                        key=lambda d: (documents[d], d), reverse=True)[:10]
        ideal = sorted(qrels[qid].values(), reverse=True)[:10]
        dcg = sum(qrels[qid].get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked))
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        per_query[qid] = dcg / idcg if idcg else 0.0
    return {"ndcg10": sum(per_query.values()) / len(per_query), "per_query": per_query}


def validation_metrics(groups, logits):
    by_source = {}
    offset = 0
    loss = 0.0
    for g in groups:
        rows = logits[offset:offset + len(g["documents"])]
        offset += len(rows)
        for label, (no, yes) in zip(g["labels"], rows):
            margin = yes - no
            loss += math.log1p(math.exp(-abs(margin))) + max(margin, 0) - label * margin
        entry = by_source.setdefault(g["source"], {"groups": [], "logits": []})
        entry["groups"].append(g)
        entry["logits"].extend(rows)
    sources = {s: ranking_metrics(v["groups"], v["logits"]) for s, v in by_source.items()}
    return {"overall": ranking_metrics(groups, logits), "per_source": sources,
            "pointwise_cross_entropy": loss / len(logits),
            "source_macro_ordering_accuracy": sum(v["ordering_accuracy"] for v in sources.values()) / len(sources)}
