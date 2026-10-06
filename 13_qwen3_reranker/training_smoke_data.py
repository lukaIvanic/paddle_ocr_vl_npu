"""Deterministic source allowlist and disjoint splits for the training smoke."""
import hashlib
import random
import unicodedata

ALLOWED_SOURCE = "pubmed_qa_labeled_len-0-500"


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def body(instruction, query, document, order):
    fields = [f"<Query>: {query}", f"<Document>: {document}"]
    if order == "document_first":
        fields.reverse()
    elif order != "query_first":
        raise ValueError(order)
    return "\n".join([f"<Instruct>: {instruction}"] + fields)


def select_groups(source, fits, train_count=256, eval_count=32, seed=731):
    if source["provenance"]["source_allowlist"] != [ALLOWED_SOURCE]:
        raise ValueError("Unexpected source allowlist")
    rows = list(source["rows"])
    random.Random(seed).shuffle(rows)
    seen_queries, candidates = set(), []
    for row in rows:
        if row["source_config"] != ALLOWED_SOURCE:
            raise ValueError("Disallowed training source")
        q = normalize(row["query"])
        if not q or q in seen_queries:
            continue
        seen_queries.add(q)
        positives, negatives, seen_docs = [], [], set()
        positive_texts = {normalize(d) for d in row["pos"]}
        for label, documents in ((1, row["pos"]), (0, row["neg"])):
            for document in documents:
                d = normalize(document)
                if not d or d in seen_docs or (not label and d in positive_texts):
                    continue
                if not fits(row["query"], document):
                    continue
                seen_docs.add(d)
                (positives if label else negatives).append(document)
        if positives and len(negatives) >= 3:
            candidates.append({"id": f'{row["source_config"]}/{row["source_row_index"]}',
                               "query": row["query"], "positives": positives,
                               "negatives": negatives})
    if len(candidates) < eval_count:
        raise ValueError("Insufficient eligible evaluation queries")
    evaluation = [{"id": r["id"], "query": r["query"],
                   "documents": [r["positives"][0]] + r["negatives"][:3],
                   "labels": [1, 0, 0, 0]} for r in candidates[:eval_count]]
    eval_docs = {normalize(d) for r in evaluation for d in r["documents"]}
    eval_queries = {normalize(r["query"]) for r in evaluation}
    training = []
    for r in candidates[eval_count:]:
        pos = [d for d in r["positives"] if normalize(d) not in eval_docs]
        neg = [d for d in r["negatives"] if normalize(d) not in eval_docs]
        if pos and neg and normalize(r["query"]) not in eval_queries:
            training.append({"id": r["id"], "query": r["query"],
                             "documents": [pos[0], neg[0]], "labels": [1, 0]})
        if len(training) == train_count:
            break
    if len(training) != train_count:
        raise ValueError(f"Only {len(training)} disjoint training queries available")
    assert not eval_queries & {normalize(r["query"]) for r in training}
    assert not eval_docs & {normalize(d) for r in training for d in r["documents"]}
    return {"provenance": source["provenance"], "seed": seed,
            "train": training, "eval": evaluation,
            "checks": {"query_disjoint": True, "document_disjoint": True,
                       "eligible_source_queries": len(candidates)}}


def ranking_metrics(groups, logits):
    import math
    wins = ties = top1 = 0
    mrr = ndcg = 0.0
    offset = 0
    for group in groups:
        rows = logits[offset:offset + len(group["documents"])]
        offset += len(rows)
        margins = [yes - no for no, yes in rows]
        wins += sum(margins[0] > x for x in margins[1:])
        ties += sum(margins[0] == x for x in margins[1:])
        # Stable original index breaks ties; report ties separately.
        ranking = sorted(range(len(rows)), key=lambda i: (-margins[i], i))
        rank = ranking.index(0) + 1
        top1 += rank == 1
        mrr += 1 / rank
        ndcg += 1 / math.log2(rank + 1)
    n = len(groups)
    return {"queries": n, "pairs": offset, "correct_orderings": wins,
            "total_orderings": offset - n, "ties": ties,
            "ordering_accuracy": wins / (offset - n), "top1_accuracy": top1 / n,
            "mrr": mrr / n, "ndcg": ndcg / n}
