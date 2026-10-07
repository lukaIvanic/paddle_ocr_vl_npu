"""Query-group Margin-MSE math, exact score-gradient replay, and diagnostics."""
import math


def validate_teacher_inputs(teacher, canonical, student, order):
    """Validate reusable query-first targets while allowing an intentional swap."""
    assert teacher == canonical, 'Saved teacher query-first tokens or truncation differ'
    assert set(student) == set(canonical), 'Student sections differ'
    if order == 'query_first':
        assert student == canonical, 'Query-first student tokens differ'
    elif order == 'contents_swapped':
        for section in canonical:
            assert student[section]['pairs'] == canonical[section]['pairs'], 'Student pair counts differ'
            assert canonical[section]['truncated'] == student[section]['truncated'] == 0, (
                'This contents-swap pilot requires untruncated teacher and student inputs')
            assert student[section]['token_ids_sha256'] != canonical[section]['token_ids_sha256'], (
                'Contents swap did not change token IDs')
    else:
        raise ValueError(order)


def lr_at(step, steps, peak, schedule, warmup=5):
    if not 1 <= step <= steps:
        raise ValueError("Optimizer step outside schedule")
    if schedule == "constant":
        return peak
    if schedule != "warmup_linear" or not 0 < warmup < steps:
        raise ValueError("Invalid schedule")
    if step <= warmup:
        return peak * step / warmup
    # Keep the final update nonzero; decay reaches zero after the run.
    return peak * (steps - step + 1) / (steps - warmup)


def margin_loss_and_score_gradient(student, teacher):
    """Mean over all unordered candidate pairs; returns exact dL/dscore.

    For deterministic forwards, replaying each microbatch with these score
    gradients is equivalent to retaining every candidate's forward graph.
    This bounds activation memory even when a group contains eight long docs.
    """
    import torch
    if student.ndim != 1 or teacher.shape != student.shape or len(student) < 2:
        raise ValueError("Expected one equally sized candidate group")
    delta = student.float() - teacher.float()
    centered = delta - delta.mean()
    k = len(delta)
    loss = (2 * k / (k - 1)) * centered.square().mean()
    gradient = (4 / (k - 1)) * centered
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    return loss, gradient


def agreement(groups, margins, teacher):
    total, correct, squared, overlap = 0, 0.0, 0.0, 0
    for g in groups:
        s, t = margins[g['id']], teacher[g['id']]
        k = len(s)
        for i in range(k):
            for j in range(i + 1, k):
                sd, td = s[i] - s[j], t[i] - t[j]
                squared += (sd - td) ** 2
                if td != 0:
                    total += 1
                    correct += 0.5 if sd == 0 else float(sd * td > 0)
        sr = sorted(range(k), key=lambda i: (s[i], i), reverse=True)
        tr = sorted(range(k), key=lambda i: (t[i], i), reverse=True)
        overlap += len(set(sr[:3]) & set(tr[:3]))
    pairs = sum(len(g['documents']) * (len(g['documents']) - 1) // 2 for g in groups)
    return {'queries': len(groups), 'pairwise_agreement': correct / total if total else None,
            'teacher_nontied_comparisons': total, 'margin_mse': squared / pairs,
            'top3_overlap': overlap / (3 * len(groups))}


def benchmark_metrics(groups, predictions):
    from curve_metrics import ndcg10
    tasks = {}
    for g in groups:
        scores = dict(zip(g['document_ids'], predictions[g['id']]))
        m = ndcg10({g['qid']: scores}, {g['qid']: g['qrels']}, g['ignore_identical_ids'])
        ranked = sorted(scores, key=lambda d: (scores[d], d), reverse=True)
        ranked = [d for d in ranked if not (g['ignore_identical_ids'] and d == g['qid'])][:10]
        tasks.setdefault(g['task'], []).append({
            'qid': g['qid'], 'ndcg10': m['ndcg10'],
            'hole10': sum(d not in g['qrels'] for d in ranked) / len(ranked),
            'language': g['language']})
    per_task = {t: {'queries': len(rows),
                   'ndcg10': sum(r['ndcg10'] for r in rows) / len(rows),
                   'hole10': sum(r['hole10'] for r in rows) / len(rows),
                   'per_query': rows, 'language': rows[0]['language']} for t, rows in tasks.items()}
    suite = {lang: sum(v['ndcg10'] for v in per_task.values() if v['language'] == lang) /
                   sum(v['language'] == lang for v in per_task.values())
             for lang in {g['language'] for g in groups}}
    return {'per_task': per_task, 'suite_macro_ndcg10': suite,
            'scope': 'fixed query sample of pinned retrieval suites, not full benchmark scores'}
