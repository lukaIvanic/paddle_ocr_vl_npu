"""Accumulate independent eight-candidate objectives across input orderings."""
from bge_baseline import loss_and_score_gradient
from distill_runtime import plans

ORDERS = ('query_first', 'document_first')


def backward_window(runtime, model, window, by_order, target_for_group):
    """Mean over unique query groups AND orderings; caller clips/steps once.

    With two orderings this is 0.5 * (mean QF loss + mean DF loss).
    Each ordering retains its own eight-candidate softmax. Teacher values are
    obtained once per group and reused for both views. Only one microbatch's
    autograd graph is retained at a time through the existing score replay.
    """
    torch = runtime.torch
    assert window and by_order
    denominator = len(window) * len(by_order)
    losses = {order: torch.zeros((), device=runtime.device) for order in by_order}
    micros = {order: 0 for order in by_order}
    for group in window:
        target = torch.tensor(target_for_group(group['id']), device=runtime.device)
        assert target.shape == (8,)
        for order, groups in by_order.items():
            rows = groups[group['id']]
            assert len(rows) == 8 and sorted(r['candidate'] for r in rows) == list(range(8))
            microplan = list(plans(rows, 4, 8192))
            values = torch.empty(8, device=runtime.device)
            with torch.no_grad():
                for micro in microplan:
                    z = runtime.logits(model, micro)
                    values[[r['candidate'] for r in micro]] = z
            loss, gradient = loss_and_score_gradient(values, target)
            losses[order] += loss / len(window)
            for micro in microplan:
                z = runtime.logits(model, micro)
                indices = [r['candidate'] for r in micro]
                assert torch.equal(z.detach(), values[indices]), 'Replay must be deterministic'
                (z * (gradient[indices] / denominator)).sum().backward()
                micros[order] += 1
    return sum(losses.values()) / len(by_order), losses, micros
