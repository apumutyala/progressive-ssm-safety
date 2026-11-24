"""Tensor checks and ordinary-byte packing used by growth_transfer."""
import torch
from corpus_model import BOS, CONFIRM, DEFER, EOS, VOCAB_SIZE

def exact_equal(left, right):
    """Independent structural equality for actual checkpoint tensors, not file digests."""
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.cpu(), right.cpu()))
    if type(left) is not type(right): return False
    if isinstance(left, dict): return left.keys() == right.keys() and all(exact_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)): return len(left) == len(right) and all(exact_equal(a, b) for a, b in zip(left, right))
    return left == right


def finite(x, label):
    if not torch.isfinite(x).all().item(): raise RuntimeError(f'Nonfinite {label}.')


def audit_ordinary_output(logits, targets, stats, certified):
    """Measure the unsupported reserved-decision invariant on actual corpus logits."""
    values = logits.detach()
    mask = targets != -100
    margins = values[..., DEFER]-values[..., CONFIRM]
    finite(margins, 'ordinary unsupported decision margins')
    minimum = float(margins[mask].min())
    unsafe = int(((values.argmax(-1) == CONFIRM) & mask).sum())
    stats.update(ordinary_defer_minus_confirm_min=minimum,
                 ordinary_unsupported_confirm_argmax=unsafe,
                 ordinary_output_vectors=int(mask.sum()))
    if certified and (minimum <= 0 or unsafe):
        raise RuntimeError('Certified ordinary-text unsupported output invariant failed.')
    return stats


def pack(rows, device):
    length = max(len(r['ids']) for r in rows)-1
    ids = torch.full((len(rows), length), EOS, dtype=torch.long, device=device)
    targets = torch.full_like(ids, -100)
    for i, row in enumerate(rows):
        values = row['ids']
        if values[0] != BOS or any(not 0 <= token < VOCAB_SIZE for token in values):
            raise RuntimeError('Invalid byte corpus framing/token domain.')
        if any(token < 8 and token != EOS for token in values[1:]):
            raise RuntimeError('Protocol or synthetic content token in ordinary corpus.')
        n = len(values)-1
        ids[i, :n] = torch.tensor(values[:-1], dtype=torch.long, device=device)
        targets[i, :n] = torch.tensor(values[1:], dtype=torch.long, device=device)
    return ids, targets
