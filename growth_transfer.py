"""Silent parallel-width insertion and its original Adam and logit checks."""
import copy
import torch
import pretrain
from corpus_model import BOS, CONFIRM, DEFER, HOLD, QUERY_A, ByteTokenizer

def bits_equal(left, right):
    return torch.equal(left.contiguous().view(torch.int32), right.contiguous().view(torch.int32))


def tensor_changes(before, model):
    result = {}
    for name, value in model.state_dict().items():
        if name in before:
            delta = value.detach().cpu().double()-before[name].double()
            result[name] = {'l2': float(torch.linalg.vector_norm(delta)), 'max_abs': float(delta.abs().max())}
    return result


def decision_probe(model, device):
    ids = [BOS]+ByteTokenizer.encode('A small parcel rests on the table. ')+[QUERY_A]
    tokens = torch.tensor([ids, ids], dtype=torch.long, device=device)
    events = torch.full((2, len(ids), 2, 3), HOLD, dtype=torch.long, device=device)
    events[:, -1, 0] = torch.tensor([[1, 1, 2], [1, 2, 2]], dtype=torch.long, device=device)
    logits, _ = model.prefill(tokens, events=events)
    pretrain.finite(logits, 'insertion decision-probe logits')
    margins = logits[:, -1, CONFIRM]-logits[:, -1, DEFER]
    if not (float(margins[0]) < 0 < float(margins[1])):
        raise RuntimeError('Intrinsic positive/negative typed decision-probe margin failed.')
    return logits, margins.tolist()


@torch.no_grad()
def insert(base, ordinary_rows):
    x, y = pretrain.pack(ordinary_rows, base.device)
    base.model.eval()
    old, old_stats = base.model.audited(x)
    pretrain.audit_ordinary_output(old, y, old_stats, True)
    old_prefill = base.model.prefill(x)[0]
    old_probe, old_margins = decision_probe(base.model, base.device)
    previous = copy.deepcopy(base.opt.state_dict())
    old_parameters = {id(p) for p in base.model.parameters()}
    base.model.grow(silent_head=True)
    new_parameters = [p for p in base.model.parameters() if id(p) not in old_parameters]
    base.opt.add_param_group({'params': new_parameters})
    current = base.opt.state_dict()
    moments_equal = (pretrain.exact_equal(previous['state'], current['state'])
        and pretrain.exact_equal(previous['param_groups'], current['param_groups'][:len(previous['param_groups'])]))
    if not moments_equal: raise RuntimeError('Insertion altered old Adam state/groups.')
    base.model.validate_parameters()
    new, new_stats = base.model.audited(x)
    pretrain.audit_ordinary_output(new, y, new_stats, True)
    new_prefill = base.model.prefill(x)[0]
    new_probe, new_margins = decision_probe(base.model, base.device)
    checks = {'forward': bits_equal(old, new), 'prefill': bits_equal(old_prefill, new_prefill),
              'typed_decision_probe': bits_equal(old_probe, new_probe)}
    if not all(checks.values()): raise RuntimeError('Zero-connected insertion changed full unchanged-path logits.')
    return {'exact_equal': True, 'components': checks, 'old_adam_state_unchanged': moments_equal,
        'forward_coordinates': old.numel(), 'prefill_coordinates': old_prefill.numel(),
        'typed_probe_coordinates': old_probe.numel(), 'max_abs_forward_delta': float((old-new).abs().max()),
        'old_typed_margins': old_margins, 'new_typed_margins': new_margins,
        'before_dynamics': {k: v for k, v in old_stats.items() if k not in
                            ('hidden_min_by_time', 'hidden_max_by_time', 'outside_box_by_time')},
        'after_dynamics': {k: v for k, v in new_stats.items() if k not in
                           ('hidden_min_by_time', 'hidden_max_by_time', 'outside_box_by_time')}}
