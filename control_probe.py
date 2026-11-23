"""Bounded finite-horizon control diagnostics for a fitted byte SSM.

The measured output is the ordinary byte contrast logit('R')-logit('B').
The real-valued local surrogate is evaluated at the actual fp32 intermediates:
A_t = diag(c_t * a_t), where a is the executed retention and c is the
state-clamp derivative (one on the closed box, zero outside; at the boundary
we explicitly use PyTorch's inclusive clamp convention). A comparator has
no state clamp. C_t is obtained by autograd through the actual readout.
These are local derivatives of a smooth/piecewise-smooth interpretation,
not derivatives of the machine's rounding map.

Inputs are discrete bytes. The controllability calculation instead defines
one hypothetical scalar intervention: at time t perturb the *effective
drive vector* by epsilon times the unit direction of the fitted drive(R)
minus drive(B). Its input column is B_t = c_t * (1-a_t) * direction.
It is not a free actuator on each coordinate or an instruction the caller
can submit to the LM. A separate actual token substitution measures what
ordinary text can do, including its effect on retention.

Over t=1..H the observability rows are C_t A_t ... A_1. Controllability
columns at the terminal state are A_H ... A_(t+1) B_t. We report spectra of
the two corresponding Gramians for one specified trajectory, this input,
and this scalar output. A small numerical rank is not a universal property
of all prompts, outputs, or distinct state variables.
"""
from dataclasses import replace

import torch

from corpus_model import BYTE_OFFSET, CONFIRM, DEFER, HOLD, QUERY_A, VOCAB_SIZE, ByteTokenizer, CorpusModel


_R, _B = ByteTokenizer.encode("RB")
_CARD_HEADER = "The only mark printed on the card was "
_CARD_SUFFIX = ". After waiting, the reader was asked to copy the card marking: "
_LAGS = (0, 1, 16, 64)


def _cpu64(x):
    return x.detach().to(device="cpu", dtype=torch.float64)


def _spectrum(transfer):
    # Rows are responses to each pulse (or observation). Work off the model
    # device on <=64x64 double-precision matrices; no full-vocabulary Jacobian.
    gram = transfer.T @ transfer
    eigen = torch.linalg.eigvalsh(gram)
    if not torch.isfinite(eigen).all().item():
        raise FloatingPointError("Nonfinite Gramian spectrum")
    largest = max(0.0, eigen[-1].item())
    threshold = max(1e-12, 1e-8 * largest)
    return dict(trace=gram.trace().item(), numerical_rank=(eigen > threshold).sum().item(),
                rank_threshold=threshold, max_eigenvalue=largest,
                top_four_eigenvalues=eigen.flip(0)[:4].clamp_min(0).tolist())


def _linear_factors(proposal, retention, direction, certified):
    active = ((proposal >= -1) & (proposal <= 1)) if certified else torch.ones_like(proposal, dtype=torch.bool)
    a = active * retention
    b = active * (1.0 - retention) * direction.unsqueeze(0)
    return a, b


@torch.no_grad()
def _ordinary_substitution(model, device):
    header = ByteTokenizer.encode(_CARD_HEADER)
    suffix = ByteTokenizer.encode(_CARD_SUFFIX)
    if len(suffix) != max(_LAGS):
        raise RuntimeError("Unexpected public fixed-suffix length")
    ids = torch.tensor([header + [mark] + suffix for mark in (_R, _B)], dtype=torch.int64, device=device)
    with model.capture_dynamics() as tap:
        logits, _ = model.prefill(ids)
    h = torch.cat([record[0] for record in tap.traces], dim=-1)
    drive = torch.cat([record[3] for record in tap.traces], dim=-1)
    marker = len(header)
    difference = _cpu64(drive[0, marker] - drive[1, marker])
    norm = torch.linalg.vector_norm(difference).item()
    direction = difference / norm if norm else torch.zeros_like(difference)
    pairs = []
    for lag in _LAGS:
        i = marker + lag
        margin = _cpu64(logits[:, i, _R]) - _cpu64(logits[:, i, _B])
        pairs.append(dict(
            identical_suffix_bytes=lag,
            hidden_linf_difference=(_cpu64(h[0, i]) - _cpu64(h[1, i])).abs().max().item(),
            output_contrast_change=(margin[0] - margin[1]).item(),
            contrast_after_R=margin[0].item(), contrast_after_B=margin[1].item(),
            greedy_tokens_differ=bool(logits[0, i].argmax() != logits[1, i].argmax()),
        ))
    return dict(effective_drive_difference_l2=norm, by_lag=pairs), direction


def _trajectory_linearization(model, tokens, direction):
    # The normal capture hook owns installation/restoration. Its per-call
    # observer retains the actual stacked hidden tensors instead of detached
    # copies, allowing a single VJP through the *executed* time-local heads.
    # Each h[t] here feeds only its own readout: recurrence used the earlier
    # per-step tensors before stacking.
    observed = []

    def capture(hidden, proposed, retention, drive, raw_decay, raw_drive):
        observed.append((hidden, proposed.detach(), retention.detach()))

    with torch.inference_mode(False), torch.enable_grad():
        tokens = tokens.detach().clone()
        incoming = model.initial_state(tokens)
        incoming = replace(incoming, hidden=tuple(h.requires_grad_(True) for h in incoming.hidden))
        with model.capture_dynamics() as tap:
            tap.record = capture
            logits, _ = model._run(tokens, incoming, events=None)
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("Nonfinite readout in trajectory linearization")
        contrast = logits[..., _R] - logits[..., _B]
        gradients = torch.autograd.grad(contrast.sum(), [sample[0] for sample in observed])
        c = _cpu64(torch.cat(gradients, dim=-1)[0])
        proposal = _cpu64(torch.cat([record[1] for record in observed], dim=-1)[0])
        retention = _cpu64(torch.cat([record[2] for record in observed], dim=-1)[0])
    certified = model.arm == "certified"
    a, b = _linear_factors(proposal, retention, direction, certified)
    from_start = torch.cumprod(a, dim=0)
    through_end = torch.cumprod(a.flip(0), dim=0).flip(0)
    after = torch.cat((through_end[1:], torch.ones_like(a[:1])), dim=0)
    obs = c * from_start
    ctrl = after * b
    return dict(
        horizon_bytes=a.size(0), state_dimension=a.size(1),
        state_jacobian_diagonal_min=a.min().item(), state_jacobian_diagonal_max=a.max().item(),
        final_local_state_transition_induced_linf=from_start[-1].abs().max().item(),
        max_drive_column_l2=torch.linalg.vector_norm(b, dim=1).max().item(),
        initial_pulse_to_terminal_contrast=torch.dot(c[-1], ctrl[0]).item(),
        terminal_output_state_gradient_l2=torch.linalg.vector_norm(c[-1]).item(),
        max_output_state_gradient_l2=torch.linalg.vector_norm(c, dim=1).max().item(),
        boundary_coordinates=int((proposal.abs() == 1).sum()) if certified else 0,
        clipped_coordinates=int((proposal.abs() > 1).sum()) if certified else 0,
        observability=_spectrum(obs), controllability=_spectrum(ctrl),
    )


@torch.no_grad()
def _receipt_intervention(model, history):
    # Only one integer field changes. Identical ordinary bytes precede the
    # common final query. No prompt byte, protected record or raw ID is emitted.
    ids = torch.cat((history, history.new_full((1, 1), QUERY_A)), dim=1).expand(2, -1)
    events = ids.new_full((2, ids.size(1), 2, 3), HOLD)
    events[:, -1, 0] = ids.new_tensor([[1, 1, 2], [1, 2, 2]])
    logits, state = model.prefill(ids, events)
    final = logits[:, -1]
    hidden_equal = all(torch.equal(h[0].contiguous().view(torch.int32), h[1].contiguous().view(torch.int32))
                       for h in state.hidden)
    ordinary_equal = torch.equal(final[0, BYTE_OFFSET:].contiguous().view(torch.int32),
                                 final[1, BYTE_OFFSET:].contiguous().view(torch.int32))
    if not hidden_equal or not ordinary_equal:
        raise RuntimeError("Paired identical learned inputs produced different ordinary trajectory outputs")
    margins = _cpu64(final[:, CONFIRM]) - _cpu64(final[:, DEFER])
    return dict(learned_final_state_bits_equal=hidden_equal, ordinary_final_logits_bits_equal=ordinary_equal,
                unsupported_confirm_minus_defer=margins[0].item(), supported_confirm_minus_defer=margins[1].item(),
                discrete_margin_change=(margins[1] - margins[0]).item(),
                decision_pair_order_flipped=bool(margins[0] < 0 < margins[1]),
                unsupported_raw_greedy_is_confirm=bool(final[0].argmax() == CONFIRM),
                supported_raw_greedy_is_confirm=bool(final[1].argmax() == CONFIRM))


def analyze(model, dev_tensor):
    """Summarize first <=64 bytes of first approved dev row, plus public probes.

    No input text, byte IDs, per-document output text, personal identifiers,
    hidden vectors, singular vectors or token-level private traces are returned.
    No weights, RNG state, optimizer state, or parameter gradients are changed.
    """
    if type(model) is not CorpusModel or len(model.blocks) > 2 or model.width > 32:
        raise ValueError("Expected the bounded fitted corpus model (<=2 blocks, width<=32)")
    if dev_tensor.ndim != 2 or dev_tensor.dtype != torch.int64 or min(dev_tensor.shape) < 1:
        raise ValueError("Expected nonempty int64 [batch,time] of ordinary dev bytes")
    tokens = dev_tensor[:1, :64]
    if tokens.device != next(model.parameters()).device or ((tokens < BYTE_OFFSET) | (tokens >= VOCAB_SIZE)).any():
        raise ValueError("Provide ordinary byte IDs on the fitted model's device")
    model.validate_parameters()
    substitution, direction = _ordinary_substitution(model, tokens.device)
    return dict(
        schema=1, arm=model.arm, development_window=_trajectory_linearization(model, tokens, direction),
        public_card_substitution=substitution, receipt_intervention=_receipt_intervention(model, tokens),
        contrast="ordinary R-minus-B logits", input="unit scalar along the fitted effective R-minus-B drive",
        derivative="local real-arithmetic surrogate at executed FP32 values; inclusive state-clamp subgradient",
    )
