"""Autoregressive byte SSM with matched constrained and comparator families.

For a token x and each state coordinate, the certified family uses
    u = tanh(clip(drive[x], -16, 16))
    a = 1/256 + (254/256) * sigmoid(clip(decay[x], -16, 16))
    h_next = clip(a*h + (1-a)*u, -1, 1).
The convex update itself preserves [-1,1] over the reals; the final clip
guards FP32 roundoff. For an identical suffix of k tokens two real-arithmetic
initial states approach within (255/256)^k times their initial infinity-norm
distance. Clipping is nonexpansive. For the rounded multiply/multiply/add
path the same comparison permits an additive rounding error per step; the
real-valued contraction expression alone is not an executable FP32 certificate.

The comparator keeps exactly the same trainable tensors, token-dependent
transition structure, signed evidence prior, and effective dense readout.
Its drive uses the clipped raw value. Its retention adds a ramp above raw
decay 7, capped at 1/4 extra, so it can exceed one. Its hidden state has no
final clip. Dense features are clipped before multiplication in both arms.
The comparator releases the protected CONFIRM residual from 0.5*tanh(c)
to 0.5*c; with |c|<=16 either decision order is reachable. Finite comparator
state outside [-1,1] is measured; a nonfinite trajectory aborts execution.

The parameter domain admits any finite fp32 raw weights. Actual effective
embeddings/coefficients and dense weights are bounded on the executed path.
Use validate_parameters after initialization/restore and each optimizer step.
For ordinary bytes the two integer evidence records remain unknown. A framed
typed query uses the inherited first-query latch and raw integer receipts.
The architectural decision guarantee concerns reserved tokens only.
"""

from contextlib import contextmanager
import operator

import torch
from torch import nn

from interval_model import CLEAR, HOLD, IntervalModel
from model import SelectiveBlock


PAD, UNK, BOS, EOS, CONFIRM, DEFER, QUERY_A, QUERY_B = range(8)
BYTE_OFFSET = 8
VOCAB_SIZE = BYTE_OFFSET + 256
SPECIALS = ("<pad>", "<unk>", "<bos>", "<eos>", "<confirm>", "<defer>", "<query_a>", "<query_b>")


class ByteTokenizer:
    """Lossless UTF-8 byte IDs. Reserved IDs are inserted only by the caller."""

    @staticmethod
    def encode_bytes(raw):
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise TypeError("encode_bytes expects a bytes-like object")
        return [BYTE_OFFSET + b for b in bytes(raw)]

    @staticmethod
    def encode(text):
        return ByteTokenizer.encode_bytes(text.encode("utf-8", errors="strict"))

    @staticmethod
    def decode_bytes(ids):
        values = [operator.index(t) for t in ids]
        if any(not BYTE_OFFSET <= t < VOCAB_SIZE for t in values):
            raise ValueError("decode_bytes requires ordinary byte IDs only")
        return bytes(t - BYTE_OFFSET for t in values)

    @staticmethod
    def decode(ids, errors="strict"):
        return ByteTokenizer.decode_bytes(ids).decode("utf-8", errors=errors)

    @staticmethod
    def render(ids):
        """Display byte spans with reserved symbols; raw IDs disambiguate literal spellings."""
        pieces, pending = [], bytearray()
        for item in ids:
            t = operator.index(item)
            if not 0 <= t < VOCAB_SIZE:
                raise ValueError("Token ID outside the frozen vocabulary")
            if t >= BYTE_OFFSET:
                pending.append(t - BYTE_OFFSET)
            else:
                pieces.extend((pending.decode("utf-8", errors="backslashreplace"), SPECIALS[t]))
                pending.clear()
        pieces.append(pending.decode("utf-8", errors="backslashreplace"))
        return "".join(pieces)


class DynamicsAudit:
    """Detached snapshots from one or more same-shape block calls in one forward."""

    def __init__(self, model):
        self.model = model
        self.traces = []

    def record(self, hidden, proposed, retention, drive, raw_decay, raw_drive):
        self.traces.append(tuple(t.detach() for t in
                                 (hidden, proposed, retention, drive, raw_decay, raw_drive)))

    @torch.no_grad()
    def summary(self):
        if len(self.traces) != len(self.model.blocks):
            raise ValueError("Capture exactly one forward before requesting its dynamics summary")
        h, proposed, keep, drive, raw_decay, raw_drive = [
            torch.stack(items) for items in zip(*self.traces)
        ]
        params = torch.cat([p.detach().flatten() for p in self.model.parameters()])
        dense = torch.cat([p.detach().flatten() for block in self.model.blocks
                           for layer in (block.mix, block.head) for p in layer.parameters()])
        if not torch.isfinite(params).all().item() or not torch.isfinite(h).all().item():
            raise FloatingPointError("Nonfinite parameter or executed hidden trajectory")
        outside = h.abs() > 1
        time_axes = (0, 1, 3)  # block, batch, width; retain token position
        lo, hi = h.amin(dim=time_axes), h.amax(dim=time_axes)
        return dict(
            blocks=len(self.traces), parameters_finite=True, state_elements=h.numel(),
            raw_parameter_min=params.min().item(), raw_parameter_max=params.max().item(),
            raw_dense_outside_effective_box=(dense.abs() > 1).sum().item(),
            effective_dense_min=dense.clamp(-1, 1).min().item(),
            effective_dense_max=dense.clamp(-1, 1).max().item(),
            raw_decay_outside_activation_domain=(raw_decay.abs() > 16).sum().item(),
            raw_drive_outside_activation_domain=(raw_drive.abs() > 16).sum().item(),
            retention_min=keep.min().item(), retention_max=keep.max().item(),
            retention_above_one=(keep > 1).sum().item(), drive_min=drive.min().item(),
            drive_max=drive.max().item(), max_abs_drive=drive.abs().max().item(),
            hidden_min=lo.min().item(), hidden_max=hi.max().item(),
            max_abs_hidden=h.abs().max().item(), outside_box=outside.sum().item(),
            preclamp_outside_box=(proposed.abs() > 1).sum().item(),
            hidden_clamp_changes=(h != proposed).sum().item(),
            hidden_min_by_time=lo.tolist(), hidden_max_by_time=hi.tolist(),
            outside_box_by_time=outside.sum(dim=time_axes).tolist(),
        )


class CorpusBlock(SelectiveBlock):
    def forward(self, tokens, previous, *, certified, audit):
        raw_drive = self.drive(tokens)
        raw_decay = self.decay(tokens)
        drive = raw_drive.clamp(-16, 16)
        decay_arg = raw_decay.clamp(-16, 16)
        keep = (1 / 256 + (254 / 256) * decay_arg.sigmoid()).clamp(1 / 256, 255 / 256)
        if certified:
            drive = drive.tanh().clamp(-1, 1)
        else:
            keep = keep + 0.125 * (decay_arg - 7.0).clamp(0, 2)
        injection = (1.0 - keep) * drive
        trajectory, proposals = [], []
        for a, b in zip(keep.unbind(1), injection.unbind(1)):
            proposed = a * previous + b
            previous = proposed.clamp(-1, 1) if certified else proposed
            trajectory.append(previous)
            if audit is not None:
                proposals.append(proposed.detach())
        hidden = torch.stack(trajectory, dim=1)
        if not certified and not torch.isfinite(hidden).all().item():
            raise FloatingPointError("Comparator recurrent state became nonfinite")
        if audit is not None:
            audit.record(hidden, torch.stack(proposals, dim=1), keep, drive, raw_decay, raw_drive)
        features = hidden.clamp(-16, 16)
        mixed = self.bounded_linear(features, self.mix).tanh().clamp(-1, 1)
        scores = self.bounded_linear(torch.cat((features, mixed), dim=-1), self.head)
        return scores.clamp(-16, 16), previous


class CorpusModel(IntervalModel):
    def __init__(self, arm, width):
        # Construct the corpus-specific blocks directly; all protocol methods
        # and state schemas (prefill, decode, first-query latch) are inherited.
        nn.Module.__init__(self)
        if arm not in ("certified", "unconstrained") or type(width) is not int or not 1 <= width <= 8192:
            raise ValueError("Use arm certified/unconstrained and width 1..8192")
        self.arm = arm
        self.width, self.vocab_size = width, VOCAB_SIZE
        self.confirm_id, self.defer_id = CONFIRM, DEFER
        self.query_a_id, self.query_b_id = QUERY_A, QUERY_B
        self.blocks = nn.ModuleList([CorpusBlock(VOCAB_SIZE, width)])
        self._audit = None

    def grow(self, silent_head=True):
        source = self.blocks[0].drive.weight
        block = CorpusBlock(self.vocab_size, self.width, silent_head=silent_head)
        block.to(device=source.device, dtype=source.dtype)
        block.train(self.training)
        self.blocks.append(block)
        return block

    @contextmanager
    def capture_dynamics(self):
        if self._audit is not None:
            raise RuntimeError("Dynamics captures cannot be nested")
        audit = DynamicsAudit(self)
        self._audit = audit
        try:
            yield audit
        finally:
            self._audit = None

    def audited(self, input_ids, events=None):
        """One normal differentiable forward plus detached on-path dynamics metrics."""
        with self.capture_dynamics() as audit:
            logits = self(input_ids, events)
        return logits, audit.summary()

    def _score(self, tokens, hidden_state, evidence):
        if len(hidden_state) != len(self.blocks):
            raise ValueError("State belongs to a different growth stage; prefill again")
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError("Corpus recurrence requires float32 parameters")
        certified = self.arm == "certified"
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            hidden, scores = [], None
            for block, previous in zip(self.blocks, hidden_state):
                output, updated = block(tokens, previous, certified=certified, audit=self._audit)
                if scores is None:
                    scores = output
                else:
                    # Symmetric FP addition preserves an old -0 bit when a
                    # silent head is added. Both branches differentiate as x+y.
                    added = torch.where(torch.signbit(scores), -((-scores) - output), scores + output)
                    scores = added.clamp(-16, 16)
                hidden.append(updated)
            prior = 2.0 * evidence.sum(dim=-1).to(torch.float32) - 3.0
            c = scores[..., CONFIRM]
            residual = 0.5 * (c.tanh().clamp(-1, 1) if certified else c)
            logits = scores.clone()
            logits[..., CONFIRM] = scores[..., DEFER] + (prior + residual)
        return logits, tuple(hidden)


def make_model(arm, width=16):
    """Re-seed before constructing each arm to obtain bit-identical initial tensors."""
    return CorpusModel(arm, width)
