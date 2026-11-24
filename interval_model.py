"""Two-plan interval variant of the selective state-space language model.

At each input position a trusted side channel may replace or clear fields for
either plan. Receipts are signed integers [batch, time, plan, field]. Fields
are permit (0 or 1), capacity lower bound and demand upper bound (integer units
0 through MAX_UNITS). HOLD retains the previous value, CLEAR makes it unknown.
Unrecognized field values also clear that field. Omitted receipts hold all
fields. A new sequence starts with no query and six unknown fields.

The first query token in a fresh sequence binds its plan A or B. Later
query tokens cannot change that selection. No text token can update receipts.
Only a fresh episode resets the selection. For the selected plan to equal the
original request, the intended query token must occur before any other query
token; the framed input must establish it before generating a decision.
Reserved query IDs must be kept out of earlier untrusted text. Both query IDs
and their plan mapping are part of the model configuration when loading weights.

For physical semantics, a permit receipt must be valid for its plan, stored
capacity must lower-bound actual capacity, and stored demand must upper-bound
actual demand in the same units for the same plan/snapshot. The source must
deliver revocations, corrections and expiration before affected predictions.
Those truth, delivery and provenance conditions are assumptions. Given them,
permit == 1 and 0 <= demand_upper <= capacity_lower imply the selected plan is
permitted and sufficient capacity exists. Unknown fields cannot establish
support. The certified comparison concerns the two reserved raw decision IDs.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from model import Model


HOLD = -2
CLEAR = -1
MAX_UNITS = 1_000_000


@dataclass(frozen=True)
class IntervalState:
    hidden: tuple[Tensor, ...]
    records: Tensor  # [batch, two plans, three fields]
    query: Tensor  # [batch], CLEAR or 0=A or 1=B


def _latest(initial: Tensor, updates: Tensor) -> Tensor:
    """Value after the most recent replacement or clear per row and field."""
    length = updates.shape[1]
    broadcast = (1, length) + (1,) * (updates.ndim - 2)
    position = torch.arange(1, length + 1, device=updates.device).reshape(broadcast)
    last = torch.where(updates != HOLD, position, 0).cummax(dim=1).values
    return torch.cat((initial.unsqueeze(1), updates), dim=1).gather(1, last)


class IntervalModel(Model):
    def __init__(
        self, vocab_size: int, width: int, confirm_id: int, defer_id: int,
        query_a_id: int, query_b_id: int,
    ):
        super().__init__(vocab_size, width, confirm_id, defer_id)
        if not all(0 <= token < vocab_size for token in (query_a_id, query_b_id)):
            raise ValueError("query IDs must be in the vocabulary")
        if len({query_a_id, query_b_id, confirm_id, defer_id}) != 4:
            raise ValueError("query and decision tokens need four distinct IDs")
        self.query_a_id = query_a_id
        self.query_b_id = query_b_id

    def initial_state(self, input_ids: Tensor) -> IntervalState:
        batch = input_ids.shape[0]
        hidden = tuple(
            block.drive.weight.new_zeros((batch, block.width)) for block in self.blocks
        )
        records = torch.full((batch, 2, 3), CLEAR, dtype=torch.int64, device=input_ids.device)
        query = torch.full((batch,), CLEAR, dtype=torch.int64, device=input_ids.device)
        return IntervalState(hidden, records, query)

    @staticmethod
    def _records(tokens: Tensor, previous: Tensor, events: Tensor | None) -> Tensor:
        batch, length = tokens.shape
        if events is None:
            return previous.unsqueeze(1).expand(batch, length, 2, 3)
        if events.shape != (batch, length, 2, 3) or events.device != tokens.device:
            raise ValueError("receipt shape must be [batch,time,2,3] on the token device")
        if events.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("receipts must use a signed integer dtype")
        raw = events.to(torch.int64)
        clean = torch.where((raw >= HOLD) & (raw <= MAX_UNITS), raw, CLEAR)
        clean[..., 0] = torch.where(clean[..., 0] <= 1, clean[..., 0], CLEAR)
        return _latest(previous, clean)

    def _queries(self, tokens: Tensor, previous: Tensor) -> Tensor:
        """First assignment persists, including across prefill/decode calls."""
        length = tokens.size(1)
        position = torch.arange(length, device=tokens.device).unsqueeze(0)
        first_a = torch.where(tokens == self.query_a_id, position, length).cummin(1).values
        first_b = torch.where(tokens == self.query_b_id, position, length).cummin(1).values
        first = torch.where(first_a < first_b, 0, torch.where(first_b < first_a, 1, CLEAR))
        return torch.where(previous[:, None] == CLEAR, first, previous[:, None])

    def _run(
        self, tokens: Tensor, state: IntervalState, events: Tensor | None,
    ) -> tuple[Tensor, IntervalState]:
        if tokens.ndim != 2 or tokens.size(1) == 0:
            raise ValueError("input_ids must have shape [batch, nonempty time]")
        records = self._records(tokens, state.records, events)
        query = self._queries(tokens, state.query)
        # A gather needs a valid index even when there is no active query.
        # Both premises include the active-query bit explicitly.
        index = query.clamp_min(0)[:, :, None, None].expand(-1, -1, 1, 3)
        permit, lower, upper = records.gather(2, index).squeeze(2).unbind(-1)
        active = query != CLEAR
        permitted = active & (permit == 1)
        sufficient = active & (lower >= 0) & (upper >= 0) & (lower >= upper)
        premises = torch.stack((permitted, sufficient), dim=-1)
        logits, hidden = self._score(tokens, state.hidden, premises)
        return logits, IntervalState(hidden, records[:, -1].clone(), query[:, -1].clone())

    def prefill(
        self, input_ids: Tensor, events: Tensor | None = None,
    ) -> tuple[Tensor, IntervalState]:
        """Consume a fresh prefix and optional raw receipts; return every logit vector."""
        return self._run(input_ids, self.initial_state(input_ids), events)

    def decode_step(
        self, input_ids: Tensor, state: IntervalState, events: Tensor | None = None,
    ) -> tuple[Tensor, IntervalState]:
        """Consume a token per row and optional [batch, two plans, three fields] receipts."""
        if input_ids.ndim != 1:
            raise ValueError("decode_step expects input_ids of shape [batch]")
        step = None if events is None else events.unsqueeze(1)
        logits, updated = self._run(input_ids.unsqueeze(1), state, step)
        return logits[:, 0], updated
