"""Autoregressive selective SSM with two protected evidence coordinates.

Events apply before predicting the following token. Only the caller can supply
events. A generated token supplied to decode_step has no event by default.
The runner admits finite fp32 parameters, verifies logits, and sets the numeric
backend. A State must originate from this model at its current growth stage.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class State:
    hidden: tuple[Tensor, ...]  # one [batch, width] tensor per block
    evidence: Tensor  # [batch, 2] booleans, in the trusted runtime


class SelectiveBlock(nn.Module):
    """Diagonal token-conditioned transition, nonlinear feature map, linear head."""

    def __init__(self, vocab_size: int, width: int, silent_head: bool = False):
        super().__init__()
        self.width = width
        self.drive = nn.Embedding(vocab_size, width)
        self.decay = nn.Embedding(vocab_size, width)
        self.mix = nn.Linear(width, width)
        self.head = nn.Linear(2 * width, vocab_size)
        with torch.no_grad():
            nn.init.normal_(self.drive.weight, std=0.35)
            nn.init.normal_(self.decay.weight, std=0.25)
            self.decay.weight.add_(torch.linspace(-1.5, 5.5, width))
            if silent_head:
                self.head.weight.zero_()
                self.head.bias.zero_()

    @staticmethod
    def bounded_linear(x: Tensor, layer: nn.Linear) -> Tensor:
        return F.linear(x, layer.weight.clamp(-1, 1), layer.bias.clamp(-1, 1))

    def forward(self, tokens: Tensor, previous: Tensor) -> tuple[Tensor, Tensor]:
        drive = torch.tanh(self.drive(tokens).clamp(-16, 16)).clamp(-1, 1)
        decay = (1 / 256 + (254 / 256) * torch.sigmoid(self.decay(tokens).clamp(-16, 16))).clamp(
            1 / 256, 255 / 256
        )
        injection = (1.0 - decay) * drive
        trajectory = []
        for keep, add in zip(decay.unbind(1), injection.unbind(1)):
            previous = (keep * previous + add).clamp(-1, 1)
            trajectory.append(previous)
        hidden = torch.stack(trajectory, dim=1)
        mixed = torch.tanh(self.bounded_linear(hidden, self.mix)).clamp(-1, 1)
        features = torch.cat((hidden, mixed), dim=-1)
        return self.bounded_linear(features, self.head).clamp(-16, 16), previous


class Model(nn.Module):
    def __init__(self, vocab_size: int, width: int, confirm_id: int, defer_id: int):
        super().__init__()
        if not 0 <= confirm_id < vocab_size or not 0 <= defer_id < vocab_size:
            raise ValueError("decision IDs must be in the vocabulary")
        if confirm_id == defer_id:
            raise ValueError("decision IDs must be distinct")
        if not 1 <= width <= 8192:
            raise ValueError("supported block width is 1 through 8192")
        self.vocab_size = vocab_size
        self.width = width
        self.confirm_id = confirm_id
        self.defer_id = defer_id
        self.blocks = nn.ModuleList([SelectiveBlock(vocab_size, width)])

    def grow(self, silent_head: bool = True) -> SelectiveBlock:
        """Append a block. The default preserves logits; false initializes a scratch baseline."""
        source = self.blocks[0].drive.weight
        new = SelectiveBlock(self.vocab_size, self.width, silent_head=silent_head)
        new.to(device=source.device, dtype=source.dtype)
        new.train(self.training)
        self.blocks.append(new)
        return new

    @torch.no_grad()
    def validate_parameters(self) -> None:
        """Admission check after initialization, updates, or checkpoint restore."""
        finite = []
        for name, parameter in self.named_parameters():
            if parameter.dtype != torch.float32:
                raise ValueError(f"{name} has unsupported dtype {parameter.dtype}")
            finite.append(torch.isfinite(parameter).all())
        if not torch.stack(finite).all().item():
            raise FloatingPointError("model has nonfinite parameters")

    def initial_state(self, input_ids: Tensor) -> State:
        batch = input_ids.shape[0]
        hidden = tuple(
            block.drive.weight.new_zeros((batch, block.width)) for block in self.blocks
        )
        evidence = torch.zeros((batch, 2), device=input_ids.device, dtype=torch.bool)
        return State(hidden, evidence)

    @staticmethod
    def _evidence_trajectory(tokens: Tensor, previous: Tensor, events: Tensor | None) -> Tensor:
        batch, length = tokens.shape
        if events is None:
            return previous.unsqueeze(1).expand(batch, length, 2)
        if events.shape != (batch, length, 2) or events.device != tokens.device:
            raise ValueError("events must have shape [batch, time, 2] on the input device")
        if events.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("events must have signed integer dtype")
        # Gather the value at each coordinate's most recent nonzero event.
        # Index zero is the incoming state. Unknown nonzero codes clear.
        positions = torch.arange(1, length + 1, device=tokens.device).view(1, length, 1)
        last = torch.where(events != 0, positions, 0).cummax(dim=1).values
        values = torch.cat((previous.unsqueeze(1), events == 1), dim=1)
        return values.gather(1, last)

    def _score(
        self, tokens: Tensor, hidden_state: tuple[Tensor, ...], evidence: Tensor
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Shared neural recurrence and signed readout; evidence is internal [B,T,2]."""
        if any(parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise ValueError("the certified execution path requires float32 parameters")
        if len(hidden_state) != len(self.blocks):
            raise ValueError("state belongs to a different growth stage; prefill again")
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            hidden = []
            scores = None
            for block, previous in zip(self.blocks, hidden_state):
                output, updated = block(tokens, previous)
                scores = output if scores is None else (scores + output).clamp(-16, 16)
                hidden.append(updated)
            # Decision tokens share an ordinary learned base score. With fewer
            # than two supported premises the signed term is at most -1. The
            # bounded learned difference can offset this term by at most 1/2.
            count = evidence.sum(dim=-1).to(torch.float32)
            signed = 2.0 * count - 3.0
            residual = 0.5 * torch.tanh(scores[..., self.confirm_id]).clamp(-1, 1)
            confirm = scores[..., self.defer_id] + (signed + residual)
            logits = scores.clone()
            logits[..., self.confirm_id] = confirm
        return logits, tuple(hidden)

    def _run(self, tokens: Tensor, state: State, events: Tensor | None) -> tuple[Tensor, State]:
        if tokens.ndim != 2 or tokens.size(1) == 0:
            raise ValueError("input_ids must have shape [batch, nonempty time]")
        evidence = self._evidence_trajectory(tokens, state.evidence, events)
        logits, hidden = self._score(tokens, state.hidden, evidence)
        return logits, State(hidden, evidence[:, -1])

    def forward(self, input_ids: Tensor, events: Tensor | None = None) -> Tensor:
        """Return one next-token logit vector per input position."""
        return self.prefill(input_ids, events)[0]

    def prefill(self, input_ids: Tensor, events: Tensor | None = None) -> tuple[Tensor, State]:
        """Consume a fresh prefix [batch, time] and return all logits and final state."""
        return self._run(input_ids, self.initial_state(input_ids), events)

    def decode_step(
        self, input_ids: Tensor, state: State, events: Tensor | None = None
    ) -> tuple[Tensor, State]:
        """Consume one token per row [batch] and optional events [batch, 2]."""
        if input_ids.ndim != 1:
            raise ValueError("decode_step expects input_ids of shape [batch]")
        event_step = None if events is None else events.unsqueeze(1)
        logits, updated = self._run(input_ids.unsqueeze(1), state, event_step)
        return logits[:, 0], updated
