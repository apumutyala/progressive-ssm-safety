"""Causal selective SSM LM with a finite prompt memory and a learned pointer.

A prompt contributes at most 256 raw token-valued rows. A row's key depends
on that token and its six preceding tokens (left padding is zero). This window
includes the plan label at the vehicle in "Plan a route : harbor by cart".
Both arms use this same input and have 24,582 parameters at V=261, width=16.
No model
method accepts field names, source indices, reference tokens as labels, or
annotated plan assignments. A teacher sequence supplies earlier answer tokens
as ordinary causal inputs. Its public prompt boundary freezes memory writes.
Source locations can supervise the returned address logits during training.

The query uses the current input token, the first framed request, and a learned
embedding of the number of answer tokens already consumed (capped at 31). A
query scores keys by bounded dot product. The addressed arm copies a convex
combination of visible token values using softmax attention. The pooled arm
exposes the same learned addresses to supervision and has identical parameters,
but copies the uniform histogram of visible token values. Its SSM still sees
the ordered sequence. Before prompt completion, only the causal prefix is
visible. Further decoding retains exactly the prefilled keys and values.

For ordinary score vector z, let p=(1-g)*softmax(z)+g*copy, 1/256<=g<=255/256.
Convert to scores u=clip(log(p)+logsumexp(z), -16, 16), retaining the SSM's
ordinary logit scale. The final learned, signed readout uses b=u[DEFER],
u[CONFIRM] and the protected premises just as in IntervalModel. Copying either
reserved symbol enters strictly before this comparison. For missing support
the learned part contributes at most +0.5 to a prior at most -1; the gap stays
at least 0.5-2^-20 under finite, deterministic fp32 arithmetic. All intermediate
dense features/weights are bounded. softmax(z) is positive for |z|<=16 and
V<=1024, and the final log input has an explicit positive floor of 2^-80.
This decision certificate does not establish whether a copied route is right.

grow() appends the inherited zero-output SSM block. The original block and all
memory parameters/inputs retain their original shape, so the complete mixture
and signed logits are unchanged in value on identical deterministic execution.
A fresh prefill is required after growing. Study arms use one SSM block.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from interval_model import IntervalModel, IntervalState
from model import SelectiveBlock


WINDOW = 7
MAX_PROMPT = 256
STEP_BUCKETS = 32


@dataclass(frozen=True)
class BindingState:
    hidden: tuple[Tensor, ...]
    records: Tensor
    query: Tensor
    keys: Tensor           # [batch,prompt,key width]
    prompt_ids: Tensor     # raw memory values, [batch,prompt]
    address_logits: Tensor # [batch,prompt] for the last emission
    step: int              # generated tokens consumed, saturated at 31


class BindingModel(IntervalModel):
    def __init__(self, arm, vocab_size, width, confirm_id, defer_id, query_a_id, query_b_id):
        if arm not in ("addressed", "pooled"):
            raise ValueError("arm must be addressed or pooled")
        if not 4 <= vocab_size <= 1024 or not 1 <= width <= 64:
            raise ValueError("Memory study requires 4..1024 tokens and width 1..64")
        super().__init__(vocab_size, width, confirm_id, defer_id, query_a_id, query_b_id)
        self.arm = arm
        self.source_embedding = nn.Embedding(vocab_size, width)
        self.request_embedding = nn.Embedding(3, width)  # unbound, first A, first B
        self.step_embedding = nn.Embedding(STEP_BUCKETS, width)
        self.key_map = nn.Linear(WINDOW * width, width)
        self.query_map = nn.Linear(3 * width, width)
        self.copy_gate = nn.Linear(width, 1)
        with torch.no_grad():
            for table in (self.source_embedding, self.request_embedding, self.step_embedding):
                nn.init.uniform_(table.weight, -0.5, 0.5)
            self.copy_gate.bias.fill_(-1)

    def _keys(self, ids):
        batch, length = ids.shape
        embed = self.source_embedding(ids).clamp(-1, 1)
        padded = torch.cat((embed.new_zeros((batch, WINDOW - 1, self.width)), embed), dim=1)
        window = torch.cat([padded[:, i:i+length] for i in range(WINDOW)], dim=-1)
        return torch.tanh(SelectiveBlock.bounded_linear(window, self.key_map)).clamp(-1, 1)

    def _mix(self, raw, ids, query, steps, positions, source_ids, keys):
        # Source keys and read queries are always learned from causal inputs.
        token_features = self.source_embedding(ids).clamp(-1, 1)
        request_features = self.request_embedding(query + 1).clamp(-1, 1)
        step_features = self.step_embedding(steps).clamp(-1, 1).expand(ids.size(0), -1, -1)
        joint = torch.cat((token_features, request_features, step_features), dim=-1)
        address_query = torch.tanh(SelectiveBlock.bounded_linear(joint, self.query_map)).clamp(-1, 1)
        address = torch.bmm(address_query, keys.transpose(1, 2)) * (2.0 / math.sqrt(self.width))
        source_positions = torch.arange(source_ids.size(1), device=ids.device)
        visible = source_positions.view(1, 1, -1) <= positions.view(1, -1, 1)
        # Causal addressing only. Every emission has at least one visible row.
        # With |unmasked score|<=16, excluded rows carry zero fp32 softmax mass.
        address = address.masked_fill(~visible, -1024.0)
        if self.arm == "addressed":
            mass = torch.softmax(address, dim=-1)
        else:
            mass = (visible.to(torch.float32) / visible.sum(dim=-1, keepdim=True)).expand_as(address)
        values = F.one_hot(source_ids.to(torch.int64), num_classes=self.vocab_size).to(torch.float32)
        copied = torch.bmm(mass, values).clamp(0, 1)
        gate_raw = SelectiveBlock.bounded_linear(address_query, self.copy_gate)
        gate = (1 / 256 + (254 / 256) * torch.sigmoid(gate_raw)).clamp(1 / 256, 255 / 256)
        mixed = (1.0 - gate) * raw.softmax(dim=-1) + gate * copied
        scores = (mixed.clamp_min(2.0**-80).log() + raw.logsumexp(dim=-1, keepdim=True)).clamp(-16, 16)
        return scores, address

    def _emit(self, ids, events, state, source_ids, keys, positions, steps):
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError("Binding readout requires fp32 parameters")
        if len(state.hidden) != len(self.blocks):
            raise ValueError("State belongs to a different growth stage; prefill again")
        records = self._records(ids, state.records, events)
        query = self._queries(ids, state.query)
        selection = query.clamp_min(0)[:, :, None, None].expand(-1, -1, 1, 3)
        permit, lower, upper = records.gather(2, selection).squeeze(2).unbind(-1)
        active = query >= 0
        permitted = active & (permit == 1)
        sufficient = active & (lower >= 0) & (upper >= 0) & (lower >= upper)
        hidden = []
        raw = None
        for block, previous in zip(self.blocks, state.hidden):
            output, updated = block(ids, previous)
            raw = output if raw is None else (raw + output).clamp(-16, 16)
            hidden.append(updated)
        scores, address = self._mix(raw, ids, query, steps, positions, source_ids, keys)
        prior = 2.0 * (permitted.to(torch.float32) + sufficient.to(torch.float32)) - 3.0
        residual = 0.5 * scores[..., self.confirm_id].tanh().clamp(-1, 1)
        logits = scores.clone()
        logits[..., self.confirm_id] = scores[..., self.defer_id] + (prior + residual)
        final = IntervalState(tuple(hidden), records[:, -1].clone(), query[:, -1].clone())
        return logits, address, final

    def _sequence(self, ids, events, prompt_len):
        if (ids.ndim != 2 or type(prompt_len) is not int
                or not 1 <= prompt_len <= min(MAX_PROMPT, ids.size(1)) or ids.size(1) > 320):
            raise ValueError("Expected [batch,time<=320], 1<=prompt_len<=min(time,256)")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Token IDs must be int32 or int64")
        source = ids[:, :prompt_len].clone()
        positions = torch.arange(ids.size(1), device=ids.device)
        steps = (positions - (prompt_len - 1)).clamp(0, STEP_BUCKETS - 1)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            keys = self._keys(source)
            logits, address, final = self._emit(
                ids, events, self.initial_state(ids), source, keys, positions, steps
            )
        return logits, address, final, source, keys

    def teacher(self, input_ids, events, prompt_len):
        """Predict each next token; return logits [B,T,V], learned addresses [B,T,P]."""
        logits, address, _, _, _ = self._sequence(input_ids, events, prompt_len)
        return logits, address

    def forward(self, input_ids, events=None, prompt_len=None):
        """For a teacher trajectory specify the prompt boundary shared by the batch."""
        length = input_ids.size(1) if prompt_len is None else prompt_len
        return self.teacher(input_ids, events, length)[0]

    def prefill(self, input_ids, events=None):
        """Consume and freeze a fresh prompt for later incremental decoding."""
        logits, address, final, source, keys = self._sequence(input_ids, events, input_ids.size(1))
        return logits, BindingState(final.hidden, final.records, final.query, keys, source,
                                    address[:, -1].clone(), 0)

    def decode_step(self, input_ids, state, events=None):
        """Consume one raw token per batch row; prompt keys/values stay frozen."""
        if input_ids.ndim != 1:
            raise ValueError("decode_step expects input_ids [batch]")
        position = torch.tensor([state.prompt_ids.size(1)], device=input_ids.device)
        step = min(state.step + 1, STEP_BUCKETS - 1)
        steps = torch.tensor([step], device=input_ids.device)
        event_step = None if events is None else events.unsqueeze(1)
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            logits, address, final = self._emit(
                input_ids.unsqueeze(1), event_step, state, state.prompt_ids, state.keys, position, steps
            )
        return logits[:, 0], BindingState(final.hidden, final.records, final.query, state.keys,
                                         state.prompt_ids, address[:, 0], step)
