"""Byte next-token LMs with a learned address over previous token positions.

Each SSM layer writes a contextual key and value after integrating the current
byte into a selective recurrent state. Its query depends on that state and the
current representation. A later question can therefore change the address.
Entity, report source and epoch are ordinary byte spans: there are no supplied
field boundaries, truth labels, copy targets, latest-write flags or answer
slots. The model must learn those distinctions. forward_without_recurrence
removes the recurrent contribution to keys, queries, values AND residual
features with fitted weights fixed. Compare raw decisions/likelihood on paired
query and evidence edits. This intervention measures reliance on the state;
it does not establish that individual coordinates encode an entity or epoch.

Typed events use a separate [B,T,2,5] integer channel (source, epoch, permit,
lower capacity, upper demand). Each plan admits its configured source and a
strictly newer epoch. Only complete snapshots or explicit revocation are
applied. Source IDs are authenticated by the caller; these integers alone
are not cryptographic credentials. Truth, timely delivery, expiry handling
and nonreuse of the epoch space within an episode are external premises.
Query A/B selects the first plan, including under
cached decoding. Normal UTF-8 bytes use IDs 8..263 and cannot write receipts.
The receipt changes only the reserved CONFIRM logit. It cannot certify an
ordinary affirmative sentence. The protected gap holds for the certified SSM
from initialization at every finite FP32 update (FP32 eager, autocast off;
validate_parameters before admitting updates). The sham decision head keeps
the same signed prior while allowing learned offsets to reverse it. Both
comparator architectures retain finite effective ranges for numerical work.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from interval_model import IntervalModel, CLEAR
from corpus_model import VOCAB_SIZE, QUERY_A, QUERY_B, CONFIRM, DEFER

ARMS = ("certified_ssm", "unconstrained_ssm", "transformer")
AUTHORITY_SOURCES = (17, 29)
MAX_EPOCH = 2**31-1


def _receipts(ids, state, events):
    """Atomic updates. Invalid or stale events hold prior state including its epoch."""
    b, t = ids.shape
    record = ids.new_full((b, 2, 3), CLEAR) if state is None else state.records
    source = ids.new_full((b, 2), CLEAR) if state is None else state.sources
    epoch = ids.new_full((b, 2), CLEAR) if state is None else state.epochs
    if events is None:
        return record[:, None].expand(-1, t, -1, -1), source, epoch
    if (events.shape != (b, t, 2, 5) or events.device != ids.device or
            events.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)):
        raise ValueError("Expected signed integer [batch,time,2,5] source/epoch/permit/lower/upper receipts")
    e = events.to(torch.int64)
    issuer, stamp, permit, lower, upper = e.unbind(-1)
    bounds = (lower >= 0) & (lower <= 1_000_000) & (upper >= 0) & (upper <= 1_000_000)
    revoke = (permit == 0) & (lower == CLEAR) & (upper == CLEAR)
    valid = ((issuer == ids.new_tensor(AUTHORITY_SOURCES)) & (stamp >= 0) & (stamp <= MAX_EPOCH)
             & ((permit == 0) | (permit == 1)) & (bounds | revoke))
    stamps = torch.where(valid, stamp, CLEAR)
    maximum = torch.cat((epoch[:, None], stamps), 1).cummax(1).values
    accept = valid & (stamp > maximum[:, :-1])
    position = torch.arange(1, t+1, device=ids.device).view(1, t, 1)
    last = torch.where(accept, position, 0).cummax(1).values
    trajectory = torch.cat((record[:, None], e[..., 2:]), 1).gather(1, last[..., None].expand(-1, -1, -1, 3))
    next_source = torch.cat((source[:, None], issuer), 1).gather(1, last[:, -1:])
    return trajectory, next_source[:, 0].clone(), maximum[:, -1].clone()


def _dimensions(arm, width, depth):
    if arm not in ARMS or type(width) is not int or not 4 <= width <= 2048 or width % 4:
        raise ValueError("Expected a registered arm and width divisible by four in 4..2048")
    if type(depth) is not int or not 1 <= depth <= 16:
        raise ValueError("Depth outside 1..16")


def parameter_count(arm, width=32, depth=2):
    _dimensions(arm, width, depth)
    layer = 12*width**2 + 9*width
    if arm != "transformer":
        layer += 2*width**2 + 2*width
    return 528*width+264+depth*layer


def matched_width(arm, target_parameters=20_000_000, depth=4):
    """Nearest permitted integer width, smaller width on ties; allocation-free."""
    if type(target_parameters) is not int or target_parameters < 1:
        raise ValueError("Expected positive target size")
    return min(range(4, 2049, 4), key=lambda w: abs(parameter_count(arm, w, depth)-target_parameters))


def resource_estimate(arm, width, depth, batch, sequence_length):
    """Logical sizes for one ungrown branch. Excludes activations, allocator/workspaces, optimizer scalars and receipts.

    Attention term counts unique causal query/key pairs; actual kernels may
    perform additional work. SDPA chooses its own kernels. A math fallback
    can materialize attention scores of size B*heads*T*T. Cached K/V views
    may retain their larger packed QKV storage, particularly after prefill.
    """
    n = parameter_count(arm, width, depth)
    if any(type(x) is not int or x < 1 for x in (batch, sequence_length)):
        raise ValueError("Positive batch and length required")
    b, t, w, d = batch, sequence_length, width, depth
    recurrent = arm != "transformer"
    return dict(parameters=n, fp32_weight_bytes=4*n, fp32_weights_grads_two_adam_moments_bytes=16*n,
                fp32_kv_cache_bytes=8*b*t*w*d, fp32_recurrent_cache_bytes=4*b*w*d*recurrent,
                dense_forward_macs=b*t*(264*w+d*(12+2*int(recurrent))*w*w),
                causal_attention_macs=b*d*w*t*(t+1), heads=_heads(w))


def _heads(width):
    # At least four independent learned addresses; large models use head dim <=64.
    return next(h for h in range(4, width+1) if width % h == 0 and width//h <= 64)


def _linear(x, layer):
    return F.linear(x, layer.weight.clamp(-1, 1), layer.bias.clamp(-1, 1)).clamp(-16, 16)


class Layer(nn.Module):
    def __init__(self, arm, width):
        super().__init__()
        self.arm, self.width, self.heads = arm, width, _heads(width)
        self.transition = None if arm == "transformer" else nn.Linear(width, 2*width)
        if self.transition is not None:
            self.register_buffer("timescale", torch.linspace(-1, 6, width), persistent=False)
        self.qkv = nn.Linear(width, 3*width)
        self.project = nn.Linear(width, width)
        self.expand = nn.Linear(width, 4*width)
        self.contract = nn.Linear(4*width, width)

    def forward(self, x, cache, ablation):
        hidden, old_k, old_v = (None, None, None) if cache is None else cache
        if self.transition is not None and ablation != "recurrence":
            # A fixed bias spread gives the learned keys multiple memory time scales.
            decay, raw = _linear(x, self.transition).chunk(2, dim=-1)
            keep = (1/256+(254/256)*(decay+self.timescale).sigmoid()).clamp(1/256, 255/256)
            if self.arm == "certified_ssm":
                drive, limit = raw.tanh(), 1.
            else:
                keep = keep + .125*(decay-7).clamp(0, 2)
                drive, limit = raw, 16.
            injection = (1-keep)*drive
            h = x.new_zeros((x.size(0), self.width)) if hidden is None else hidden
            trajectory = []
            for a, v in zip(keep.unbind(1), injection.unbind(1)):
                h = (a*h+v).clamp(-limit, limit)
                trajectory.append(h)
            hidden = h
            x = (x+torch.stack(trajectory, dim=1)).clamp(-16, 16)
        normalized = F.layer_norm(x, (self.width,))
        qkv = _linear(normalized, self.qkv).view(x.size(0), x.size(1), 3, self.heads, -1)
        q, k, v = (part.transpose(1, 2) for part in qkv.unbind(2))
        if old_k is not None:
            if x.size(1) != 1:
                raise ValueError("Cached path takes exactly one new token")
            k, v = torch.cat((old_k, k), dim=2), torch.cat((old_v, v), dim=2)
        if ablation != "retrieval":
            read = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=old_k is None)
            x = (x+_linear(read.transpose(1, 2).reshape_as(x), self.project)).clamp(-16, 16)
        feed = _linear(F.gelu(_linear(F.layer_norm(x, (self.width,)), self.expand)), self.contract)
        return (x+feed).clamp(-16, 16), (hidden, k, v)


class Branch(nn.Module):
    def __init__(self, arm, width, depth, context, silent):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, width)
        self.layers = nn.ModuleList([Layer(arm, width) for _ in range(depth)])
        angles = torch.arange(context)[:, None]*torch.exp(torch.arange(0, width, 2)*(-math.log(10_000.)/width))
        self.register_buffer("position", torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1), persistent=False)
        self.head = nn.Linear(width, VOCAB_SIZE)
        with torch.no_grad():
            nn.init.normal_(self.embedding.weight, std=.25)
            if silent:
                self.head.weight.zero_()
                self.head.bias.zero_()

    def forward(self, ids, offset, caches, ablation):
        x = self.embedding(ids).clamp(-4, 4)+.25*self.position[offset:offset+ids.size(1)]
        updated = []
        for i, layer in enumerate(self.layers):
            x, cache = layer(x, None if caches is None else caches[i], ablation)
            updated.append(cache)
        return _linear(F.layer_norm(x, (self.head.in_features,)), self.head), tuple(updated)


@dataclass(frozen=True)
class State:
    caches: tuple
    query: torch.Tensor
    records: torch.Tensor
    sources: torch.Tensor
    epochs: torch.Tensor
    length: int

    @property
    def hidden(self):
        return tuple(h for branch in self.caches for h, _, _ in branch if h is not None)


class QualificationModel(nn.Module):
    query_a_id, query_b_id = QUERY_A, QUERY_B
    confirm_id, defer_id, vocab_size = CONFIRM, DEFER, VOCAB_SIZE

    def __init__(self, arm, width, depth, max_context):
        super().__init__()
        _dimensions(arm, width, depth)
        if type(max_context) is not int or not 2 <= max_context <= 8192:
            raise ValueError("Expected context in 2..8192")
        self.arm, self.width, self.depth, self.max_context = arm, width, depth, max_context
        self.branches = nn.ModuleList([Branch(arm, width, depth, max_context, False)])

    def validate_parameters(self):
        for p in self.parameters():
            if p.dtype != torch.float32 or not torch.isfinite(p).all().item():
                raise ValueError("Finite FP32 parameters required")

    def grow(self, silent_head=True):
        """Append an independent branch; caller registers new Adam parameters and re-prefills caches."""
        if silent_head is not True:
            raise ValueError("Only certificate-preserving zero-head insertion is available")
        b = Branch(self.arm, self.width, self.depth, self.max_context, True).to(self.branches[0].head.weight)
        b.train(self.training)
        self.branches.append(b)
        return b

    def grow_width_with_adam(self, optimizer):
        """Add a silent branch to one live Adam group. Existing tensors/moments stay in place."""
        old = list(self.parameters())
        if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups) != 1:
            raise ValueError("Width qualification requires one Adam parameter group")
        group = optimizer.param_groups[0]
        if [id(p) for p in group["params"]] != [id(p) for p in old]:
            raise ValueError("Optimizer does not exactly own this model")
        self.validate_parameters()
        branch = self.grow()
        # New Adam moments are created by Adam on the first useful gradient.
        # Extending the same group also preserves the existing learning-rate schedule.
        group["params"].extend(branch.parameters())
        return branch

    def _run(self, ids, events, state, ablation=None):
        if ids.ndim != 2 or ids.dtype != torch.long or ids.size(1) < 1:
            raise ValueError("Expected nonempty [batch,time] int64 IDs")
        offset = 0 if state is None else state.length
        if offset+ids.size(1) > self.max_context:
            raise ValueError("Frozen context length exceeded")
        if state is not None and len(state.caches) != len(self.branches):
            raise ValueError("State predates growth; prefill again")
        selected = ids.new_full((ids.size(0),), CLEAR) if state is None else state.query
        records, sources, epochs = _receipts(ids, state, events)
        query = IntervalModel._queries(self, ids, selected)
        idx = query.clamp_min(0)[:, :, None, None].expand(-1, -1, 1, 3)
        permit, lower, upper = records.gather(2, idx).squeeze(2).unbind(-1)
        active = query != CLEAR
        facts = (active & (permit == 1)).float()+(active & (lower >= 0) & (upper >= 0) & (lower >= upper)).float()
        scores, caches = None, []
        with torch.autocast(device_type=ids.device.type, enabled=False):
            for i, branch in enumerate(self.branches):
                values, updated = branch(ids, offset, None if state is None else state.caches[i], ablation)
                caches.append(updated)
                if scores is None:
                    scores = values
                else:
                    scores = torch.where(torch.signbit(scores), -((-scores)-values), scores+values).clamp(-16, 16)
            residual = scores[..., CONFIRM]
            if self.arm == "certified_ssm":
                residual = residual.tanh().clamp(-1, 1)
            logits = scores.clone()
            logits[..., CONFIRM] = scores[..., DEFER]+((2*facts-3)+.5*residual)
        return logits, State(tuple(caches), query[:, -1].clone(), records[:, -1].clone(), sources, epochs,
                             offset+ids.size(1))

    def forward(self, ids, events=None):
        return self._run(ids, events, None)[0]

    def forward_without_recurrence(self, ids, events=None):
        return self._run(ids, events, None, "recurrence")[0]

    def forward_without_retrieval(self, ids, events=None):
        return self._run(ids, events, None, "retrieval")[0]

    def prefill(self, ids, events=None):
        return self._run(ids, events, None)

    def decode_step(self, ids, state, events=None):
        if ids.ndim != 1:
            raise ValueError("Expected one token per batch row")
        logits, updated = self._run(ids[:, None], None if events is None else events[:, None], state)
        return logits[:, 0], updated


def make_model(arm, width=32, depth=2, max_context=1024):
    return QualificationModel(arm, width, depth, max_context)
