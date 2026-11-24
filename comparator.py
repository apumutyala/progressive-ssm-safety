"""Lazy, parameter-matched GRU comparator; no experiment is launched here.

``make_gru_model(width=18, blocks=2)`` returns a trainable byte language model
with the existing forward/prefill/decode_step/events/grow/validate_parameters
API. Importing this module, counting parameters and reading its contract use
only the standard library. Torch and the model modules load inside the factory.

Both arms inherit CorpusModel._score, so they execute the *same* bounded dense
readout, signed reserved-token ranking, and signed-zero-preserving block sum.
Both inherit the authentic integer record transition and first-query latch.
The GRU changes learned memory, not who may authenticate a premise. Neither the
typed channel nor its ranking is evidence of learned alignment or correct prose.

The GRU follows the reset-after-hidden-affine convention of PyTorch GRUCell:
  r=sigmoid(i_r+h_r), z=sigmoid(i_z+h_z),
  n=tanh(i_n+r*h_n), h'=z*h+(1-z)*n.
Its embeddings/preactivations, effective affine weights and executed hidden
states are bounded explicitly. This is a bounded GRU comparator, not an
unconstrained stock GRUCell. In particular z alone is NOT a diagonal transition
Jacobian: r, z and n depend on h. Do not apply the SSM contraction theorem or
its retention-in-[1/256,255/256] admission test to this model. The shared hidden
box and protected readout remain meaningful, separate contracts.

For the fixed 264-token vocabulary, one SSM width16 block has 17,432 parameters;
one GRU width18 block has 16,914. Width18 is the closest integer width in this
specified family (embedding dimension equals hidden width). Two blocks have
33,828 versus 34,864 parameters: 1,036 fewer, or 2.9715%. Hidden coordinates are
36 versus 32; state size is not exactly matched. There are no unused padding
parameters. The analytic dense-MAC counts are 23,544 versus 17,408 per token
for two blocks; these omit lookups, nonlinear/scalar operations, clamps,
training, audits and I/O, and are NOT measured FLOPs or equal-compute training.

The future runner must use identical approved corpus/splits, byte codec,
ordinary/evidence mixing, loss masks/weights, Adam recipe, clipping, data order,
context and development windows, endpoint/update schedule and authority inputs.
Report actual parameters, all cumulative target exposures, wall time and compute
for both arms. Use complete-vocabulary raw greedy outputs and keep stochastic
samples separate. Freeze width before seeing outcomes. A GRU scratch arm must
receive the same pretraining/mixing budget as an SSM scratch arm; a checkpoint
from a different backbone cannot be advertised as equivalent initialization.

All initial heads are active. ``grow(silent_head=True)`` appends an independent
silent block, as in the SSM; it is not stacked recurrent depth. A runner must
add new parameters to Adam while retaining old moments, and test full insertion
logits on an unchanged math path. Existing strict ``type(model) is CorpusModel``
probes intentionally reject this subclass until explicitly generalized. Pending
runtime QA: actual parameter counts, finite forward/backward, optimizer update,
prefill/decode agreement, protected premise cases, and growth/logit parity.
The fixed-count original pretrain.Worker also needs an explicit comparator
runner; do not bypass its architecture admission. Serialize state_dict plus
source/configuration pins, not the lazily defined runtime class object.
No fitted or numerical result for this comparator is claimed.
"""
from functools import lru_cache


VOCABULARY_SIZE = 264
DEFAULT_SSM_WIDTH = 16
DEFAULT_GRU_WIDTH = 18
DEFAULT_BLOCKS = 2


def _integer(value, label, upper=8192):
    if type(value) is not int or not 1 <= value <= upper:
        raise ValueError(f"{label} must be an integer in 1..{upper}")
    return value


def parameter_count(backbone, width, blocks=1, vocab_size=VOCABULARY_SIZE):
    """Exact trainable count for this source's SSM or equal-input-width GRU."""
    w = _integer(width, "width")
    b = _integer(blocks, "blocks")
    v = _integer(vocab_size, "vocabulary")
    if backbone == "ssm":
        # Two V*w tables, w*w+w mix, and (2*w)*V+V head.
        per_block = w*w + (4*v+1)*w + v
    elif backbone == "gru":
        # V*w embedding, two (3*w)*w+3*w gate affines, same mix/head.
        per_block = 7*w*w + (3*v+7)*w + v
    else:
        raise ValueError("backbone must be ssm or gru")
    return b * per_block


def nearest_gru_width(ssm_width=DEFAULT_SSM_WIDTH, vocab_size=VOCABULARY_SIZE):
    """Closest width in the declared family; ties favor the smaller model."""
    w = _integer(ssm_width, "SSM width")
    v = _integer(vocab_size, "vocabulary")
    target = parameter_count("ssm", w, vocab_size=v)
    # At 2*w (or the admitted upper width) the monotone GRU count exceeds
    # the SSM target, so no still-larger width can improve this minimum.
    upper = min(8192, 2*w)
    return min(range(1, upper+1), key=lambda h: (abs(parameter_count("gru", h, vocab_size=v)-target), h))


def analytic_comparison(ssm_width=DEFAULT_SSM_WIDTH, gru_width=DEFAULT_GRU_WIDTH,
                        blocks=DEFAULT_BLOCKS):
    """Scalar accounting only; dense MACs deliberately exclude other work."""
    s = _integer(ssm_width, "SSM width")
    g = _integer(gru_width, "GRU width")
    b = _integer(blocks, "blocks")
    v = VOCABULARY_SIZE
    small, comparator = parameter_count("ssm", s, b), parameter_count("gru", g, b)
    return dict(vocabulary_size=v, blocks=b, ssm_width=s, gru_width=g,
                ssm_hidden_coordinates=b*s, gru_hidden_coordinates=b*g,
                ssm_parameters=small, gru_parameters=comparator,
                gru_minus_ssm_parameters=comparator-small,
                gru_parameter_difference_percent=100*(comparator-small)/small,
                nearest_equal_embedding_gru_width=nearest_gru_width(s),
                ssm_dense_macs_per_token=b*(s*s+2*v*s),
                gru_dense_macs_per_token=b*(7*g*g+2*v*g))


@lru_cache(maxsize=1)
def _runtime_classes():
    # Keep all heavy imports behind the explicit model construction boundary.
    import torch
    from torch import nn
    from corpus_model import CorpusModel, VOCAB_SIZE, CONFIRM, DEFER, QUERY_A, QUERY_B
    from model import SelectiveBlock

    if VOCAB_SIZE != VOCABULARY_SIZE:
        raise ValueError("Comparator requires the unchanged 264-token byte codec")

    class GRUBlock(nn.Module):
        bounded_linear = staticmethod(SelectiveBlock.bounded_linear)

        def __init__(self, width, silent_head=False):
            super().__init__()
            self.width = width
            # drive is also the inherited initial_state allocation anchor.
            self.drive = nn.Embedding(VOCAB_SIZE, width)
            self.input_gates = nn.Linear(width, 3*width)
            self.hidden_gates = nn.Linear(width, 3*width)
            self.mix = nn.Linear(width, width)
            self.head = nn.Linear(2*width, VOCAB_SIZE)
            with torch.no_grad():
                nn.init.normal_(self.drive.weight, std=0.35)
                if silent_head:
                    self.head.weight.zero_()
                    self.head.bias.zero_()

        def forward(self, tokens, previous, *, certified, audit):
            if not certified:
                raise ValueError("GRU comparator shares the protected readout arm only")
            inputs = self.drive(tokens).clamp(-16, 16).tanh().clamp(-1, 1)
            trajectory, proposals, keeps, candidates, raw_keeps, raw_candidates = [], [], [], [], [], []
            for value in inputs.unbind(1):
                i_r, i_z, i_n = self.bounded_linear(value, self.input_gates).chunk(3, dim=-1)
                h_r, h_z, h_n = self.bounded_linear(previous, self.hidden_gates).chunk(3, dim=-1)
                reset = (i_r+h_r).clamp(-16, 16).sigmoid()
                raw_keep = i_z+h_z
                keep = raw_keep.clamp(-16, 16).sigmoid().clamp(0, 1)
                raw_candidate = i_n+reset*h_n
                candidate = raw_candidate.clamp(-16, 16).tanh().clamp(-1, 1)
                proposed = keep*previous+(1.0-keep)*candidate
                previous = proposed.clamp(-1, 1)
                trajectory.append(previous)
                if audit is not None:
                    proposals.append(proposed.detach())
                    keeps.append(keep.detach())
                    candidates.append(candidate.detach())
                    raw_keeps.append(raw_keep.detach())
                    raw_candidates.append(raw_candidate.detach())
            hidden = torch.stack(trajectory, dim=1)
            if not torch.isfinite(hidden).all().item():
                raise FloatingPointError("GRU comparator produced a nonfinite hidden trajectory")
            if audit is not None:
                # Existing field names: retention=z, drive=n; their raw fields
                # mean update/candidate preactivation here, not SSM embeddings.
                audit.record(hidden, torch.stack(proposals, dim=1), torch.stack(keeps, dim=1),
                             torch.stack(candidates, dim=1), torch.stack(raw_keeps, dim=1),
                             torch.stack(raw_candidates, dim=1))
            features = hidden.clamp(-16, 16)
            mixed = self.bounded_linear(features, self.mix).tanh().clamp(-1, 1)
            scores = self.bounded_linear(torch.cat((features, mixed), dim=-1), self.head)
            return scores.clamp(-16, 16), previous

    class GRUCorpusModel(CorpusModel):
        def __init__(self, width, blocks):
            nn.Module.__init__(self)
            self.arm = "certified"  # Select the identical *shared readout*.
            self.backbone_kind = "bounded_gru"
            self.width, self.vocab_size = width, VOCAB_SIZE
            self.confirm_id, self.defer_id = CONFIRM, DEFER
            self.query_a_id, self.query_b_id = QUERY_A, QUERY_B
            self.blocks = nn.ModuleList([GRUBlock(width) for _ in range(blocks)])
            self._audit = None

        def grow(self, silent_head=True):
            if type(silent_head) is not bool:
                raise ValueError("silent_head must be boolean")
            source = self.blocks[0].drive.weight
            block = GRUBlock(self.width, silent_head=silent_head)
            block.to(device=source.device, dtype=source.dtype)
            block.train(self.training)
            self.blocks.append(block)
            return block

    return GRUCorpusModel, GRUBlock


def make_gru_model(width=DEFAULT_GRU_WIDTH, blocks=DEFAULT_BLOCKS):
    """Construct active scratch blocks; caller owns seed, optimizer and device.

    Model/optimizer checkpoints must identify this source and backbone. This
    factory does not load an SSM checkpoint, reseed RNG, train, read data or set
    a device. It must be called only by an admitted numerical worker.
    """
    _integer(width, "width")
    _integer(blocks, "blocks")
    model_class, _ = _runtime_classes()
    return model_class(width, blocks)
