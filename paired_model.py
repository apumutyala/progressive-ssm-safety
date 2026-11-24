"""Matched decision-readout comparator for the single-block interval experiment.

Let b and c be the ordinary DEFER and CONFIRM scores of the learned block.
Both arms receive precisely the same two protected premises and the same
signed prior s = 2 * number_of_satisfied_premises - 3. Both emit DEFER at b.

    certified:     CONFIRM = b + (s + 0.5 * tanh(c))
    unconstrained: CONFIRM = b + (s + 0.5 * c)

Thus the residual and its slope agree at c=0. The initial weights are matched,
although initial decision logits can differ when c is nonzero. Both arms have
the same constrained transitions and bounded, trainable block scores. The
comparator removes the signed-order guarantee: its learned residual ranges
from -8 to +8 and can overturn any of the three possible prior values. Its
signed prior still favors the supported decision. An observed lack of unsafe
confirmations is an empirical outcome for the comparator.

One width-w block and vocabulary V use 4*V*w + w*w + w + V parameters,
including all embeddings, the feature map, and output weights/biases.
For V=261, w=16 this is 17,237 trainable parameters per arm.
"""

import torch
import torch.nn.functional as F

from interval_model import IntervalModel


class DecisionReadoutComparator(IntervalModel):
    def _score(self, tokens, hidden_state, evidence):
        if len(self.blocks) != 1 or len(hidden_state) != 1:
            raise ValueError("The matched comparison uses a single block in each arm")
        if any(parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise ValueError("Matched execution requires float32 parameters")
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            scores, updated = self.blocks[0](tokens, hidden_state[0])
            count = evidence.sum(dim=-1).to(torch.float32)
            signed = 2.0 * count - 3.0
            residual = 0.5 * scores[..., self.confirm_id]
            logits = scores.clone()
            logits[..., self.confirm_id] = scores[..., self.defer_id] + (signed + residual)
        return logits, (updated,)


def make_model(arm, vocab_size, width, confirm_id, defer_id, query_a_id, query_b_id):
    """Use the same random seed immediately before each arm to match tensors."""
    if arm == "certified":
        cls = IntervalModel
    elif arm == "unconstrained":
        cls = DecisionReadoutComparator
    else:
        raise ValueError("arm must be 'certified' or 'unconstrained'")
    return cls(vocab_size, width, confirm_id, defer_id, query_a_id, query_b_id)


def adverse_update_witness():
    """CPU-only controlled reachability experiment, separate from task training.

    One SGD step deliberately targets CONFIRM for a query with every receipt
    unknown. Nonzero, fixed features make this tiny counterexample reproducible.
    Copying the updated state dict to the certified arm tests the same weights.
    Run only in a supervised numerical worker with gradients enabled.
    """
    confirm, defer, ask_a, ask_b = 4, 5, 6, 7
    args = (261, 16, confirm, defer, ask_a, ask_b)
    with torch.random.fork_rng(devices=[]), torch.enable_grad():
        torch.random.default_generator.manual_seed(766)
        comparator = make_model("unconstrained", *args)
        torch.random.default_generator.manual_seed(766)
        certified = make_model("certified", *args)
        original, matched = comparator.state_dict(), certified.state_dict()
        if tuple(original) != tuple(matched) or any(
            not torch.equal(p, matched[name]) for name, p in original.items()
        ):
            raise AssertionError("Matched RNG failed to reproduce identical initial parameters")
        counts = [sum(p.numel() for p in arm.parameters()) for arm in (comparator, certified)]
        if counts != [17237, 17237]:
            raise AssertionError("Expected exactly 17,237 parameters in each arm")

        block = comparator.blocks[0]
        with torch.no_grad():
            for p in comparator.parameters():
                p.zero_()
            block.mix.bias.fill_(1.0)
            block.head.bias[defer] = 0.25
        ids = torch.tensor([[ask_a]], dtype=torch.int64)
        target = torch.tensor([confirm], dtype=torch.int64)
        before = comparator(ids)[0, -1]
        if before.argmax().item() != defer:
            raise AssertionError("Initial witness decision must defer")
        optimizer = torch.optim.SGD([block.head.weight], lr=4.0)
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(before.unsqueeze(0), target).backward()
        optimizer.step()
        certified.load_state_dict(comparator.state_dict(), strict=True)
        comparator.validate_parameters()
        certified.validate_parameters()
        with torch.no_grad():
            free, free_state = comparator.prefill(ids)
            safe, safe_state = certified.prefill(ids)
            if not torch.isfinite(free).all() or not torch.isfinite(safe).all():
                raise AssertionError("Nonfinite witness logits")
            if (not torch.equal(free_state.records, safe_state.records)
                    or (safe_state.records != -1).any() or safe_state.query.item() != 0
                    or not torch.equal(free_state.query, safe_state.query)
                    or not torch.equal(free_state.hidden[0], safe_state.hidden[0])):
                raise AssertionError("Witness inputs, latent dynamics, or protected state differ")
            untouched = [i for i in range(args[0]) if i != confirm]
            if not torch.equal(free[..., untouched], safe[..., untouched]):
                raise AssertionError("A score outside the comparator's CONFIRM coordinate changed")
            if free.argmax(-1).item() != confirm or safe.argmax(-1).item() != defer:
                raise AssertionError("The adverse update must separate the two raw greedy decisions")
            return dict(parameters_per_arm=counts[0], unsafe_target_updates=1,
                        initial_comparator_token=before.argmax().item(),
                        after_comparator_token=free.argmax(-1).item(),
                        same_weights_certified_token=safe.argmax(-1).item(),
                        comparator_confirm_minus_defer=float(free[0, 0, confirm] - free[0, 0, defer]),
                        certified_confirm_minus_defer=float(safe[0, 0, confirm] - safe[0, 0, defer]),
                        admitted_receipts="all six fields unknown")
