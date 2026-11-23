"""CPU endpoint-verifier checks. Execution belongs to the reviewed supervisor."""
import unittest

import torch

from model import Model
from prefix_bound import same_fp32_bits, verify_suffix


class PrefixBoundTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(911)
        self.model = Model(11, 3, 2, 3)
        self.tokens = torch.ones((2, 16), dtype=torch.long)
        with torch.no_grad():
            self.model.blocks[0].drive.weight.fill_(0.5)
            self.model.blocks[0].decay.weight.fill_(-16)

    def test_fast_learned_transition_coalesces_full_logits(self):
        report = verify_suffix(self.model, self.tokens)
        self.assertTrue(report["certified_prior_learned_state_independent_final_logits"])
        self.assertEqual(report["max_final_hidden_width"], 0)
        self.assertTrue(report["endpoint_final_logits_equal_bits"])
        premises = torch.zeros((2, 16, 2), dtype=torch.bool)
        baseline = None
        for initial in (-1.0, -0.5, -0.0, 0.0, 0.25, 0.75, 1.0):
            states = (torch.full((2, 3), initial),)
            logits, _ = self.model._score(self.tokens, states, premises)
            if baseline is None:
                baseline = logits[:, -1].clone()
            self.assertTrue(same_fp32_bits(baseline, logits[:, -1]).all())

    def test_long_memory_is_inconclusive_despite_constant_readout(self):
        with torch.no_grad():
            self.model.blocks[0].decay.weight.fill_(16)
            self.model.blocks[0].head.weight.zero_()
            self.model.blocks[0].head.bias.zero_()
        report = verify_suffix(self.model, self.tokens)
        self.assertFalse(report["certified_prior_learned_state_independent_final_logits"])
        self.assertGreater(report["max_final_hidden_width"], 1)
        self.assertTrue(report["endpoint_final_logits_equal_bits"])

    def test_all_blocks_must_coalesce_even_if_added_head_is_silent(self):
        new = self.model.grow()
        with torch.no_grad():
            new.decay.weight.fill_(16)
        report = verify_suffix(self.model, self.tokens)
        self.assertTrue(report["blocks"][0]["coalesced_nonzero_everywhere"])
        self.assertFalse(report["blocks"][1]["coalesced_nonzero_everywhere"])
        self.assertFalse(report["certified_prior_learned_state_independent_final_logits"])

    def test_signed_zero_bits_are_distinguished_and_zero_is_conservative(self):
        plus, minus = torch.tensor([0.0]), torch.tensor([-0.0])
        self.assertFalse(same_fp32_bits(plus, minus).item())
        with torch.no_grad():
            self.model.blocks[0].drive.weight.zero_()
        report = verify_suffix(self.model, torch.ones((2, 64), dtype=torch.long))
        self.assertGreater(report["blocks"][0]["zero_singleton_coordinates"], 0)
        self.assertFalse(report["certified_prior_learned_state_independent_final_logits"])

    def test_protected_context_is_fixed_and_cannot_be_inferred(self):
        empty = verify_suffix(self.model, self.tokens)
        supported = verify_suffix(self.model, self.tokens, torch.ones((2, 16, 2), dtype=torch.bool))
        self.assertTrue(empty["certified_prior_learned_state_independent_final_logits"])
        self.assertTrue(supported["certified_prior_learned_state_independent_final_logits"])
        self.assertNotEqual(empty["protected_premises"], supported["protected_premises"])
        for invalid in (torch.zeros((2, 16, 2)), torch.zeros((2, 16, 3), dtype=torch.bool)):
            with self.assertRaises(ValueError):
                verify_suffix(self.model, self.tokens, invalid)

    def test_invalid_endpoint_execution_is_rejected(self):
        with torch.no_grad():
            self.model.blocks[0].drive.weight[1, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            verify_suffix(self.model, self.tokens)


if __name__ == "__main__":
    unittest.main()
