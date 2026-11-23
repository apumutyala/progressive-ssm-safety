"""Controlled low-dimensional checks for the fitted-model endpoint diagnostic."""
import unittest

import torch

from control_probe import _linear_factors, analyze
from corpus_model import BOS, ByteTokenizer, make_model


class ControlProbeTests(unittest.TestCase):
    def scalar_model(self):
        model = make_model("certified", 1)
        r, b = ByteTokenizer.encode("RB")
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.blocks[0].drive.weight[r] = 1.0
            model.blocks[0].drive.weight[b] = -1.0
            model.blocks[0].head.weight[r, 0] = 1.0
        return model

    def test_known_three_step_gramians_and_receipt_intervention(self):
        model = self.scalar_model()
        for p in model.parameters():
            p.grad = torch.full_like(p, 0.125)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        with torch.inference_mode():
            out = analyze(model, torch.tensor([ByteTokenizer.encode("xyz")]))
        probe = out["development_window"]
        # Scalar A=B=1/2, C=1. Each spectrum has one eigenvalue
        # (1/2)^2 + (1/4)^2 + (1/8)^2 = 21/64.
        for kind in ("observability", "controllability"):
            self.assertEqual(probe[kind]["numerical_rank"], 1)
            self.assertAlmostEqual(probe[kind]["trace"], 21 / 64, places=10)
        self.assertEqual(probe["final_local_state_transition_induced_linf"], 0.125)
        self.assertEqual(probe["terminal_output_state_gradient_l2"], 1.0)
        self.assertEqual(probe["initial_pulse_to_terminal_contrast"], 0.125)
        typed = out["receipt_intervention"]
        self.assertTrue(typed["decision_pair_order_flipped"])
        self.assertTrue(typed["learned_final_state_bits_equal"])
        self.assertTrue(typed["ordinary_final_logits_bits_equal"])
        self.assertEqual(typed["discrete_margin_change"], 2.0)
        self.assertIsNone(model._audit)
        for name, p in model.named_parameters():
            self.assertTrue(torch.equal(p, before[name]))
            self.assertTrue((p.grad == 0.125).all())

    def test_state_remembers_fact_even_when_readout_cannot_see_it(self):
        model = self.scalar_model()
        with torch.no_grad():
            model.blocks[0].head.weight.zero_()
        out = analyze(model, torch.tensor([ByteTokenizer.encode("xyz")]))
        self.assertEqual(out["development_window"]["observability"]["numerical_rank"], 0)
        self.assertEqual(out["development_window"]["controllability"]["numerical_rank"], 1)
        pairs = out["public_card_substitution"]["by_lag"]
        self.assertEqual([p["identical_suffix_bytes"] for p in pairs], [0, 1, 16, 64])
        self.assertGreater(pairs[0]["hidden_linf_difference"], 0)
        self.assertTrue(all(p["output_contrast_change"] == 0 for p in pairs))

    def test_clamp_convention_and_unconstrained_factor(self):
        p = torch.tensor([[-2., -1., 0., 1., 2.]], dtype=torch.float64)
        keep = torch.full_like(p, 0.75)
        direction = torch.ones(5, dtype=torch.float64)
        a, b = _linear_factors(p, keep, direction, certified=True)
        self.assertEqual(a.tolist(), [[0., 0.75, 0.75, 0.75, 0.]])
        self.assertEqual(b.tolist(), [[0., 0.25, 0.25, 0.25, 0.]])
        a, b = _linear_factors(p, keep, direction, certified=False)
        self.assertTrue((a == 0.75).all() and (b == 0.25).all())

    def test_only_ordinary_development_window_is_admitted(self):
        with self.assertRaises(ValueError):
            analyze(self.scalar_model(), torch.tensor([[BOS]]))


if __name__ == "__main__":
    unittest.main()
