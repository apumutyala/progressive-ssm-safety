"""CPU cases for the rounded endpoint theorem; run inside the study supervisor."""
from dataclasses import replace
import unittest

import torch

from corpus_model import BYTE_OFFSET, CONFIRM, make_model
from corpus_sensitivity import _fp32_units, verify_suffix


class SuffixSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.rng = torch.random.fork_rng(devices=[])
        self.rng.__enter__()
        self.addCleanup(self.rng.__exit__, None, None, None)
        torch.manual_seed(23)
        self.addCleanup(torch.use_deterministic_algorithms, torch.are_deterministic_algorithms_enabled(),
                        warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
        self.addCleanup(torch.set_float32_matmul_precision, torch.get_float32_matmul_precision())
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        self.model = make_model("certified", width=4).eval()
        self.tokens = torch.tensor([[BYTE_OFFSET + 32, BYTE_OFFSET + 65] * 8,
                                    [BYTE_OFFSET + 123, BYTE_OFFSET + 14] * 8])

    @torch.no_grad()
    def test_intermediate_seeds_enclosed_on_executed_multiblock_path(self):
        self.model.grow()
        state = self.model.initial_state(self.tokens)
        for length in (1, 4, 16):
            def run(hidden):
                _, out = self.model._run(self.tokens[:, :length], replace(state, hidden=hidden), None)
                return torch.cat(out.hidden, dim=-1)

            low = run(tuple(torch.full_like(h, -1) for h in state.hidden))
            high = run(tuple(torch.ones_like(h) for h in state.hidden))
            for _ in range(3):
                mid = run(tuple(torch.rand_like(h) * 2 - 1 for h in state.hidden))
                self.assertTrue(((low <= mid) & (mid <= high)).all().item())
            report = verify_suffix(self.model, self.tokens[:, :length])
            self.assertGreaterEqual(report["max_final_hidden_width_upper_bound"],
                                    (high.double() - low.double()).max().item())

    @torch.no_grad()
    def test_slow_retention_equal_output_does_not_certify_erasure(self):
        block = self.model.blocks[0]
        block.decay.weight.fill_(16)
        block.drive.weight.zero_()
        block.head.weight.zero_()
        report = verify_suffix(self.model, self.tokens)
        self.assertGreater(report["max_final_hidden_width_upper_bound"], 1.8)
        self.assertEqual(report["observed_endpoint_final_logit_equal_rows"], 2)
        self.assertEqual(report["certified_independent_final_logit_rows"], 0)

    @torch.no_grad()
    def test_nonzero_coalescence_and_zero_ambiguity(self):
        block = self.model.blocks[0]
        block.decay.weight.fill_(-16)
        block.drive.weight.fill_(1)
        report = verify_suffix(self.model, self.tokens)
        self.assertEqual(report["max_final_hidden_width_upper_bound"], 0)
        self.assertEqual(report["certified_independent_final_logit_rows"], 2)
        block.drive.weight.zero_()
        report = verify_suffix(self.model, self.tokens.repeat(1, 4))
        self.assertEqual(report["zero_singleton_coordinates"], 8)
        self.assertEqual(report["certified_independent_final_logit_rows"], 0)

    def test_excludes_specials_and_unconstrained_dynamics(self):
        with self.assertRaises(ValueError):
            verify_suffix(self.model, torch.tensor([[CONFIRM]]))
        with self.assertRaises(ValueError):
            verify_suffix(make_model("unconstrained", 4), self.tokens)

    def test_exact_reporting_units_for_subnormals_and_signed_zero(self):
        self.assertEqual(_fp32_units(1), 1)
        self.assertEqual(_fp32_units(0x80000001), -1)
        self.assertEqual(_fp32_units(0x80000000), 0)
        self.assertEqual(_fp32_units(0x00800000), 1 << 23)
        self.assertEqual(_fp32_units(0xBF800000), -(1 << 149))
        with self.assertRaises(FloatingPointError):
            _fp32_units(0x7FC00001)
