import unittest

from evidence_control import Contrast, OFFSET, _probe_prediction, margin_scores, public_contrasts


class PureEvidenceControlTests(unittest.TestCase):
    def test_margin_competes_against_reserved_ids_and_preserves_argmax_ties(self):
        values = [0.] * 264
        values[2], values[73], values[74] = 3., 2., -1.
        got = margin_scores(values, 73, 74)
        self.assertEqual(got["named_foil_margin"], 3.)
        self.assertEqual(got["strict_full_margin"], -1.)
        self.assertEqual(got["strongest_rival_id"], 2)
        values[2], values[74] = 0., 2.
        tied = margin_scores(values, 73, 74)
        self.assertTrue(tied["expected_first_index_argmax"])
        self.assertEqual(tied["strict_full_margin"], 0.)

    def test_disjoint_named_public_histories_and_unresolved_centroid_tie(self):
        train = public_contrasts("probe_train")
        test = public_contrasts("diagnostic")
        for pair in train + test:
            self.assertEqual(pair.branch(), (0, (ord("Y") + OFFSET, ord("N") + OFFSET)))
        self.assertEqual({p.kind for p in test}, {"evidence", "revoke", "long_distractor"})
        self.assertFalse({x+p.suffix for p in train for x in p.prefixes} &
                         {x+p.suffix for p in test for x in p.prefixes})
        self.assertIsNone(_probe_prediction([0.], {"a": [-1.], "b": [1.]}))


class ExecutedEvidenceControlTests(unittest.TestCase):
    """Future supervisor-only CPU fixtures. Authoring/AST validation needs no Torch."""
    def setUp(self):
        import torch
        from corpus_model import make_model
        self.addCleanup(torch.use_deterministic_algorithms, torch.are_deterministic_algorithms_enabled(),
                        warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
        self.addCleanup(torch.set_float32_matmul_precision, torch.get_float32_matmul_precision())
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        with torch.random.fork_rng(devices=[]):
            self.model = make_model("certified", 1).eval()
        with torch.no_grad():
            for p in self.model.parameters():
                p.zero_()
            drive = self.model.blocks[0].drive.weight
            drive[ord("+")+OFFSET] = 1.
            drive[ord("-")+OFFSET] = -1.
        self.train = Contrast("train", "evidence", "probe_train", (b"T+",b"T-"), b"?", (b"Ay",b"An"), ("yes","no"))
        self.test = Contrast("test", "evidence", "diagnostic", (b"E+",b"E-"), b"?", (b"Ay",b"An"), ("yes","no"))

    def run_probe(self):
        from evidence_control import analyze_pair, fit_probe
        frozen = fit_probe(self.model, [self.train])
        return analyze_pair(self.model, self.test, frozen)

    def test_readable_evidence_can_have_no_causal_effect_on_output(self):
        report = self.run_probe()
        self.assertEqual(report["branch_zero_based"], 1)
        self.assertTrue(report["swap_reproduced_donor_full_logits"])
        self.assertTrue(report["zeroed_pair_full_logits_equal"])
        for row in report["rows"]:
            outcomes = row["interventions"]
            self.assertEqual(outcomes["intact"]["probe_label"], row["oracle_label"])
            self.assertEqual(outcomes["swap"]["probe_label"], row["donor_label"])
            self.assertFalse(outcomes["intact"]["raw"]["prefix_matches_reference"])
            self.assertTrue(outcomes["intact"]["raw"]["protected_state_unchanged"])
            actual = outcomes["swap"]["finite_intervention"]["named_foil"]
            self.assertEqual(actual["local_jvp_prediction"], 0.)
            self.assertEqual(actual["actual_margin_change"], 0.)

    def test_linear_head_has_correct_finite_swap_jvp(self):
        import torch
        with torch.no_grad():
            head = self.model.blocks[0].head.weight
            head[ord("y")+OFFSET, 0] = .5
            head[ord("n")+OFFSET, 0] = -.5
        for row in self.run_probe()["rows"]:
            actual = row["interventions"]["swap"]["finite_intervention"]["named_foil"]
            self.assertLess(actual["actual_margin_change"], 0.)
            self.assertAlmostEqual(actual["actual_margin_change"], actual["local_jvp_prediction"], places=6)
