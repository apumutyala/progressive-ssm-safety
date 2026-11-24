"""Tiny algebraic fixtures; no model, text, network, or numerical runtime."""
import unittest

from grammar_prefix_bound import bound_expected_byte, real_suffix_box


def blank_block(lo=-1., hi=1.):
    return {"hidden_lower": [lo], "hidden_upper": [hi],
            "mix_weight": [[1.]], "mix_bias": [0.],
            "head_weight": [[0., 0.] for _ in range(264)], "head_bias": [-1.] * 264}


class GrammarPrefixBoundTests(unittest.TestCase):
    def test_common_suffix_can_certify_byte_with_nonzero_box(self):
        lo, hi = real_suffix_box([[.5], [.5]], [[1.], [1.]])
        self.assertEqual((lo, hi), ([.5], [1.]))
        block = blank_block(lo[0], hi[0])
        block["head_weight"][73] = [0., 1.]
        block["head_bias"][73] = 0.
        block["head_bias"][74] = .4
        block["head_bias"][4] = 1.  # unbound CONFIRM replaced below DEFER
        result = bound_expected_byte(73, [block])
        # tanh(.5) ~= .4621, and the worst ordinary competitor is .4.
        self.assertTrue(result["real_readout_unique_expected_byte"])
        self.assertTrue(.06 < result["real_readout_margin_lower"] < .063)
        self.assertTrue(.030 < result["uniform_logit_error_must_be_strictly_below"] < .032)

    def test_negative_bound_can_be_inconclusive_due_to_dependency(self):
        block = blank_block()
        block["head_weight"][73] = [1., 0.]
        block["head_weight"][74] = [1., 0.]
        block["head_bias"][73] = .25
        block["head_bias"][74] = 0.
        # The real pointwise gap is .25 at *every* state. Separate score
        # intervals lose their shared h and therefore cannot certify it here.
        result = bound_expected_byte(73, [block])
        self.assertFalse(result["real_readout_unique_expected_byte"])
        self.assertEqual(result["real_readout_margin_lower"], -1.75)

    def test_reserved_competition_and_multiblock_clipping(self):
        block = blank_block()
        block["head_bias"][73] = .5
        block["head_bias"][3] = .75
        self.assertEqual(bound_expected_byte(73, [block])["real_readout_margin_lower"], -.25)
        block["head_bias"][73] = 1.
        block["head_bias"][3] = .875
        self.assertEqual(bound_expected_byte(73, [block] * 20)["real_readout_margin_lower"], 0.)

    def test_rejects_invalid_suffix_box_or_target(self):
        with self.assertRaises(ValueError):
            real_suffix_box([[1.1]], [[0.]])
        with self.assertRaises(ValueError):
            bound_expected_byte(73, [blank_block(.5, -.5)])
        with self.assertRaises(ValueError):
            bound_expected_byte(3, [blank_block()])
