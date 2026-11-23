"""Two-plan history and scoped semantic checks for the supervised CPU test worker."""

import random
import unittest

import torch

from interval_model import CLEAR, HOLD, MAX_UNITS, IntervalModel


CONFIRM, DEFER, ASK_A, ASK_B, WORD = range(5)


def history_oracle(tokens, receipts):
    """Scalar reference from the receipt protocol; returns ground truth at each step."""
    records = [[None] * 3 for _ in range(2)]
    query = None
    trace = []
    for token, receipt in zip(tokens, receipts):
        for plan, fields in enumerate(receipt):
            for field, replacement in enumerate(fields):
                if replacement == HOLD:
                    continue
                maximum = 1 if field == 0 else MAX_UNITS
                records[plan][field] = replacement if 0 <= replacement <= maximum else None
        if query is None and token in (ASK_A, ASK_B):
            query = (ASK_A, ASK_B).index(token)
        supported = False
        if query is not None:
            permit, lower, upper = records[query]
            supported = permit == 1 and lower is not None and upper is not None and lower >= upper
        snapshot = [[CLEAR if value is None else value for value in plan] for plan in records]
        trace.append((supported, snapshot, CLEAR if query is None else query))
    return trace


class IntervalCertificateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(421)
        self.model = IntervalModel(8, 6, CONFIRM, DEFER, ASK_A, ASK_B)

    def test_query_scope_replacements_revocation_unknown_and_positive_choice(self):
        tokens = [[WORD, ASK_A, WORD, ASK_B, ASK_A, WORD, WORD, ASK_B, WORD, WORD],
                  [WORD, ASK_B, WORD, ASK_A, ASK_B, WORD, WORD, ASK_A, WORD, WORD]]
        receipts = torch.full((1, 10, 2, 3), HOLD, dtype=torch.int64)
        receipts[0, 0] = torch.tensor([[1, 7, 7], [1, 5, 6]])
        receipts[0, 2, 1, 2] = 4        # change unqueried plan B
        receipts[0, 4, 0, 1] = 6        # A's replacement lower bound is insufficient
        receipts[0, 5, 0, 2] = 0        # zero is a known bound
        receipts[0, 6, 0, 0] = 0        # revoke permit on A only
        receipts[0, 8, 1, 1] = CLEAR    # unknown B capacity
        receipts[0, 9, 1, 1] = 4        # restore B capacity at equality
        with torch.no_grad():
            head = self.model.blocks[0].head
            head.weight.zero_()
            head.bias.zero_()
            head.bias[DEFER] = 1
            logits = self.model(torch.tensor(tokens), receipts.expand(2, -1, -1, -1))
        expected = [
            [False, True, True, True, False, True, False, False, False, False],
            [False, False, True, True, True, True, True, True, False, True],
        ]
        for i, decisions in enumerate(expected):
            self.assertEqual([r[0] for r in history_oracle(tokens[i], receipts[0].tolist())], decisions)
            self.assertEqual(logits.argmax(-1)[i].tolist(), [CONFIRM if ok else DEFER for ok in decisions])

    def test_scalar_oracle_matches_prefill_decode_and_growth(self):
        rng = random.Random(918)
        batch, length = 8, 22
        tokens = [[rng.randrange(8) for _ in range(length)] for _ in range(batch)]
        alphabet = (HOLD, HOLD, HOLD, CLEAR, 0, 1, 4, MAX_UNITS, MAX_UNITS + 1, -(2**63), 2**63 - 1)
        receipts = [
            [[[rng.choice(alphabet) for _ in range(3)] for _ in range(2)] for _ in range(length)]
            for _ in range(batch)
        ]
        # Include boundary support then cross-plan missing evidence for each row.
        for row in range(batch):
            tokens[row][:2] = [ASK_A, ASK_B] if row % 2 == 0 else [ASK_B, ASK_A]
            receipts[row][0] = [[1, MAX_UNITS, MAX_UNITS], [0, 3, 0]]
            receipts[row][1] = [[HOLD] * 3, [HOLD] * 3]
        reference = [history_oracle(t, e) for t, e in zip(tokens, receipts)]
        ids, events = torch.tensor(tokens), torch.tensor(receipts)
        with torch.no_grad():
            full, final = self.model.prefill(ids, events)
            state = self.model.initial_state(ids)
            for t in range(length):
                step, state = self.model.decode_step(ids[:, t], state, events[:, t])
                torch.testing.assert_close(step, full[:, t], rtol=1e-5, atol=1e-5)
                self.assertEqual(state.records.tolist(), [row[t][1] for row in reference])
                self.assertEqual(state.query.tolist(), [row[t][2] for row in reference])
                signed = step[:, CONFIRM] - step[:, DEFER]
                supported = torch.tensor([row[t][0] for row in reference])
                self.assertTrue(torch.where(supported, signed >= 0.49999, signed <= -0.49999).all())
            self.assertTrue(torch.equal(final.records, state.records))
            self.model.grow()
            grown, final_grown = self.model.prefill(ids, events)
            self.assertTrue(torch.equal(grown, full))
            self.assertTrue(torch.equal(final.records, final_grown.records))
            self.assertTrue(torch.equal(final.query, final_grown.query))

    def test_free_text_and_generated_tokens_never_write_receipts(self):
        tokens = torch.tensor([[ASK_A, CONFIRM, WORD, ASK_B, DEFER, ASK_A]])
        with torch.no_grad():
            logits, state = self.model.prefill(tokens)
            self.assertTrue((state.records == CLEAR).all())
            self.assertEqual(state.query.item(), 0)
            self.assertFalse((logits.argmax(-1) == CONFIRM).any())
            with self.assertRaises(ValueError):
                self.model(tokens, torch.zeros((1, 6, 2, 3), dtype=torch.float32))

    def test_generated_query_retarget_counterexample_and_first_query_fix(self):
        # Preserve a counterexample to latest-query binding in a test fixture.
        # Both models have identical weights, inputs, and receipt history.
        class LatestQueryBaseline(IntervalModel):
            def _queries(self, tokens, previous):
                active = previous
                sequence = []
                for token in tokens.unbind(1):
                    active = torch.where(token == self.query_a_id, 0, active)
                    active = torch.where(token == self.query_b_id, 1, active)
                    sequence.append(active)
                return torch.stack(sequence, dim=1)

        with torch.no_grad():
            head = self.model.blocks[0].head
            head.weight.zero_()
            head.bias.zero_()
            head.bias[DEFER] = 0.5
            head.bias[ASK_B] = 1
            receipts = torch.tensor([[[[0, 0, 1], [1, 1, 1]]]])
            old = LatestQueryBaseline(8, 6, CONFIRM, DEFER, ASK_A, ASK_B)
            old.load_state_dict(self.model.state_dict())
            for model, expected_second, expected_plan in ((old, CONFIRM, 1), (self.model, ASK_B, 0)):
                first, state = model.prefill(torch.tensor([[ASK_A]]), receipts)
                generated_query = first[:, -1].argmax(-1)
                self.assertEqual(generated_query.item(), ASK_B)
                second, state = model.decode_step(generated_query, state)
                self.assertEqual(second.argmax(-1).item(), expected_second)
                self.assertEqual(state.query.item(), expected_plan)
                self.assertTrue(torch.equal(state.records, receipts[:, 0]))
            # Fresh episodes can bind B. An earlier unescaped query token also
            # binds B, illustrating why framed input provenance is required.
            prefixed = torch.tensor([[ASK_B, WORD, ASK_A]])
            first = torch.full((1, 3, 2, 3), HOLD)
            first[:, :1] = receipts
            outputs, state = self.model.prefill(prefixed, first)
            self.assertEqual(state.query.item(), 1)
            self.assertEqual(outputs[0, -1].argmax().item(), CONFIRM)


if __name__ == "__main__":
    unittest.main()
