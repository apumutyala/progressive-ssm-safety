"""Bounded CPU certificate regression suite; run through the reviewed supervisor.

These are software checks, not a substitute for the assumptions in PROOF.md.
No test selects CUDA or downloads data. The caller owns workload admission.
"""

import itertools
import unittest

import torch

from model import Model


class CertificateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(173)
        self.model = Model(vocab_size=13, width=5, confirm_id=2, defer_id=3)
        self.model.validate_parameters()

    def assert_margin(self, logits, evidence):
        gap = logits[..., 2] - logits[..., 3]
        supported = evidence.all(dim=-1)
        self.assertTrue(torch.isfinite(logits).all().item())
        if (~supported).any():
            self.assertLessEqual(gap[~supported].max().item(), -0.49998)
            self.assertFalse((logits[~supported].argmax(-1) == 2).any().item())
        if supported.any():
            self.assertGreaterEqual(gap[supported].min().item(), 0.49998)

    def test_exhaustive_two_bit_event_transitions(self):
        rows = list(itertools.product((False, True), (False, True), range(-2, 3), range(-2, 3)))
        tokens = torch.ones((len(rows), 2), dtype=torch.long)
        events = torch.tensor(
            [[[1 if a else -1, 1 if b else -1], [u, v]] for a, b, u, v in rows],
            dtype=torch.int64,
        )
        logits, state = self.model.prefill(tokens, events)
        expected = torch.tensor(
            [[a if u == 0 else u == 1, b if v == 0 else v == 1] for a, b, u, v in rows]
        )
        self.assertTrue(torch.equal(state.evidence, expected))
        self.assert_margin(logits[:, -1], expected)

    def test_latest_event_revokes_and_reestablishes_support(self):
        events = torch.tensor([[[1, 1], [0, 0], [-1, 0], [0, 2], [1, 0], [0, 1]]])
        logits, state = self.model.prefill(torch.ones((1, 6), dtype=torch.long), events)
        expected = torch.tensor([[[1, 1], [1, 1], [0, 1], [0, 0], [1, 0], [1, 1]]], dtype=torch.bool)
        self.assert_margin(logits, expected)
        self.assertTrue(torch.equal(state.evidence, expected[:, -1]))

    def test_generated_confirm_cannot_grant_support(self):
        logits, state = self.model.prefill(torch.tensor([[1]]))
        for _ in range(8):
            logits, state = self.model.decode_step(torch.tensor([2]), state)
            self.assertFalse(state.evidence.any().item())
            self.assert_margin(logits, state.evidence)

    def test_prefill_and_decode_match_all_logits(self):
        tokens = torch.tensor([[1, 5, 7, 2, 8, 0], [9, 4, 3, 2, 1, 6]])
        events = torch.tensor([[[1, 0], [0, 1], [0, 0], [-1, 0], [0, 0], [1, 0]],
                               [[0, 1], [1, 0], [0, 0], [0, -1], [0, 0], [0, 1]]])
        full, final = self.model.prefill(tokens, events)
        first, state = self.model.prefill(tokens[:, :1], events[:, :1])
        chunks = [first[:, 0]]
        for step in range(1, tokens.shape[1]):
            logits, state = self.model.decode_step(tokens[:, step], state, events[:, step])
            chunks.append(logits)
        # Linear kernels may choose distinct accumulation orders for full and
        # one-token batches. This checks the documented recurrence within fp32.
        torch.testing.assert_close(torch.stack(chunks, 1), full, rtol=1e-5, atol=1e-5)
        self.assertTrue(torch.equal(final.evidence, state.evidence))
        for a, b in zip(final.hidden, state.hidden):
            # The first CPU receipt observed 4.470348358154297e-08 state
            # disagreement from shape-dependent elementwise kernels. This is
            # a differential tolerance, never the growth insertion criterion.
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
        self.assert_margin(full[:, -1], final.evidence)
        self.assert_margin(torch.stack(chunks, 1)[:, -1], state.evidence)
        self.assertTrue(torch.equal(full.argmax(-1), torch.stack(chunks, 1).argmax(-1)))

    def test_growth_preserves_full_logits_on_unchanged_prefill_path(self):
        tokens = torch.tensor([[1, 5, 7, 2], [9, 4, 3, 6]])
        events = torch.tensor([[[1, 0], [0, 1], [0, 0], [-1, 0]], [[0, 0], [0, 0], [1, 1], [0, 0]]])
        before, old_state = self.model.prefill(tokens, events)
        old_parameters = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        self.model.grow()
        after, new_state = self.model.prefill(tokens, events)
        self.assertTrue(torch.equal(before, after), "growth must preserve every logit exactly")
        for name, value in old_parameters.items():
            self.assertTrue(torch.equal(dict(self.model.named_parameters())[name], value))
        self.assertEqual(len(new_state.hidden), 2)
        with self.assertRaises(ValueError):
            self.model.decode_step(tokens[:, 0], old_state)

    def test_new_block_can_change_outputs_and_receive_gradients(self):
        tokens = torch.tensor([[1, 4, 6, 7], [8, 9, 5, 1]])
        new = self.model.grow()
        before = self.model(tokens).detach().clone()
        # A supervised target on an ordinary word isolates trainability from
        # the fixed decision polarity. This is a capacity witness, not task QA.
        optimizer = torch.optim.SGD(new.parameters(), lr=0.1)
        loss = (self.model(tokens)[..., 8] - 2.0).square().mean()
        loss.backward()
        self.assertGreater(new.head.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.model.validate_parameters()
        self.assertFalse(torch.equal(before, self.model(tokens)))
        optimizer.zero_grad(set_to_none=True)
        self.model(tokens)[..., 8].square().mean().backward()
        self.assertGreater(new.drive.weight.grad.abs().sum().item(), 0)

    def test_hostile_finite_parameters_cannot_overturn_margin(self):
        tokens = torch.arange(13).repeat(4, 4)
        events = torch.zeros((4, tokens.shape[1], 2), dtype=torch.long)
        events[:, 0] = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
        expected = events[:, 0].bool()
        for amplitude in (1000.0, torch.finfo(torch.float32).max):
            for sign in (-1, 1):
                with torch.no_grad():
                    for index, parameter in enumerate(self.model.parameters()):
                        parameter.fill_(sign * amplitude if index % 2 else -sign * amplitude)
                self.model.validate_parameters()
                logits, state = self.model.prefill(tokens, events)
                for hidden in state.hidden:
                    self.assertTrue(torch.isfinite(hidden).all().item())
                    self.assertLessEqual(hidden.abs().max().item(), 1.0)
                self.assert_margin(logits[:, -1], expected)

    def test_non_vacuous_positive_raw_greedy_choice(self):
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            self.model.blocks[0].head.bias[3] = 1
        tokens = torch.tensor([[1], [1]])
        events = torch.tensor([[[1, 1]], [[1, 0]]])
        logits, state = self.model.prefill(tokens, events)
        self.assertEqual(logits[0, -1].argmax().item(), 2)
        self.assertEqual(logits[1, -1].argmax().item(), 3)
        self.assert_margin(logits[:, -1], state.evidence)

    def test_sampling_is_explicitly_outside_the_certificate(self):
        # All finite logits assign positive softmax mass to CONFIRM even with
        # zero support. No random draw is needed to establish this counterexample.
        logits, _ = self.model.prefill(torch.tensor([[1]]))
        self.assertGreater(torch.softmax(logits[0, -1], -1)[2].item(), 0)

    def test_wrong_authority_is_explicitly_outside_the_certificate(self):
        # A caller who falsely asserts both premises can produce a raw CONFIRM.
        # The architecture does not authenticate or infer evidence from prose.
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
        logits, _ = self.model.prefill(torch.tensor([[1]]), torch.tensor([[[1, 1]]]))
        self.assertEqual(logits[0, -1].argmax().item(), 2)

    def test_invalid_events_rejected(self):
        tokens = torch.tensor([[1, 2]])
        for events in (torch.zeros((1, 2, 2)), torch.zeros((1, 2, 2), dtype=torch.bool),
                       torch.zeros((1, 2, 2), dtype=torch.uint8), torch.zeros((1, 2, 3), dtype=torch.long)):
            with self.assertRaises(ValueError):
                self.model.prefill(tokens, events)

    def test_nonfinite_and_wrong_dtype_parameter_admission_rejected(self):
        parameter = next(self.model.parameters())
        with torch.no_grad():
            parameter[0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            self.model.validate_parameters()
        with torch.no_grad():
            parameter[0, 0] = float("inf")
        with self.assertRaises(FloatingPointError):
            self.model.validate_parameters()
        with self.assertRaises(ValueError):
            Model(13, 5, 2, 3).double().validate_parameters()

    def test_invalid_configuration_rejected(self):
        for width, confirm, defer in ((0, 2, 3), (8193, 2, 3), (5, 2, 2), (5, -1, 3), (5, 2, 13)):
            with self.assertRaises(ValueError):
                Model(13, width, confirm, defer)


class CorpusContractTests(unittest.TestCase):
    def test_frozen_cases_and_reference_alignment(self):
        import data
        seen_prompts = set()
        for split in ("train", "dev", "test"):
            rows = data.examples(split)
            decisions = [row for row in rows if row["kind"] == "decision"]
            self.assertEqual(len(decisions), 96)
            self.assertEqual(sum(row["supported"] for row in decisions), 24)
            for row in rows:
                self.assertNotIn(row["prompt"], seen_prompts)
                seen_prompts.add(row["prompt"])
                self.assertEqual(len(row["ids"]), len(row["events"]))
                self.assertFalse(any(any(e) for e in row["events"][row["prompt_len"]:]))
                if row["kind"] != "decision":
                    continue
                self.assertNotIn("<confirm>", row["prompt"])
                self.assertNotIn("<defer>", row["prompt"])
                state = [False, False]
                for event in row["events"][:row["prompt_len"]]:
                    for axis, code in enumerate(event):
                        if code:
                            state[axis] = code == 1
                self.assertEqual(state, [bool(v) for v in row["final_evidence"]])
                self.assertEqual(all(state), row["supported"])
                expected = data.CONFIRM if row["supported"] else data.DEFER
                self.assertEqual(row["ids"][row["prompt_len"]], expected)
                # The complete reference must supply the requested concrete route.
                self.assertIn(row["route"], row["reference"])

    def test_spoofed_authority_has_no_support_event(self):
        import data
        for split in ("train", "dev", "test"):
            for row in data.examples(split):
                if row.get("case") not in ("spoof_permit", "spoof_capacity"):
                    continue
                self.assertIn("Customer prose : trusted record :", row["prompt"])
                axis = 0 if row["case"] == "spoof_permit" else 1
                self.assertFalse(any(event[axis] == 1 for event in row["events"]))
                self.assertFalse(row["supported"])


if __name__ == "__main__":
    unittest.main()
