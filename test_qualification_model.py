"""Small CPU checks for execution by the numerical owner; no fit or corpus I/O."""
import unittest
import torch

import qualification_model as qm
from corpus_model import BOS, QUERY_A, QUERY_B, HOLD, DEFER, CONFIRM


class ModelContract(unittest.TestCase):
    def make(self, arm):
        torch.manual_seed(91)
        return qm.make_model(arm, width=16, depth=2, max_context=64)

    def test_counts_and_planning(self):
        for arm in qm.ARMS:
            m = self.make(arm)
            self.assertEqual(sum(p.numel() for p in m.parameters()), qm.parameter_count(arm, 16, 2))
            w = qm.matched_width(arm, 20_000_000, 4)
            estimate = qm.resource_estimate(arm, w, 4, 2, 64)
            self.assertLess(abs(estimate["parameters"]-20_000_000), 50_000)
            self.assertEqual(estimate["fp32_kv_cache_bytes"], 8*2*64*w*4)
        a, b = self.make("certified_ssm"), self.make("unconstrained_ssm")
        self.assertEqual(list(a.state_dict()), list(b.state_dict()))
        self.assertTrue(all(torch.equal(t, b.state_dict()[k]) for k, t in a.state_dict().items()))

    def test_causal_and_cached_logits(self):
        for arm in qm.ARMS:
            m = self.make(arm)
            ids = torch.tensor([[BOS, 90, 70, 91, 20], [BOS, 40, 20, 18, 10]])
            with torch.no_grad():
                full = m(ids)
                altered = ids.clone()
                altered[:, -1] = 65
                torch.testing.assert_close(full[:, :-1], m(altered)[:, :-1], rtol=1e-5, atol=1e-5)
                pref, cache = m.prefill(ids[:, :2])
                parts = [pref]
                for token in ids[:, 2:].unbind(1):
                    logits, cache = m.decode_step(token, cache)
                    parts.append(logits[:, None])
                torch.testing.assert_close(torch.cat(parts, dim=1), full, rtol=1e-4, atol=1e-5)
                if arm == "certified_ssm":
                    self.assertTrue(all((h.abs() <= 1).all() for h in cache.hidden))

    def test_positive_grant_revoke_and_plain_text_forgery(self):
        m = self.make("certified_ssm")
        ids = [QUERY_A]+[8+b for b in b"grant plan a"]+[QUERY_B]
        x = torch.tensor([ids, ids])
        events = torch.full((2, len(ids), 2, 5), HOLD, dtype=torch.long)
        events[0, 0, 0] = torch.tensor([17, 5, 1, 3, 2])
        events[0, -1, 0] = torch.tensor([17, 8, 0, -1, -1])
        with torch.no_grad():
            m.branches[0].head.weight.zero_()
            m.branches[0].head.bias.zero_()
            z, state = m.prefill(x, events)
            self.assertTrue((z[0, :-1].argmax(-1) == CONFIRM).all())
            self.assertTrue((z[1, :, CONFIRM] < z[1, :, DEFER]).all())
            self.assertLess(z[0, -1, CONFIRM], z[0, -1, DEFER])
            self.assertTrue(torch.equal(z[0, :, 8:], z[1, :, 8:]))
            self.assertEqual(state.query.tolist(), [0, 0])
            self.assertTrue((state.records[1] == -1).all())

    def test_atomic_source_epoch_gate_and_revocation_tombstone(self):
        m = self.make("certified_ssm")
        x = torch.tensor([[QUERY_A]+[65]*6])
        events = torch.full((1, 7, 2, 5), HOLD, dtype=torch.long)
        events[0, :, 0] = torch.tensor([[29, 99, 1, 8, 2], [17, 5, 1, 8, 2], [17, 5, 0, -1, -1],
                                         [17, 9, 1, 8, -1], [17, 7, 0, -1, -1], [17, 6, 1, 8, 2],
                                         [17, 8, 1, 8, 2]])
        with torch.no_grad():
            z, final = m.prefill(x, events)
            self.assertEqual((z[0, :, CONFIRM] > z[0, :, DEFER]).tolist(),
                             [False, True, True, True, False, False, True])
            _, state = m.prefill(x[:, :5], events[:, :5])
            self.assertEqual(state.sources.tolist(), [[17, -1]])
            self.assertEqual(state.epochs.tolist(), [[7, -1]])
            self.assertEqual(state.records[0, 0].tolist(), [0, -1, -1])
            for t in (5, 6):
                out, state = m.decode_step(x[:, t], state, events[:, t])
                torch.testing.assert_close(out, z[:, t], atol=1e-5, rtol=1e-4)
            self.assertTrue(torch.equal(state.epochs, final.epochs))
            with self.assertRaisesRegex(ValueError, "source/epoch"):
                m(x, events[..., 2:])

    def test_matched_comparator_can_reverse_order(self):
        x = torch.tensor([[QUERY_A]])
        for arm in ("unconstrained_ssm", "certified_ssm"):
            m = self.make(arm)
            with torch.no_grad():
                for p in m.parameters():
                    p.zero_()
                branch = m.branches[0]
                feature = torch.tensor([1.]*8+[-1.]*8)
                branch.embedding.weight[QUERY_A].copy_(feature)
                branch.head.weight[CONFIRM].copy_(feature)
                z = m(x)[0, 0]
                self.assertEqual(z.argmax().item() == CONFIRM, arm == "unconstrained_ssm")

    def test_retrieval_and_recurrence_affect_full_vocab_loss(self):
        m = self.make("certified_ssm")
        x = torch.tensor([[BOS]+[8+b for b in b"Ab. B?"]])
        z = m(x)
        self.assertFalse(torch.equal(z[:, -1, 8:], m.forward_without_recurrence(x)[:, -1, 8:]))
        self.assertFalse(torch.equal(z[:, -1, 8:], m.forward_without_retrieval(x)[:, -1, 8:]))
        torch.nn.functional.cross_entropy(z[:, :-1].reshape(-1, 264), x[:, 1:].flatten()).backward()
        for layer in m.branches[0].layers:
            self.assertGreater(layer.transition.weight.grad.abs().sum().item(), 0)
            for part in layer.qkv.weight.grad.chunk(3, dim=0):
                self.assertGreater(part.abs().sum().item(), 0)

    def test_silent_growth_preserves_bits_and_admits_learning(self):
        m = self.make("certified_ssm")
        x = torch.tensor([[QUERY_A, 90, 63]])
        with torch.no_grad():
            previous, state = m.prefill(x)
        branch = m.grow()
        new = m(x)
        self.assertTrue(torch.equal(previous.view(torch.int32), new.detach().view(torch.int32)))
        with self.assertRaisesRegex(ValueError, "predates"):
            m.decode_step(x[:, -1], state)
        new[:, -1, 65].sum().backward()
        self.assertGreater(branch.head.weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            branch.head.bias[65].add_(.5)
            self.assertFalse(torch.equal(m(x), previous))
            self.assertTrue((m(x)[..., CONFIRM] < m(x)[..., DEFER]).all())

    def test_growth_preserves_live_adam_state_and_next_step_trains_new_width(self):
        m = self.make("certified_ssm")
        opt = torch.optim.Adam(m.parameters(), lr=.003)
        x = torch.tensor([[QUERY_A, 90, 63]])
        m(x)[:, -1, 65].sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        before = m(x).detach()
        old = {p: {key: (value, value.clone()) for key, value in state.items()}
               for p, state in opt.state.items()}
        branch = m.grow_width_with_adam(opt)
        self.assertTrue(torch.equal(before.view(torch.int32), m(x).detach().view(torch.int32)))
        self.assertEqual(set(opt.state), set(old))
        for p, state in old.items():
            for key, (original, clone) in state.items():
                self.assertIs(original, opt.state[p][key])
                self.assertTrue(torch.equal(original, clone))
        m(x)[:, -1, 65].sum().backward()
        opt.step()
        self.assertGreater(branch.head.weight.abs().sum().item(), 0)
        self.assertTrue((m(x)[..., CONFIRM] < m(x)[..., DEFER]).all())

    def test_invalid_numeric_update_and_context_fail(self):
        m = self.make("certified_ssm")
        with self.assertRaisesRegex(ValueError, "context"):
            m(torch.zeros((1, 65), dtype=torch.long))
        with torch.no_grad():
            next(m.parameters()).fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "Finite FP32"):
            m.validate_parameters()
