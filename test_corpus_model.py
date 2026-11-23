"""Small CPU tests for the byte LM, its execution audit and its matched arms."""

import unittest

import torch
import torch.nn.functional as F

from corpus_model import (
    BOS, BYTE_OFFSET, CONFIRM, DEFER, EOS, HOLD, QUERY_A, QUERY_B, VOCAB_SIZE,
    ByteTokenizer, make_model,
)


def bits(x):
    return x.detach().contiguous().view(torch.int32)


class CorpusModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(721)

    def test_bytes_utf8_and_protocol_are_separate(self):
        raw = bytes(range(256))
        self.assertEqual(ByteTokenizer.encode_bytes(raw), list(range(8, 264)))
        self.assertEqual(ByteTokenizer.decode_bytes(ByteTokenizer.encode_bytes(raw)), raw)
        literal = "Kayak 🌊 — café <confirm> <query_a>."
        ids = ByteTokenizer.encode(literal)
        self.assertTrue(all(BYTE_OFFSET <= t < VOCAB_SIZE for t in ids))
        self.assertEqual(ByteTokenizer.decode(ids), literal)
        self.assertEqual(ByteTokenizer.render([BOS] + ids + [EOS]), "<bos>" + literal + "<eos>")
        self.assertEqual(ByteTokenizer.render([BYTE_OFFSET + 255]), r"\xff")
        with self.assertRaises(ValueError):
            ByteTokenizer.decode_bytes([CONFIRM])
        with self.assertRaises(UnicodeDecodeError):
            ByteTokenizer.decode([BYTE_OFFSET + 255])

    def test_ordinary_next_byte_training_prefill_and_decode(self):
        ids = torch.tensor([[BOS] + ByteTokenizer.encode("A blue kite flew.") + [EOS]])
        x, y = ids[:, :-1], ids[:, 1:]
        for arm in ("certified", "unconstrained"):
            model = make_model(arm, width=8)
            logits, audit = model.audited(x)
            self.assertEqual(logits.shape, (1, x.size(1), 264))
            self.assertEqual(audit["state_elements"], x.numel() * 8)
            self.assertEqual(len(audit["hidden_min_by_time"]), x.size(1))
            self.assertEqual(audit["outside_box"], sum(audit["outside_box_by_time"]))
            self.assertTrue(torch.equal(bits(logits), bits(model(x))))
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.assertGreater(model.blocks[0].drive.weight.grad.abs().sum().item(), 0)
            optimizer.step()
            model.validate_parameters()
            full, final = model.prefill(x)
            self.assertFalse(torch.equal(bits(full), bits(logits)))
            self.assertEqual(final.query.item(), -1)
            self.assertTrue((final.records == -1).all())
            first, state = model.prefill(x[:, :1])
            per_step = [first[:, 0]]
            for i in range(1, x.size(1)):
                step, state = model.decode_step(x[:, i], state)
                per_step.append(step)
            torch.testing.assert_close(torch.stack(per_step, 1), full, rtol=1e-5, atol=1e-5)
            # Raw unconstrained-vocabulary autoregression also consumes its own output.
            next_id = full[:, -1].argmax(-1)
            generated, _ = model.decode_step(next_id, final)
            prefix_plus_token = torch.cat((x, next_id[:, None]), dim=1)
            torch.testing.assert_close(generated, model(prefix_plus_token)[:, -1], rtol=1e-5, atol=1e-5)

    def test_matched_parameter_schemas_and_actual_comparator_state_escape(self):
        torch.manual_seed(182)
        safe = make_model("certified", 8)
        torch.manual_seed(182)
        free = make_model("unconstrained", 8)
        self.assertEqual(sum(p.numel() for p in safe.parameters()), 8784)
        self.assertEqual(tuple(safe.state_dict()), tuple(free.state_dict()))
        for name, p in safe.state_dict().items():
            self.assertTrue(torch.equal(bits(p), bits(free.state_dict()[name])))
        with torch.no_grad():
            safe.blocks[0].drive.weight.fill_(2.0)
            safe.blocks[0].decay.weight.fill_(9.0)
        free.load_state_dict(safe.state_dict())
        x = torch.tensor([ByteTokenizer.encode("aaaaaaaaaaaa")])
        _, actual_safe = safe.audited(x)
        _, actual_free = free.audited(x)
        self.assertLessEqual(actual_safe["retention_max"], 255 / 256)
        self.assertLessEqual(actual_safe["max_abs_drive"], 1)
        self.assertEqual(actual_safe["outside_box"], 0)
        self.assertGreater(actual_free["retention_min"], 1)
        self.assertEqual(actual_free["drive_min"], 2)
        self.assertGreater(actual_free["outside_box"], 0)
        self.assertGreater(actual_free["max_abs_hidden"], 1)
        self.assertEqual(actual_free["hidden_clamp_changes"], 0)

    def test_shared_suffix_forgets_a_real_byte_prefix(self):
        model = make_model("certified", 4)
        a, b, suffix = ByteTokenizer.encode("abx")
        with torch.no_grad():
            block = model.blocks[0]
            block.decay.weight.zero_()
            block.drive.weight.zero_()
            block.drive.weight[a].fill_(1)
            block.drive.weight[b].fill_(-1)
            _, state = model.prefill(torch.tensor([[a], [b]]))
            gap = (state.hidden[0][0] - state.hidden[0][1]).abs().max()
            for k in range(1, 13):
                _, state = model.decode_step(torch.tensor([suffix, suffix]), state)
                actual = (state.hidden[0][0] - state.hidden[0][1]).abs().max()
                torch.testing.assert_close(actual, gap * (0.5 ** k), rtol=0, atol=0)

    def test_typed_first_query_replacement_and_nonvacuity(self):
        model = make_model("certified", 8)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.blocks[0].head.bias[DEFER] = 1
        literal = ByteTokenizer.encode("<query_b>")
        ids = torch.tensor([[BOS] + literal + [QUERY_A, QUERY_B, QUERY_A]])
        receipts = torch.full((1, ids.size(1), 2, 3), HOLD, dtype=torch.int64)
        receipts[0, 0, 0] = torch.tensor([1, 8, 8])
        receipts[0, -1, 0, 1] = 7
        logits, state = model.prefill(ids, receipts)
        chosen = logits.argmax(-1)[0]
        self.assertTrue((chosen[:-3] == DEFER).all())
        self.assertEqual(chosen[-3:].tolist(), [CONFIRM, CONFIRM, DEFER])
        self.assertEqual(state.query.item(), 0)
        self.assertEqual(state.records[0, 0].tolist(), [1, 7, 8])

    def test_one_adverse_gradient_update_can_flip_only_comparator_decision(self):
        free = make_model("unconstrained", 16)
        safe = make_model("certified", 16)
        with torch.no_grad():
            for p in free.parameters():
                p.zero_()
            free.blocks[0].mix.bias.fill_(1)
            free.blocks[0].head.bias[DEFER] = 0.25
        ids = torch.tensor([[QUERY_A]])  # six unknown fields
        before = free(ids)[:, -1]
        self.assertEqual(before.argmax().item(), DEFER)
        opt = torch.optim.SGD([free.blocks[0].head.weight], lr=4.0)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(before, torch.tensor([CONFIRM])).backward()
        opt.step()
        safe.load_state_dict(free.state_dict())
        self.assertEqual(free(ids).argmax(-1).item(), CONFIRM)
        safe_logits = safe(ids)[0, 0]
        self.assertEqual(safe_logits.argmax().item(), DEFER)
        self.assertGreaterEqual((safe_logits[DEFER] - safe_logits[CONFIRM]).item(), 0.49999)

    def test_extreme_finite_parameters_and_admission(self):
        model = make_model("certified", 4)
        with torch.no_grad():
            for i, p in enumerate(model.parameters()):
                p.fill_(torch.finfo(torch.float32).max * (1 if i % 2 else -1))
        model.validate_parameters()
        logits, audit = model.audited(torch.tensor([[BOS, BYTE_OFFSET, QUERY_A, EOS]]))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertGreater(audit["raw_dense_outside_effective_box"], 0)
        self.assertGreater(audit["raw_drive_outside_activation_domain"], 0)
        self.assertEqual(audit["outside_box"], 0)
        self.assertLessEqual(audit["effective_dense_max"], 1)
        with torch.no_grad():
            model.blocks[0].drive.weight[0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            model.validate_parameters()

    def test_all_logits_bit_exact_at_growth_and_new_transition_learns(self):
        x = torch.tensor([[BOS] + ByteTokenizer.encode("A cat."),
                          [QUERY_A] + ByteTokenizer.encode("A dog.")])
        target = torch.full_like(x, BYTE_OFFSET + ord("b"))
        for arm in ("certified", "unconstrained"):
            model = make_model(arm, 8)
            receipts = torch.full((2, x.size(1), 2, 3), HOLD)
            receipts[1, 0, 0] = torch.tensor([1, 4, 3])
            before, old_state = model.prefill(x, receipts)
            old_step, _ = model.decode_step(torch.tensor([EOS, EOS]), old_state)
            old_params = {n: bits(p).clone() for n, p in model.named_parameters()}
            new = model.grow()
            after, state = model.prefill(x, receipts)
            step, _ = model.decode_step(torch.tensor([EOS, EOS]), state)
            self.assertTrue(torch.equal(bits(before), bits(after)))
            self.assertTrue(torch.equal(bits(old_step), bits(step)))
            for name, old in old_params.items():
                self.assertTrue(torch.equal(old, bits(dict(model.named_parameters())[name])))
            with self.assertRaises(ValueError):
                model.decode_step(torch.tensor([EOS, EOS]), old_state)
            new_opt = torch.optim.SGD(new.parameters(), lr=0.2)
            start = new.drive.weight.detach().clone()
            for _ in range(2):
                new_opt.zero_grad(set_to_none=True)
                F.cross_entropy(model(x, receipts).flatten(0, 1), target.flatten()).backward()
                new_opt.step()
            self.assertFalse(torch.equal(bits(start), bits(new.drive.weight)))
            self.assertFalse(torch.equal(bits(after), bits(model(x, receipts))))


if __name__ == "__main__":
    unittest.main()
