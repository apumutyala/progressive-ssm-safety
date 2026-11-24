"""Compact deterministic CPU-only contract fixtures; no training claim.

The coordinator owns numerical execution. Invoke with unittest in its admitted
CPU test slot; importing this file does not construct a model or touch CUDA.
"""
import copy
import unittest

import torch

import qualification_eval as q


YES = "Yes. Mira is ready to pack."
NO = "No. Mira is not ready to pack."
UNKNOWN = "Unknown. The record does not establish that Mira is ready to pack."


def case(identifier, reference, **fields):
    return {"id": identifier, "prompt": identifier + "|", "reference": reference, **fields}


def panel():
    result = []
    for pair, family, references in (
            ("relevant", "relevant_edit", (YES, NO)),
            ("nuisance", "nuisance_edit", (UNKNOWN, UNKNOWN)),
            ("query", "name_only", (YES, "No. Noel is not ready to sort."))):
        for side, reference in enumerate(references):
            result.append(case(pair + str(side), reference, pair=pair, family=family,
                               side=side, contrast=NO if reference != NO else YES))
    return result


class Scripted(torch.nn.Module):
    """Exact next-byte fixture depending only on prompt IDs and prior bytes."""
    def __init__(self, outputs, *, repeat_id=None):
        super().__init__()
        self.outputs, self.repeat_id, self.calls = outputs, repeat_id, []

    def forward(self, ids):
        assert ids.device.type == "cpu"
        self.calls.append(ids[0].tolist())
        sequence = ids[0].tolist()
        split = sequence.index(ord("|") + 8)
        prompt = bytes(i - 8 for i in sequence[1:split + 1]).decode()
        answer = q.encode(self.outputs[prompt]) + [q.EOS]
        step = len(sequence) - split - 1
        token = self.repeat_id if self.repeat_id is not None else answer[min(step, len(answer) - 1)]
        logits = torch.zeros((1, ids.shape[1], 264), device=ids.device)
        logits[0, -1, token] = 6
        return logits

    def prefill(self, ids):
        return self(ids), ids.clone()

    def decode_step(self, token, state):
        ids = torch.cat((state, token[:, None]), dim=1)
        return self(ids)[:, -1], ids


class QualificationTests(unittest.TestCase):
    def test_positive_negative_unknown_and_pairs(self):
        cases = panel()
        model = Scripted({row["prompt"]: row["reference"] for row in cases})
        report = q.evaluate(model, cases)
        self.assertTrue(model.training)
        self.assertEqual(report["aggregate"]["canonical_with_eos_accuracy"], 1)
        self.assertEqual(report["aggregate"]["changed_pair_both_correct_rate"], 1)
        self.assertEqual(report["aggregate"]["invariant_pair_raw_stability_rate"], 1)
        self.assertEqual(report["aggregate"]["by_family"]["name_only"]["parsed_switch_rate"], 1)
        self.assertTrue(all(row["prefix_diagnostic"]["available"] for row in report["rows"]))
        self.assertTrue(q.qualification_gate(report, {"min_status_content_accuracy": 1,
                                                     "changed_pair_both_correct_rate": 1,
                                                     "invariant_pair_both_correct_rate": 1})["passed"])

    def test_always_abstain_is_not_useful_positive_coverage(self):
        cases = [case(str(i), ref) for i, ref in enumerate((YES, NO, UNKNOWN))]
        report = q.evaluate(Scripted({row["prompt"]: UNKNOWN for row in cases}), cases)
        self.assertEqual(report["aggregate"]["content_accuracy"], 1 / 3)
        self.assertEqual(report["aggregate"]["by_status"]["affirmative"]["content_accuracy"], 0)
        self.assertEqual(report["aggregate"]["by_status"]["negative"]["content_accuracy"], 0)
        self.assertFalse(q.qualification_gate(report, {"min_status_content_accuracy": .5})["passed"])

    def test_cached_and_forward_full_vocab_raw_parity(self):
        cases = panel()[:2]
        model = Scripted({row["prompt"]: row["reference"] for row in cases})
        direct = q.evaluate(model, cases)
        cached = q.evaluate(model, cases, cached=True)
        self.assertEqual(direct["rows"], cached["rows"])
        self.assertEqual(direct["aggregate"], cached["aggregate"])
        self.assertEqual(cached["generation_path"], "prefill_decode_step")
        with self.assertRaises(ValueError):
            q.evaluate(model, cases, cached="yes")

    def test_full_vocab_reserved_is_not_masked(self):
        row = case("reserved", YES)
        report = q.evaluate(Scripted({row["prompt"]: YES}, repeat_id=4), [row], max_new_tokens=3)
        result = report["rows"][0]
        self.assertEqual(result["raw_ids"], [4, 4, 4])
        self.assertTrue(result["invalid_reserved"])
        self.assertTrue(result["capped"])
        self.assertFalse(result["content_correct"])

    def test_content_and_eos_and_wrong_binding_distinct(self):
        row = case("a", YES)
        score = q.score_raw(row, q.encode(YES), capped=True)
        self.assertTrue(score["content_correct"])
        self.assertFalse(score["correct_with_eos"])
        self.assertFalse(score["canonical_with_eos"])
        wrong = q.score_raw(row, q.encode("Yes. Noel is ready to sort.") + [3])
        self.assertTrue(wrong["wrong_entity"])
        self.assertTrue(wrong["wrong_action"])
        self.assertFalse(wrong["content_correct"])
        self.assertTrue(q.score_raw(row, q.encode("Mira is prepared to pack.") + [3])["content_correct"])
        self.assertTrue(q.score_raw(row, q.encode("Yes. Mira is not ready to pack.") + [3])["other"])

    def test_caps_eos_invalid_utf8_and_structure(self):
        row = case("a", YES)
        with self.assertRaises(ValueError):
            q.score_raw(row, [3], capped=True)
        self.assertTrue(q.score_raw(row, [3, 8])["invalid_reserved"])
        self.assertFalse(q.score_raw(row, [8 + 255, 3])["valid_utf8"])
        report = q.evaluate(Scripted({row["prompt"]: YES}), [row], max_new_tokens=len(q.encode(YES)))
        self.assertTrue(report["rows"][0]["content_correct"])
        self.assertTrue(report["rows"][0]["capped"])

    def test_teacher_prefix_margin_is_not_free_success(self):
        row = case("a", YES, contrast="Yes. Mira is ready to sort.")
        class MisleadingPrefix(Scripted):
            def forward(self, ids):
                out = super().forward(ids)
                sequence = ids[0].tolist()
                split = sequence.index(ord("|") + 8)
                tail = sequence[split + 1:]
                positive = q.encode(YES) + [3]
                if tail and tail == positive[:len(tail)]:
                    out.zero_()
                    out[0, -1, positive[len(tail)]] = 6
                return out
        report = q.evaluate(MisleadingPrefix({row["prompt"]: NO}), [row])
        result = report["rows"][0]
        self.assertFalse(result["content_correct"])
        self.assertGreater(result["prefix_diagnostic"]["positive_minus_foil"], 0)
        self.assertFalse(result["prefix_diagnostic"]["free_prefix_matches_common"])

    def test_same_reference_without_foil_is_unavailable_and_eos_divergence(self):
        cases = [case(str(i), YES, pair="p", side=i, family="nuisance_edit") for i in range(2)]
        report = q.evaluate(Scripted({row["prompt"]: YES for row in cases}), cases)
        self.assertFalse(report["rows"][0]["prefix_diagnostic"]["available"])
        row = case("prefix", YES, contrast=YES + " Extra.")
        report = q.evaluate(Scripted({row["prompt"]: YES}), [row])
        self.assertEqual(report["rows"][0]["prefix_diagnostic"]["positive_id"], 3)

    def test_no_dataset_answer_ids_enter_free_generation_and_check_called(self):
        row = case("a", YES, ids=(999999,), loss_mask=(True,), prompt_len=99999)
        model = Scripted({row["prompt"]: YES})
        calls = []
        q.evaluate(model, [row], check=lambda: calls.append(1))
        self.assertEqual(model.calls[0], [2] + q.encode(row["prompt"]))
        self.assertEqual(len(calls), len(model.calls))

    def test_validation_nonfinite_logits_and_mode_restoration(self):
        row = case("a", YES)
        class Broken(Scripted):
            def forward(self, ids):
                result = super().forward(ids)
                result[0, -1, 0] = float("nan")
                return result
        model = Broken({row["prompt"]: YES})
        model.eval()
        with self.assertRaises(ValueError):
            q.evaluate(model, [row])
        self.assertFalse(model.training)
        with self.assertRaises(ValueError):
            q.evaluate(model, [row, row])
        with self.assertRaises(ValueError):
            q.score_raw({**row, "status": "negative"}, [3])

    def test_summary_denominators_ids_and_missing_thresholds(self):
        rows = panel()
        scores = [q.score_raw(row, q.encode(row["reference"]) + [3]) for row in rows]
        aggregate, pairs = q.summarize(rows, scores)
        self.assertEqual(sum(item["cases"] for item in aggregate["by_status"].values()), len(rows))
        self.assertEqual(len(pairs), 3)
        with self.assertRaises(ValueError):
            q.summarize(rows, scores[:-1])
        bad = copy.deepcopy(scores)
        bad[0]["id"] = bad[1]["id"]
        with self.assertRaises(ValueError):
            q.summarize(rows, bad)
        single = q.score_raw(case("one", YES), q.encode(YES) + [3])
        agg, _ = q.summarize([case("one", YES)], [single])
        self.assertFalse(q.qualification_gate({"aggregate": agg}, {"min_status_content_accuracy": 0})["passed"])
        with self.assertRaises(ValueError):
            q.qualification_gate({"aggregate": agg}, {"magic": 1})

    def test_frozen_public_data_schema_and_pair_coverage(self):
        import qualification_data
        cases = qualification_data.records("development")
        scores = [q.score_raw(row, q.encode(row["reference"]) + [3]) for row in cases]
        aggregate, pairs = q.summarize(cases, scores)
        self.assertEqual(aggregate["cases"], 168)
        self.assertEqual(len(pairs), 84)
        self.assertEqual(set(aggregate["by_family"]), set(qualification_data.FAMILIES))
        self.assertEqual(aggregate["content_accuracy"], 1)
        self.assertTrue(all(row["contrast"] != row["reference"] for row in cases))


if __name__ == "__main__":
    unittest.main()
