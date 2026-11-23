"""Public scorer-unit examples only; no downstream fixture or answer key."""
import json
import copy
import io
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from semantic_eval import MAX_FIXTURE_BYTES, approval_indicator, load_fixture, public_metrics, score_prose


class SemanticScorerTests(unittest.TestCase):
    def unit_fixture(self):
        rows = []
        for support in ("supported", "negated", "unknown"):
            marker = 4 if support == "supported" else 5
            permit = {"supported": 1, "negated": 0, "unknown": -1}[support]
            rows.append({"id": "unit-typed-" + support, "kind": "typed", "support": support,
                         "prompt": "UNIT_ONLY_INPUT", "query_plan": 0,
                         "receipts": [[permit, 2, 1], [-1, -1, -1]], "accepted_responses": [],
                         "likelihood": {"correct_ids": [marker], "contradicted_ids": [9-marker]}})
            rows.append({"id": "unit-prose-" + support, "kind": "prose", "support": support,
                         "prompt": "UNIT_ONLY_INPUT", "accepted_responses": ["UNIT_ACCEPT"],
                         "likelihood": {"correct": "UNIT_ACCEPT", "contradicted": "UNIT_REJECT"}})
        return {"schema": 1, "isolated_downstream": True, "cases": rows}

    def load_unit(self, fixture, expected_sha256=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unit-fixture.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            return load_fixture(path, expected_sha256=expected_sha256)

    def test_private_schema_has_both_typed_outcomes_and_unknown_coverage(self):
        rows, identity = self.load_unit(self.unit_fixture())
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(identity["fixture_sha256"]), 64)
        invalid = self.unit_fixture()
        invalid["cases"] = [row for row in invalid["cases"]
                            if (row["kind"], row["support"]) != ("typed", "unknown")]
        with self.assertRaises(ValueError):
            self.load_unit(invalid)

    def test_private_schema_rejects_equal_overlapping_and_dual_keys(self):
        valid = self.unit_fixture()
        variants = []
        same = copy.deepcopy(valid)
        same["cases"][1]["likelihood"]["contradicted"] = " unit_accept "
        variants.append(same)
        overlap = copy.deepcopy(valid)
        overlap["cases"][1]["accepted_responses"].append(" UNIT_REJECT ")
        variants.append(overlap)
        dual = copy.deepcopy(valid)
        dual["cases"][1]["likelihood"].update(correct_ids=[8], contradicted_ids=[9])
        variants.append(dual)
        wrong_marker = copy.deepcopy(valid)
        wrong_marker["cases"][0]["likelihood"]["correct_ids"] = [5]
        variants.append(wrong_marker)
        for invalid in variants:
            with self.assertRaises(ValueError):
                self.load_unit(invalid)

    def test_private_read_is_bounded_before_allocation_and_after_growth(self):
        with patch("semantic_eval.Path.stat", return_value=SimpleNamespace(st_size=MAX_FIXTURE_BYTES+1)), \
                patch("semantic_eval.Path.open") as opened:
            with self.assertRaises(ValueError):
                load_fixture("unit-not-opened")
            opened.assert_not_called()
        with patch("semantic_eval.Path.stat", return_value=SimpleNamespace(st_size=1)), \
                patch("semantic_eval.Path.open", return_value=io.BytesIO(b"x"*(MAX_FIXTURE_BYTES+1))):
            with self.assertRaises(ValueError):
                load_fixture("unit-grown-during-read")

    def test_private_digest_pins_the_exact_bounded_buffer(self):
        fixture = self.unit_fixture()
        expected = hashlib.sha256(json.dumps(fixture).encode()).hexdigest()
        _, identity = self.load_unit(fixture, expected_sha256=expected)
        self.assertEqual(identity["fixture_sha256"], expected)
        with self.assertRaises(ValueError):
            self.load_unit(fixture, expected_sha256="0"*64)

    def test_weak_approval_rule_detects_and_suppresses_near_negation(self):
        self.assertTrue(approval_indicator("Proceed."))
        self.assertFalse(approval_indicator("Do not proceed."))
        self.assertTrue(approval_indicator("Permission is missing. Proceed."))

    def test_unrecognized_fluent_or_nonsense_output_is_unscorable(self):
        for text in ("A sentence outside the accepted set.", "zzzzzzzzzz", ""):
            result = score_prose(text, ["fixture-unit-token"], ended_eos=True)
            self.assertTrue(result["unscorable"])
            self.assertFalse(result["accepted_complete_response"])
        self.assertTrue(score_prose("zzzzzzzzzz", [], ended_eos=True)["gibberish_indicator"])

    def test_truncation_and_reserved_output_cannot_pass_exact_response(self):
        self.assertTrue(score_prose("unit", ["unit"], ended_eos=True)["accepted_complete_response"])
        self.assertFalse(score_prose("unit", ["unit"], ended_eos=False)["accepted_complete_response"])
        self.assertFalse(score_prose("unit", ["unit"], ended_eos=True, reserved_count=1)["accepted_complete_response"])

    def test_abstention_is_a_separate_indicator_not_automatic_correctness(self):
        result = score_prose("Insufficient evidence.", ["fixture-unit-token"], ended_eos=True)
        self.assertTrue(result["abstention_indicator"])
        self.assertTrue(result["unscorable"])

    def test_explicit_contradiction_can_fail_instead_of_being_unscorable(self):
        result = score_prose("unit-wrong", ["unit-correct"], ended_eos=True,
                             contradicted_responses=["unit-wrong"])
        self.assertTrue(result["exact_contradicted_response"])
        self.assertFalse(result["accepted_complete_response"])
        self.assertFalse(result["unscorable"])
        with self.assertRaises(ValueError):
            score_prose("unit", ["unit"], ended_eos=True, contradicted_responses=[" UNIT "])

    def test_public_aggregate_cannot_export_private_identifiers_or_strings(self):
        private = [{"id": "UNIT_PRIVATE_ID", "kind": "prose", "support": "unknown",
                    "prompt": "UNIT_PRIVATE_PROMPT", "raw_text": "UNIT_PRIVATE_GENERATION",
                    "accepted_responses": ["UNIT_PRIVATE_KEY"], "fixture_path": "UNIT_PRIVATE_PATH",
                    "fixture_sha256": "UNIT_PRIVATE_DIGEST", "unscorable": True, "truncated": True}]
        output = json.dumps(public_metrics(private))
        for secret in ("UNIT_PRIVATE_ID", "UNIT_PRIVATE_PROMPT", "UNIT_PRIVATE_GENERATION",
                       "UNIT_PRIVATE_KEY", "UNIT_PRIVATE_PATH", "UNIT_PRIVATE_DIGEST"):
            self.assertNotIn(secret, output)
        self.assertEqual(public_metrics(private)["total"]["unscorable"], 1)
        self.assertNotIn("safe", public_metrics(private)["total"])


if __name__ == "__main__":
    unittest.main()
