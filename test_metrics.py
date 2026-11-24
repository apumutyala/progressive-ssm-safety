"""Synthetic scoring checks. The model and evaluation data are not loaded."""
import unittest

from distance_metrics import correct_fields, first_error
from paired_metrics import Vocabulary, evaluate, majority_lookup


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocabulary(("end", "yes", "wait", "move", "bead", "to",
                                 "loft", "by", "sled", "pier"), 0, 1, 2)
        self.rows = []
        for side, place in enumerate(("loft", "pier")):
            answer = [1, 3, 4, 5, self.vocab.VOCAB.index(place), 7, 8, 0]
            self.rows.append(dict(id=str(side), side=side, pair="example",
                                  family="edit", query="fixture", supported=True,
                                  prompt=place, reference=" ".join(
                                      self.vocab.VOCAB[i] for i in answer),
                                  prompt_len=1, ids=[0, *answer], entity="bead",
                                  destination=place, vehicle="sled"))

    def test_correct_and_unchanged_pairs(self):
        predictions = {row["id"]:row["ids"][1:] for row in self.rows}
        exact = evaluate(self.rows, predictions, self.vocab)["by_family"]["edit"]
        self.assertEqual((exact["both_exact"], exact["both_helpful"], exact["response_changed"]),
                         (1, 1, 1))
        lookup = majority_lookup(self.rows)
        repeated = {row["id"]:lookup[("fixture", True)] for row in self.rows}
        failed = evaluate(self.rows, repeated, self.vocab)["by_family"]["edit"]
        self.assertEqual((failed["both_exact"], failed["both_helpful"], failed["response_changed"]),
                         (0, 0, 0))

    def test_distance_and_truncation(self):
        left, right = (r["ids"][1:] for r in self.rows)
        self.assertEqual(first_error(left, right), "4")
        self.assertFalse(correct_fields(left, right, [4]))
        self.assertEqual(first_error(left[:-1], left), str(len(left) - 1))
        self.assertTrue(correct_fields(left, right, [0, 2]))


if __name__ == "__main__":
    unittest.main()
