import unittest
from collections import Counter
from itertools import product

from routed_cases import Event, ROUTES, make_cases
from routed_measure import score_predictions


class TestRoutedFit(unittest.TestCase):
    def test_balanced_schedule(self):
        from routed_fit_neural import training_plan
        cases=make_cases()
        plan=training_plan(cases)
        self.assertEqual(len(plan),8)
        used=[]
        for selector, prose in plan:
            self.assertEqual(len(selector),17)
            self.assertEqual({cases[i].expected_route for i in selector},set(ROUTES)|{None})
            self.assertEqual(Counter(cases[i].expected_kind for i in prose),Counter({"ALLOW":2,"NO":2,"UNKNOWN":2}))
            used.extend((cases[i].expected_route,cases[i].expected_kind) for i in prose)
        self.assertEqual(Counter(used),Counter(product(ROUTES,("ALLOW","NO","UNKNOWN"))))

    def test_byte_codec_has_no_record_collisions(self):
        from routed_fit_neural import OPCODES,compact_prefix
        seen=set()
        for route,revision,(kind,value) in product(ROUTES,(1,2,3),OPCODES):
            encoded=compact_prefix(b"XY", (Event(kind,route,revision,value),))
            self.assertEqual(encoded[:1],b"\x01")
            self.assertEqual(encoded[2:],b"XY")
            self.assertNotIn(encoded,seen);seen.add(encoded)
        self.assertEqual(len(seen),240)
        with self.assertRaises(ValueError):
            compact_prefix(b"X",(Event("grant",ROUTES[0],4),))

    def test_grammar_refusal_is_separate_from_learned_abstention(self):
        cases=make_cases()
        positives=next(c for c in cases if c.tag=="person_switch:ada").expected_route
        wrong=score_predictions("fixed",cases,[positives]*len(cases),lambda:None)
        abstain=score_predictions("fixed",cases,[None]*len(cases),lambda:None)
        self.assertEqual((wrong["metrics"]["grammar_only_unsupported"],wrong["metrics"]["learned_unsupported"]),(10,0))
        self.assertEqual((abstain["metrics"]["grammar_only_unsupported"],abstain["metrics"]["learned_unsupported"]),(0,10))
        self.assertGreater(wrong["by_kind"]["UNSUPPORTED"]["prewitness_false_positive"],0)
        self.assertEqual(wrong["metrics"]["wrong_accepted"],0)
        self.assertEqual(abstain["metrics"]["correct_positive"],0)

    def test_exact_diagnostic_route_satisfies_all_pairs(self):
        cases=make_cases()
        ideal=score_predictions("oracle only",cases,[c.expected_route for c in cases],lambda:None)
        self.assertTrue(all(p["both_exact"] for p in ideal["pairs"].values()))
        self.assertEqual(ideal["metrics"]["correct_positive"],20)
        self.assertEqual(ideal["metrics"]["presink_exact"],74)
