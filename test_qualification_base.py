import unittest
import torch
from qualification_base import prepare_base_rows, objective
from qualification_data import ordinary_records
from qualification_model import make_model


class OrdinaryLanguageObjective(unittest.TestCase):
    def test_identical_plan_multiple_models_and_padding(self):
        rows = ordinary_records()[:2]
        plan = prepare_base_rows(rows)
        self.assertEqual(plan,prepare_base_rows(rows))
        self.assertEqual(plan[1],sum(len(r['ids'])-1 for r in rows))
        for arm in ('certified_ssm','unconstrained_ssm','transformer'):
            model=make_model(arm,width=4,depth=1,max_context=256)
            loss=objective(model,plan,'cpu')
            terms=[objective(model,prepare_base_rows([r]),'cpu')*(len(r['ids'])-1) for r in rows]
            self.assertAlmostEqual(loss.item(),(sum(terms)/plan[1]).item(),places=5)

    def test_rejects_prompt_masked_or_synthetic_training_rows(self):
        row = ordinary_records()[0]
        with self.assertRaises(ValueError):
            prepare_base_rows([{**row,'kind':'synthetic'}])
        with self.assertRaises(ValueError):
            prepare_base_rows([{**row,'loss_mask':(False,)*len(row['loss_mask'])}])
