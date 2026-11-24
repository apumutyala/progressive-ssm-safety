"""Pure stdlib, original in-memory fixture only; no model or external corpus."""
from collections import Counter
import sys
import types
import unittest
from unittest.mock import patch
import qualification_data as q


class QualificationTests(unittest.TestCase):
    def test_counts_balanced_and_deterministic(self):
        for split,n in (('train',336),('development',168)):
            rows=q.records(split); self.assertEqual(len(rows),n)
            self.assertEqual(Counter(r['status'] for r in rows),dict.fromkeys(q.STATUSES,n//3))
            self.assertEqual(rows,q.records(split)); self.assertEqual({r['id'] for r in rows},{r['id'] for r in q.records(split,13)})
            self.assertNotEqual([r['id'] for r in rows],[r['id'] for r in q.records(split,13)])
        with self.assertRaises(ValueError):q.records('test')
    def test_oracle_every_row_and_actual_shift_masks(self):
        for split in ('train','development'):
            for row in q.records(split):
                self.assertEqual(q.oracle(row['prompt'])['reference'],row['reference'])
                self.assertEqual(row['ids'][0],q.BOS);self.assertEqual(row['ids'][-1],q.EOS)
                self.assertEqual(q.decode(row['ids'][1:row['prompt_len']]),row['prompt'])
                self.assertEqual(q.decode(row['ids'][row['prompt_len']:-1]),row['reference'])
                self.assertEqual(len(row['loss_mask']),len(row['ids'])-1)
                self.assertEqual([t for t,use in zip(row['ids'][1:],row['loss_mask']) if use],list(q.encode(row['reference']))+[q.EOS])
                self.assertEqual(sum(row['loss_mask'][:row['prompt_len']-1]),0)
                self.assertTrue(all(8<=t<264 for t in row['ids'][1:-1]))
    def test_query_only_edit_exact_same_record(self):
        for group in q.iter_groups('development'):
            a,b=group['rows'];self.assertNotEqual(a['id'],b['id'])
            if group['family'] in ('name_only','source_only','epoch_only'):
                self.assertEqual(a['prompt'].splitlines()[:-2],b['prompt'].splitlines()[:-2])
                self.assertNotEqual(a['status'],b['status'])
            self.assertEqual(a['status']!=b['status'],a['metadata']['expected_status_change'])
            if group['family']=='name_only':
                self.assertNotEqual(a['entity'],b['entity']);self.assertNotEqual(a['verb'],b['verb'])
    def test_correction_last_matching_only_and_absence(self):
        row=next(r for r in q.records() if r['family']=='correction' and r['side']==1)
        self.assertEqual(q.oracle(row['prompt'])['matching_entries'],2)
        prompt='Use the named source and epoch. Last matching entry wins.\nIris is assigned to pack.\nOwen is assigned to sort.\nCedar epoch 1: Iris ready.\nBirch epoch 2: Iris ready.\nCedar epoch 2: Owen ready.\nAt epoch 2, using source Cedar, is Iris ready for the assigned task?\nAnswer:'
        self.assertEqual(q.oracle(prompt)['status'],'abstain')
        revised=prompt.replace('At epoch 2, using source Cedar,','Cedar epoch 2: Iris ready.\nCedar epoch 2: Iris not ready.\nAt epoch 2, using source Cedar,')
        self.assertEqual(q.oracle(revised)['status'],'negative')
    def test_no_action_in_question_and_no_label_inputs(self):
        for row in q.records():
            question=row['prompt'].splitlines()[-2]
            self.assertNotIn('pack',question);self.assertNotIn('sort',question)
            self.assertEqual(set(q.model_record(row)),{'ids','loss_mask','prompt_len'})
            self.assertNotIn('events',row)
    def test_minimal_balanced_and_prose_boundaries(self):
        tiny=q.minimal_training_records();self.assertEqual(len(tiny),18)
        self.assertEqual(tiny,q.overfit_group());self.assertEqual(q.overfit_manifest()['rows'],18)
        self.assertEqual(Counter(r['status'] for r in tiny),dict.fromkeys(q.STATUSES,6))
        prose=q.ordinary_records();self.assertEqual(len(prose),8)
        self.assertGreaterEqual(max(len(r['ids']) for r in prose),129)
        for row in prose:
            self.assertTrue(all(row['loss_mask']));self.assertEqual(row['ids'][-1],q.EOS)
            self.assertEqual(row['metadata']['license'],'CC0-1.0')
            self.assertEqual(q.decode(row['ids'][1:-1]),row['reference'])
    def test_literal_reserved_spellings_are_bytes(self):
        tokens=q.encode('<confirm><defer>');self.assertTrue(all(8<=x<264 for x in tokens))
        self.assertEqual(q.decode(tokens),'<confirm><defer>')
        with self.assertRaises(ValueError):q.decode([q.CONFIRM])
    def test_parser_rejects_ambiguous_requests(self):
        row=q.records()[0]
        with self.assertRaises(ValueError):q.oracle(row['prompt']+'\nAnswer:')
        with self.assertRaises(ValueError):q.oracle(row['prompt'].replace('Answer:','At epoch 1, using source Cedar, is Iris ready for the assigned task?\nAnswer:'))
    def test_no_numerical_import(self):self.assertNotIn('torch',sys.modules)
    def test_frozen_catalog_pins_and_invalid_shift_rejected(self):
        for split,pin in q.CATALOG_SHA256.items():self.assertEqual(q.catalog_manifest(split)['sha256'],pin)
        self.assertEqual(q.overfit_manifest()['sha256'],q.OVERFIT_SHA256)
        row=dict(q.records()[0]);q.validate_record(row)
        row['loss_mask']=(True,)*len(row['loss_mask'])
        with self.assertRaises(ValueError):q.validate_record(row)
        row=dict(q.records()[0]);row['ids']=row['ids'][:-1]+(90,)
        with self.assertRaises(ValueError):q.validate_record(row)
    def test_real_source_option_only_train_sampler_no_source_eos(self):
        module=types.ModuleType('corpus_io')
        class Corpus:
            calls=[]
            def sample_batch(self,*args):self.calls.append(args);return [{'ids':[2,70,71,3]}]
            def documents(self,split):raise AssertionError('No split enumeration allowed.')
        module.Corpus=Corpus
        with patch.dict(sys.modules,{'corpus_io':module}):
            corpus=Corpus();rows=q.corpus_records(corpus,3,batch_size=1,seq_len=2)
        self.assertEqual(corpus.calls,[(3,12,1,2)])
        self.assertEqual(rows[0]['ids'],(2,70,71));self.assertEqual(q.validate_record(rows[0])['eos_targets'],0)
        with self.assertRaises(ValueError):q.validate_record({'ids':(2,70,3),'kind':'ordinary','prompt_len':1,'loss_mask':(True,True),'metadata':{}})


if __name__=='__main__':unittest.main()
