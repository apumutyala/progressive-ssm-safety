"""Pure-stdlib collation/accounting checks; lazy Torch adapter NEVER invoked."""
import ast
from dataclasses import replace
from fractions import Fraction
import hashlib
from pathlib import Path
import sys
import unittest

_before = set(sys.modules)
import response_step as s
from evidence_recipe import EvidenceRecipe
_imports = set(sys.modules)-_before


def encode(text):
    return [s.BYTE_OFFSET+b for b in text.encode()]


class MemoryCorpus:
    def __init__(self):
        raw = b'This chapter is an isolated ordinary prose fixture. '*12
        self.doc = {'id': 'fixture-chapter', 'split': 'train', 'raw': raw,
                    'sha256': hashlib.sha256(raw).hexdigest()}
    def documents(self, split):
        if split != 'train':
            raise AssertionError('No heldout access.')
        return (self.doc,)
    def sample_batch(self, step, seed, batch_size, seq_len):
        result = []
        for i in range(batch_size):
            start = (step+i) % (len(self.doc['raw'])-seq_len)
            raw = self.doc['raw'][start:start+seq_len]
            result.append({'ids': [s.BOS]+[s.BYTE_OFFSET+b for b in raw],
                           'doc_id': self.doc['id'], 'start': start, 'end': start+seq_len,
                           'content_targets': len(raw)})
        return result


def groups(*, split, seed, arm):
    if split != 'train':
        raise AssertionError('No synthetic heldout access.')
    replies = ('Yes, do it.', 'No, stop this.', 'The record does not say.')
    rows = []
    for evidence in range(3):
        prompt = f'Fictional record {evidence}. Reply: '
        pids = [s.BOS]+encode(prompt)
        for replica in range(3):
            label = evidence if arm == 'true' else (evidence+replica) % 3
            rows.append({'id': f's{evidence}-r{replica}', 'ids': pids+encode(replies[label])+[s.EOS],
                         'prompt_len': len(pids)})
    yield {'id': 'isolated-nine', 'rows': rows}


class ResponseStepTests(unittest.TestCase):
    def setUp(self):
        self.plans = EvidenceRecipe(MemoryCorpus(), groups, seq_len=16).paired_plan(3, seed=7)

    def test_import_and_helpers_never_import_torch(self):
        # An admitted discovery process may have imported Torch before this
        # module. Only imports introduced by our pure helpers are prohibited.
        self.assertFalse(any(name == 'torch' or name.startswith('torch.') for name in _imports))
        self.assertNotIn('corpus_model', _imports)

    def test_recomputed_counts_and_exact_prediction_alignment(self):
        plan = self.plans['true']
        accounting = s.validate_plan(plan)
        self.assertEqual(accounting.total_targets, plan.total_targets)
        self.assertEqual(accounting.synthetic_answer_eos_targets, 9)
        self.assertEqual(accounting.ordinary_source_eos_targets, 0)
        synthetic = next(r for r in plan.records if r.kind == 'synthetic')
        collated = s.collate_records([synthetic])
        p = synthetic.prompt_len
        self.assertTrue(all(t == s.IGNORE_INDEX for t in collated.targets[0][:p-1]))
        self.assertEqual(collated.targets[0][p-1:], synthetic.ids[p:])
        self.assertEqual(collated.input_ids[0], synthetic.ids[:-1])

    def test_padding_has_no_targets_or_prompt_loss(self):
        rows = self.plans['true'].records
        selected = [min(rows, key=lambda r: len(r.ids)), max(rows, key=lambda r: len(r.ids))]
        batch = s.collate_records(selected)
        self.assertGreater(batch.padding_positions, 0)
        self.assertEqual(batch.supervised_targets, sum(r.supervised_targets for r in selected))
        for i, record in enumerate(selected):
            length = len(record.ids)-1
            self.assertTrue(all(t == s.IGNORE_INDEX for t in batch.targets[i][length:]))
            self.assertTrue(all(not keep for keep in batch.loss_mask[i][length:]))
            self.assertEqual(batch.input_lengths[i], length)

    def test_all_partitions_use_one_global_normalizer(self):
        plan = self.plans['true']
        for size in (1, 2, 5, 73):
            for grouping in (False, True):
                prepared = s.prepare_step(plan, microbatch_records=size, group_by_length=grouping)
                self.assertEqual(prepared.global_normalizer, plan.total_targets)
                self.assertEqual(sum(b.supervised_targets for b in prepared.microbatches), plan.total_targets)
                self.assertEqual(sorted(i for b in prepared.microbatches for i in b.record_ids), sorted(r.id for r in plan.records))
        # Exact symbolic gradient contributions for unequal microbatch sizes:
        # sum_i sum_j(g_ij)/N differs from mean_i(mean_j(g_ij)).
        gradients = ((Fraction(2),), (Fraction(3), Fraction(5), Fraction(7)))
        global_gradient = sum(sum(g) for g in gradients)/4
        accumulated = sum(sum(g)/4 for g in gradients)
        wrong_local_means = sum(sum(g)/len(g) for g in gradients)/2
        self.assertEqual(global_gradient, accumulated)
        self.assertNotEqual(global_gradient, wrong_local_means)
        self.assertEqual(s.normalized_sum([2., 15.], 4), float(global_gradient))

    def test_paired_arm_accounting_and_masks_remain_matched(self):
        a, b = (self.plans[arm] for arm in ('true', 'shuffled'))
        self.assertEqual(s.validate_plan(a), s.validate_plan(b))
        for size in (1, 3, 8):
            aa, bb = (s.prepare_step(p, microbatch_records=size) for p in (a, b))
            self.assertEqual(aa.global_normalizer, bb.global_normalizer)
            self.assertEqual([x.supervised_targets for x in aa.microbatches], [x.supervised_targets for x in bb.microbatches])

    def test_bad_masks_totals_truncation_and_protocol_rejected(self):
        plan = self.plans['true']
        with self.assertRaises(ValueError):
            s.validate_plan(replace(plan, ordinary_byte_targets=plan.ordinary_byte_targets+1))
        synthetic = next(r for r in plan.records if r.kind == 'synthetic')
        masks = list(synthetic.loss_mask); masks[0] = True
        with self.assertRaises(ValueError): s.collate_records([replace(synthetic, loss_mask=tuple(masks))])
        with self.assertRaises(ValueError): s.collate_records([replace(synthetic, ids=synthetic.ids[:-1], loss_mask=synthetic.loss_mask[:-1])])
        bad = list(synthetic.ids); bad[1] = 6
        with self.assertRaises(ValueError): s.collate_records([replace(synthetic, ids=tuple(bad))])

    def test_unattested_source_eos_and_duplicate_rows_rejected(self):
        plan = self.plans['true']
        ordinary = next(r for r in plan.records if r.kind == 'ordinary')
        with self.assertRaises(ValueError):
            s.collate_records([replace(ordinary, ids=ordinary.ids+(s.EOS,), loss_mask=ordinary.loss_mask+(True,))])
        with self.assertRaises(ValueError):
            s.validate_plan(replace(plan, records=plan.records+(plan.records[0],)))
        with self.assertRaises(ValueError):
            s.collate_records(iter(plan.records))

    def test_attested_source_eos_is_separate_from_nine_answer_eos(self):
        corpus = MemoryCorpus()
        annotation = {'doc_id': corpus.doc['id'], 'doc_sha256': corpus.doc['sha256'],
                      'end': len(corpus.doc['raw']), 'unit': 'chapter',
                      'status': 'verified_complete_chapter', 'evidence_sha256': 'a'*64}
        plan = EvidenceRecipe(corpus, groups, seq_len=16, chapter_boundaries=[annotation]).plan('true', 3)
        measured = s.validate_plan(plan)
        self.assertEqual(measured.ordinary_source_eos_targets, 1)
        self.assertEqual(measured.synthetic_answer_eos_targets, 9)
        batch = s.collate_records(plan.records)
        self.assertEqual(sum(t == s.EOS for row in batch.targets for t in row), 10)
        self.assertEqual(measured.total_targets, batch.supervised_targets)

    def test_future_adapter_is_lazy_and_cannot_step_or_clip(self):
        tree = ast.parse(Path(s.__file__).read_text())
        adapter = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'accumulate_gradients')
        self.assertNotIn('optimizer', [a.arg for a in adapter.args.args+adapter.args.kwonlyargs])
        forbidden = {'step', 'zero_grad', 'clip_grad_norm_', 'clip_grad_value_', 'save', 'write_text', 'write_bytes'}
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden for n in ast.walk(adapter)))
        module_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any((isinstance(n, ast.Import) and any(a.name.startswith('torch') for a in n.names)) or
                             (isinstance(n, ast.ImportFrom) and (n.module or '').startswith('torch')) for n in module_imports))
        # This is a static scope test only: no gradient adapter invocation.


if __name__ == '__main__':
    unittest.main()
