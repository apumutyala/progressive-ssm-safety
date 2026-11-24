"""Stdlib tests use only self-authored in-memory bytes and fictional reply groups."""
from collections import Counter
import copy
import hashlib
import sys
import unittest

_BEFORE_RECIPE_IMPORT = set(sys.modules)
from evidence_recipe import EvidenceRecipe, BOS, EOS, BYTE_OFFSET
_RECIPE_IMPORTS = set(sys.modules)-_BEFORE_RECIPE_IMPORT


def encode(text): return [BYTE_OFFSET+b for b in text.encode()]
def sha(raw): return hashlib.sha256(raw).hexdigest()


class MemoryCorpus:
    def __init__(self, *, force_end=False):
        self.force_end = force_end; self.calls = []
        # These are made-up test chapters, not external corpus text.
        raw = (b'The imaginary blue leaf lies beside a wooden cup. '*12)+b'End of this test chapter.\n'
        self.docs = tuple({'id': 'made-up-'+str(i), 'split': 'train', 'raw': raw+bytes([65+i]),
            'sha256': sha(raw+bytes([65+i])), 'source_id': 'in-memory-test', 'license_id': 'self-authored-CC0'} for i in range(2))
    def documents(self, split):
        self.calls.append(('documents', split))
        if split != 'train': raise AssertionError('Holdout read forbidden in these tests.')
        return self.docs
    def sample_batch(self, step, seed, batch_size, seq_len):
        self.calls.append(('sample', step, seed, batch_size, seq_len))
        rows = []
        for i in range(batch_size):
            h = int(sha(f'{seed}:{step}:{i}'.encode()), 16)
            doc = self.docs[h % len(self.docs)]
            start = len(doc['raw'])-seq_len if self.force_end else h % (len(doc['raw'])-seq_len+1)
            end = start+seq_len; content = doc['raw'][start:end]
            rows.append({'doc_id': doc['id'], 'start': start, 'end': end, 'content_targets': len(content),
                'ids': [BOS]+[BYTE_OFFSET+b for b in content]+([EOS] if end == len(doc['raw']) else []),
                'synthetic_eos_targets': int(end == len(doc['raw']))})
        return rows
    def evaluation_windows(self, *args, **kwargs): raise AssertionError('No evaluation access permitted.')


def groups(*, split, seed, arm):
    if split != 'train': raise AssertionError('Synthetic holdout read forbidden.')
    for group in range(4):
        rows = []
        replies = ('Yes. Luma is ready to pack.', 'No. Luma is not ready to pack.',
                   'Unknown. The record does not establish that Luma is ready to pack.')
        for status in range(3):
            prompt = f'Fictional setup {group}. Readiness state {status}. Is Luma ready to pack? Answer: '
            for replica in range(3):
                reply = replies[status if arm == 'true' else (status+replica) % 3]
                p = [BOS]+encode(prompt)
                rows.append({'id': f'g{group}-s{status}-r{replica}', 'ids': p+encode(reply)+[EOS], 'prompt_len': len(p)})
        yield {'id': f'g{group}', 'rows': rows}


def attest(corpus):
    return [{'doc_id': d['id'], 'doc_sha256': d['sha256'], 'end': len(d['raw']), 'unit': 'chapter',
             'status': 'verified_complete_chapter', 'evidence_sha256': sha(b'fictional XML replay test evidence')}
            for d in corpus.docs]


class RecipeTests(unittest.TestCase):
    def test_only_stdlib_no_model_or_torch_import(self):
        self.assertNotIn('torch', _RECIPE_IMPORTS)
        self.assertNotIn('corpus_model', _RECIPE_IMPORTS)

    def test_actual_target_floor_and_global_denominator(self):
        recipe = EvidenceRecipe(MemoryCorpus(), groups)
        for step in range(12):
            for plan in recipe.paired_plan(step).values():
                self.assertGreaterEqual(plan.ordinary_targets/plan.total_targets, .75)
                self.assertGreaterEqual(plan.ordinary_byte_targets/plan.total_targets, .75)
                self.assertEqual(plan.loss_normalizer, sum(sum(r.loss_mask) for r in plan.records))
                self.assertEqual(plan.synthetic_answer_eos_targets, 9)
                self.assertEqual(plan.summary()['synthetic_prompt_targets'], 0)

    def test_control_identical_corpus_prompts_marginals(self):
        recipe = EvidenceRecipe(MemoryCorpus(), groups)
        pair = recipe.paired_plan(17, seed=4); a, b = pair['true'], pair['shuffled']
        ordinary = lambda plan: tuple(r for r in plan.records if r.kind == 'ordinary')
        self.assertEqual(ordinary(a), ordinary(b))
        self.assertEqual(a.ordinary_draw_sha256, b.ordinary_draw_sha256)
        self.assertEqual(a.synthetic_prompt_sha256, b.synthetic_prompt_sha256)
        self.assertEqual(a.synthetic_response_multiset_sha256, b.synthetic_response_multiset_sha256)
        self.assertNotEqual(a.plan_sha256, b.plan_sha256)
        replies = lambda plan: Counter(r.ids[r.prompt_len:] for r in plan.records if r.kind == 'synthetic')
        self.assertEqual(replies(a), replies(b))

    def test_synthetic_mask_targets_exact_complete_reply(self):
        plan = EvidenceRecipe(MemoryCorpus(), groups).plan('real', 0)
        for r in plan.records:
            if r.kind != 'synthetic': continue
            self.assertFalse(any(r.loss_mask[:r.prompt_len-1]))
            self.assertTrue(all(r.loss_mask[r.prompt_len-1:]))
            self.assertEqual(tuple(t for t, keep in zip(r.ids[1:], r.loss_mask) if keep), r.ids[r.prompt_len:])
            self.assertEqual(r.ids[-1], EOS)
            self.assertEqual(r.supervised_targets, len(r.ids)-r.prompt_len)

    def test_unverified_fragment_eos_removed_even_at_doc_end(self):
        corpus = MemoryCorpus(force_end=True)
        plan = EvidenceRecipe(corpus, groups).plan('true', 0)
        self.assertEqual(plan.ordinary_source_eos_targets, 0)
        self.assertTrue(all(EOS not in r.ids for r in plan.records if r.kind == 'ordinary'))

    def test_attested_chapter_dedicated_boundary_and_scope(self):
        corpus = MemoryCorpus()
        plan = EvidenceRecipe(corpus, groups, chapter_boundaries=attest(corpus)).plan('true', 0)
        boundaries = [r for r in plan.records if r.kind == 'ordinary' and r.ids[-1] == EOS]
        self.assertGreaterEqual(len(boundaries), 1)
        for row in boundaries:
            info = dict(row.lineage)
            self.assertEqual(info['source_eos_unit'], 'chapter')
            self.assertEqual(info['end'], len(next(d for d in corpus.docs if d['id'] == info['doc_id'])['raw']))
            self.assertIn('boundary_evidence_sha256', info)

    def test_no_punctuation_inference_and_no_book_claim(self):
        corpus = MemoryCorpus(force_end=True)
        annotation = attest(corpus)[0]
        for key, value in [('status', 'punctuation_looks_finished'), ('unit', 'book'), ('end', annotation['end']-1),
                           ('doc_sha256', '0'*64), ('evidence_sha256', 'not-a-pin'), ('doc_id', 'dev-hidden')]:
            with self.subTest(key=key):
                bad = dict(annotation); bad[key] = value
                with self.assertRaises(ValueError): EvidenceRecipe(corpus, groups, chapter_boundaries=[bad])

    def test_same_step_seed_is_exact_even_after_intervening_steps(self):
        corpus = MemoryCorpus(); recipe = EvidenceRecipe(corpus, groups, chapter_boundaries=attest(corpus))
        first = recipe.paired_plan(8, 12); recipe.paired_plan(9, 12)
        self.assertEqual(first, recipe.paired_plan(8, 12))
        other = EvidenceRecipe(MemoryCorpus(), groups, chapter_boundaries=attest(corpus))
        self.assertEqual(first, other.paired_plan(8, 12))

    def test_holdouts_never_requested(self):
        corpus = MemoryCorpus(); recipe = EvidenceRecipe(corpus, groups)
        recipe.paired_plan(0)
        self.assertTrue(all(call[1] == 'train' for call in corpus.calls if call[0] == 'documents'))

    def test_plain_container_conversion_cannot_mutate_frozen_plan(self):
        plan = EvidenceRecipe(MemoryCorpus(), groups).plan('true', 0)
        converted = plan.as_records(); converted[0]['ids'][0] = 99; converted[0]['loss_mask'][0] = False
        self.assertEqual(plan.records[0].ids[0], BOS)
        self.assertTrue(plan.records[0].loss_mask[0])

    def test_mismatched_prompt_or_control_mapping_rejected(self):
        for mode in ('prompt', 'identity', 'bad_null', 'incomplete', 'missing_eos'):
            def bad_groups(**kwargs):
                rows = copy.deepcopy(list(groups(**kwargs)))
                if kwargs['arm'] == 'shuffled':
                    r = rows[0]['rows'][0]
                    if mode == 'prompt': r['ids'][1] += 1
                    if mode == 'identity': r['id'] += '-other'
                    if mode == 'bad_null': rows = copy.deepcopy(list(groups(split='train', seed=kwargs['seed'], arm='true')))
                    if mode == 'incomplete': rows[0]['rows'].pop()
                    if mode == 'missing_eos': r['ids'].pop()
                yield from rows
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError): EvidenceRecipe(MemoryCorpus(), bad_groups).paired_plan(0)

    def test_corpus_byte_identity_failure_rejected(self):
        class BadCorpus(MemoryCorpus):
            def sample_batch(self, **kwargs):
                rows = super().sample_batch(**kwargs); rows[0]['ids'][2] += 1; return rows
        with self.assertRaises(ValueError): EvidenceRecipe(BadCorpus(), groups).paired_plan(0)

    def test_actual_authored_training_group_api_with_memory_corpus_only(self):
        import evidence_pairs
        corpus = MemoryCorpus()
        recipe = EvidenceRecipe(corpus, evidence_pairs.iter_groups, chapter_boundaries=attest(corpus))
        pair = recipe.paired_plan(0, 12)
        for plan in pair.values():
            self.assertEqual(plan.synthetic_targets, 378)
            self.assertEqual(plan.synthetic_answer_eos_targets, 9)
            self.assertGreaterEqual(plan.ordinary_source_eos_targets, 1)
            self.assertGreaterEqual(plan.ordinary_byte_targets/plan.total_targets, .75)
            self.assertEqual(len([r for r in plan.records if r.kind == 'synthetic']), 9)
        self.assertEqual(pair['true'].ordinary_draw_sha256, pair['shuffled'].ordinary_draw_sha256)


if __name__ == '__main__': unittest.main()
