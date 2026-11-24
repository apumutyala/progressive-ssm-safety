"""CPU fixture tests for PARENT execution only; implementation author ran no training.

Pure tests need only stdlib. Numerical tests import Torch inside their methods,
construct only a tiny ephemeral byte model and never read study dev/final data.
"""
from collections import Counter
import copy
import unittest
import qualification_overfit as q


def fixture_rows():
    rows = []
    for status, answer in zip(q.STATUSES, ('A.', 'B.', 'C.')):
        for i in range(6):
            prompt = f'{status[0]}{i}? '
            p = (2,)+tuple(8+b for b in prompt.encode())
            ids = p+tuple(8+b for b in answer.encode())+(3,)
            rows.append({'id': f'{status}-{i}', 'kind': 'synthetic', 'status': status,
                         'ids': ids, 'prompt_len': len(p),
                         'loss_mask': (False,)*(len(p)-1)+(True,)*(len(ids)-len(p))})
    return q.validate_rows(rows, 64)


class ContractTests(unittest.TestCase):
    def test_balanced_schedule_and_complete_six_update_cycle(self):
        rows = fixture_rows()
        seen = []
        for step in range(6):
            batch = q.batch_rows(rows, step)
            self.assertEqual(Counter(r['status'] for r in batch), Counter({s: 1 for s in q.STATUSES}))
            seen.extend(r['id'] for r in batch)
        self.assertEqual(len(set(seen)), 18)
        self.assertEqual(q.batch_rows(rows, 5), q.batch_rows(rows, 5))

    def test_shifted_masked_targets_and_actual_counts(self):
        rows = q.batch_rows(fixture_rows(), 0)
        x, y, counts = q.collate(rows)
        self.assertEqual(counts['answer_byte_targets'], 6)
        self.assertEqual(counts['answer_eos_targets'], 3)
        self.assertEqual(counts['total_targets'], 9)
        for i, row in enumerate(rows):
            p = row['prompt_len']
            self.assertTrue(all(v == -100 for v in y[i][:p-1]))
            self.assertEqual(tuple(v for v in y[i] if v != -100), row['ids'][p:])
            self.assertEqual(x[i][:len(row['ids'])-1], row['ids'][:-1])

    def test_invalid_mask_and_bounds_rejected(self):
        rows = list(fixture_rows())
        bad = dict(rows[0]); mask = list(bad['loss_mask']); mask[0] = True; bad['loss_mask'] = tuple(mask); rows[0] = bad
        with self.assertRaises(ValueError): q.validate_rows(rows, 64)
        with self.assertRaises(ValueError): q.batch_rows(fixture_rows(), 128)
        self.assertEqual(q.gate_description()['cases'], 18)
        self.assertEqual(q.gate_description()['complete_grounded_raw_greedy_correct_required'], 18)


class NumericalFixtureTests(unittest.TestCase):
    def setup_fixture(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Parent must supply an admitted Torch runtime for CPU fixture checks.')
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(264, 4)
                self.dropout = torch.nn.Dropout(.1)  # tests actual RNG restoration
                self.out = torch.nn.Linear(4, 264)
            def forward(self, ids): return self.out(self.dropout(self.emb(ids)))
            def validate_parameters(self):
                if any(p.dtype != torch.float32 or not torch.isfinite(p).all() for p in self.parameters()):
                    raise ValueError('invalid tiny fixture state')
        identity = {'arm': 'fixture', 'source_sha256': 'a'*64,
                    'data': {'split': 'train', 'model_record_sha256': q.sha(q.canonical(fixture_rows()))},
                    'recipe': {'fixture': True}}
        return torch, Tiny, identity

    def execute(self, updates, checkpoint=None, checkpoint_sha256=None, identity=None):
        torch, Tiny, original = self.setup_fixture()
        return q._run_slice('fixture', fixture_rows(), lambda _: Tiny(), identity or original,
            updates=updates, checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha256,
            device='cpu', check=lambda: True, production=False)

    def test_fresh_vs_split_restore_model_adam_rng_cursor_exact(self):
        torch, _, _ = self.setup_fixture()
        full = self.execute(8)
        first = self.execute(4)
        resumed = self.execute(4, first['state'], first['state_sha256'])
        self.assertEqual(full['component_hashes'], resumed['component_hashes'])
        self.assertEqual(full['counts'], resumed['counts'])
        self.assertEqual(full['step'], 8)
        self.assertLessEqual(full['state_bytes'], 4*1024**2)
        self.assertEqual(resumed['training_rows'][0]['step'], 5)
        self.assertFalse(resumed['persisted'])

    def test_corrupt_pin_identity_stop_and_failed_guard_rejected(self):
        first = self.execute(1)
        with self.assertRaises(ValueError): self.execute(1, first['state'], '0'*64)
        _, _, identity = self.setup_fixture()
        identity = copy.deepcopy(identity); identity['data']['model_record_sha256'] = 'b'*64
        with self.assertRaises(ValueError): self.execute(1, first['state'], first['state_sha256'], identity)
        with self.assertRaises(ValueError): self.execute(9)
        torch, Tiny, original = self.setup_fixture()
        with self.assertRaises(ValueError):
            q._run_slice('fixture', fixture_rows(), lambda _: Tiny(), original, updates=1,
                device='cpu', check=lambda: False, production=False)


if __name__ == '__main__':
    unittest.main()
