"""Stdlib fabricated fixtures only; never opens an actual study holdout."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import io
import corpus_io as c


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        evidence = b'Original isolated test fixture. Dedicated under CC0.'
        (self.root/'rights.txt').write_bytes(evidence)
        self.docs = []
        texts = {'train': 'A small workshop opened near the bridge. '*8,
                 'dev': 'Birds gathered beside the lake during autumn. '*7,
                 'test': 'The observatory kept careful records of winter skies. '*6}
        for split, text in texts.items():
            raw = text.encode()
            path = f'{split}.txt'
            (self.root/path).write_bytes(raw)
            fp = c.content_fingerprint(raw)
            self.docs.append({'id': split+'-doc', 'path': path, 'split': split,
                'group': split+'-work', 'author': split+'-author', 'source_id': 'fixture:'+split,
                'license_id': 'CC0-1.0', 'rights_status': 'CC0_original_fixture',
                'model_fit_allowed': split != 'test', 'evaluation_allowed': True,
                'bytes': len(raw), 'sha256': c.sha(raw), 'canonical_sha256': fp['canonical_sha256'],
                'shingles_sha256': fp['shingles_sha256'],
                'evidence': [{'path': 'rights.txt', 'bytes': len(evidence), 'sha256': c.sha(evidence)}]})
        totals = {r['split']: r['bytes'] for r in self.docs}
        self.obj = {'schema': 'corpus_first_external_v1', 'documents': self.docs,
                    'content_bytes': totals, 'content_byte_cap': sum(totals.values()),
                    'duplicate_audit': {'cross_split_duplicates': 0, 'method': 'NFKC_casefold_whitespace_and_word5'}}

    def load(self, obj=None):
        raw = json.dumps(self.obj if obj is None else obj, sort_keys=True).encode()
        path = self.root/'manifest.json'
        path.write_bytes(raw)
        return c.Corpus.load(path, c.sha(raw), self.root)

    def test_lazy_and_path_free_receipt(self):
        corpus = self.load()
        (self.root/'test.txt').unlink()
        corpus.receipt()
        corpus.sample_batch(0, 12, 2, 32)
        self.assertNotIn(str(self.root), json.dumps(corpus.receipt()))
        with self.assertRaises(FileNotFoundError):
            corpus.documents('test')

    def test_stateless_resume_and_byte_identity(self):
        corpus = self.load()
        a = corpus.sample_batch(9, 12, 8, 64)
        corpus.sample_batch(8, 12, 8, 64)
        self.assertEqual(a, corpus.sample_batch(9, 12, 8, 64))
        raw = corpus.documents('train')[0]['raw']
        for row in a:
            body = [t-8 for t in row['ids'][1:] if t >= 8]
            self.assertEqual(bytes(body), raw[row['start']:row['end']])
            self.assertEqual(row['content_targets'], 64)
            self.assertFalse(set(body).difference(range(256)))

    def test_heldout_changes_do_not_change_train_draws(self):
        before = self.load().sample_batch(3, 10, 8, 32)
        obj = copy.deepcopy(self.obj)
        raw = b'Fresh entirely separate fictional public verification sample. '*6
        (self.root/'test.txt').write_bytes(raw)
        fp = c.content_fingerprint(raw)
        obj['documents'][2].update(bytes=len(raw), sha256=c.sha(raw), canonical_sha256=fp['canonical_sha256'], shingles_sha256=fp['shingles_sha256'])
        obj['content_bytes']['test'] = len(raw)
        obj['content_byte_cap'] = sum(obj['content_bytes'].values())
        self.assertEqual(before, self.load(obj).sample_batch(3, 10, 8, 32))

    def test_eval_disjoint_full_coverage_and_eos(self):
        corpus = self.load()
        windows = corpus.evaluation_windows('dev', 31)
        self.assertEqual(sum(r['content_targets'] for r in windows), self.docs[1]['bytes'])
        self.assertEqual(sum(r['synthetic_eos_targets'] for r in windows), 1)
        self.assertTrue(all(a['end'] == b['start'] for a, b in zip(windows, windows[1:])))
        self.assertEqual(corpus.evaluation_windows('dev', 31, 2), windows[:2])

    def test_no_fit_and_group_boundaries(self):
        obj = copy.deepcopy(self.obj)
        obj['documents'][0]['model_fit_allowed'] = False
        with self.assertRaises(ValueError): self.load(obj)
        obj = copy.deepcopy(self.obj)
        obj['documents'][1]['group'] = obj['documents'][0]['group']
        with self.assertRaises(ValueError): self.load(obj)

    def test_duplicate_manifest_identity_rejected(self):
        obj = copy.deepcopy(self.obj)
        obj['documents'][1]['canonical_sha256'] = obj['documents'][0]['canonical_sha256']
        with self.assertRaises(ValueError): self.load(obj)

    def test_path_escape_and_hash_tamper(self):
        obj = copy.deepcopy(self.obj)
        obj['documents'][0]['path'] = '../outside.txt'
        with self.assertRaises(ValueError): self.load(obj).documents('train')
        corpus = self.load()
        (self.root/'train.txt').write_bytes(b'x'*self.docs[0]['bytes'])
        with self.assertRaises(ValueError): corpus.documents('train')

    def test_pin_and_sampling_bounds(self):
        corpus = self.load()
        with self.assertRaises(ValueError): c.Corpus.load(self.root/'manifest.json', '0'*64, self.root)
        with self.assertRaises(ValueError): corpus.sample_batch(0, 12, 0, 32)
        with self.assertRaises(ValueError): corpus.evaluation_windows('train', 32)

    def test_loaded_near_duplicate_rejected(self):
        obj = copy.deepcopy(self.obj)
        train = (' '.join(f'unique{i}' for i in range(60))+'.').encode()
        dev = train+b' A short changed ending.'
        for index, raw in ((0, train), (1, dev)):
            row = obj['documents'][index]
            (self.root/row['path']).write_bytes(raw)
            fp = c.content_fingerprint(raw)
            row.update(bytes=len(raw), sha256=c.sha(raw), canonical_sha256=fp['canonical_sha256'], shingles_sha256=fp['shingles_sha256'])
            obj['content_bytes'][row['split']] = len(raw)
        obj['content_byte_cap'] = sum(obj['content_bytes'].values())
        corpus = self.load(obj)
        corpus.documents('train')
        with self.assertRaises(ValueError): corpus.documents('dev')

    def test_protocol_spellings_remain_ordinary_bytes(self):
        raw = b'<confirm> <query_a> <query_b>'
        doc = {'raw': raw, 'id': 'literal'}
        row = c.Corpus._window(doc, 0, len(raw))
        self.assertFalse({c.CONFIRM, c.DEFER, c.QUERY_A, c.QUERY_B}.intersection(row['ids']))
        self.assertEqual(bytes(t-8 for t in row['ids'] if t >= 8), raw)

    def test_rights_evidence_tampering_rejected(self):
        corpus = self.load()
        path = self.root/'rights.txt'
        path.write_bytes(b'x'*len(path.read_bytes()))
        with self.assertRaises(ValueError): corpus.documents('train')

    def test_private_freeform_metadata_never_exported(self):
        obj = copy.deepcopy(self.obj)
        secret = r'C:\private\person\secret-file.txt'
        obj['duplicate_audit'].update(private_path=secret, normalization=secret, notes=[secret])
        obj['documents'][0].update(author=secret, group=secret, source_id=secret,
                                   license_id=secret, private_extra=secret)
        obj['documents'][0]['evidence'][0]['private_extra'] = secret
        receipt = self.load(obj).receipt()
        exported = json.dumps(receipt)
        self.assertNotIn(secret, exported)
        self.assertNotIn('private', exported)
        self.assertNotIn('normalization', receipt['duplicate_audit'])
        self.assertEqual(set(receipt['duplicate_audit']), {'method', 'cross_split_duplicates'})
        for name in ('author', 'group', 'source_id', 'license_id'):
            self.assertNotIn(name, receipt['documents'][0])

    def test_private_id_or_nonnumeric_audit_rejected(self):
        obj = copy.deepcopy(self.obj)
        obj['documents'][0]['id'] = 'private/path.txt'
        with self.assertRaises(ValueError): self.load(obj)
        obj = copy.deepcopy(self.obj)
        obj['duplicate_audit']['comparisons'] = r'C:\private\secret'
        with self.assertRaises(ValueError): self.load(obj)
        obj = copy.deepcopy(self.obj)
        obj['documents'][0]['evidence'][0]['bytes'] = r'C:\private\secret'
        with self.assertRaises(ValueError): self.load(obj)

    def test_read_is_max_plus_one_and_rejects_growth(self):
        path = self.root/'bounded.txt'
        path.write_bytes(b'x')
        class Probe(io.BytesIO):
            def __init__(self, raw):
                super().__init__(raw)
                self.requests = []
            def read(self, size=-1):
                self.requests.append(size)
                return super().read(size)
        probe = Probe(b'x'*100)
        with mock.patch.object(Path, 'open', return_value=probe):
            with self.assertRaises(ValueError): c._read_bounded(path, 10)
        self.assertEqual(probe.requests, [11])


if __name__ == '__main__':
    unittest.main()
