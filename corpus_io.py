"""Bounded local ordinary-text corpus IO. No Torch, network, or fitted tokenizer.

Caller supplies an external corpus root, external manifest, and reviewed SHA256.
No source path or corpus text is embedded here. Test text is loaded only by an
explicit documents/evaluation request. Ordinary byte b always has ID8+b.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import unicodedata

PAD, UNK, BOS, EOS, CONFIRM, DEFER, QUERY_A, QUERY_B = range(8)
BYTE_OFFSET, VOCAB_SIZE = 8, 264
MAX_MANIFEST_BYTES = 65536
MAX_DOCUMENTS = 32
MAX_DOCUMENT_BYTES = 65536
MAX_CONTENT_BYTES = 262144
MAX_EVIDENCE_BYTES = 262144


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(obj):
    return sha(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode())


def content_fingerprint(raw):
    """Curation-only duplicate fingerprints; no vocabulary/statistic fitting."""
    text = raw.decode('utf-8', errors='strict')
    canonical = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    words = re.findall(r'\w+', canonical)
    shingles = {tuple(words[i:i+5]) for i in range(max(0, len(words)-4))}
    return {'canonical_sha256': sha(canonical.encode()),
            'shingles_sha256': _digest(sorted(shingles)), 'shingles': shingles}


def _validate_spec(spec, cap):
    if type(spec) is not dict:
        raise ValueError('file receipt must be an object')
    rel, size, expected = spec.get('path'), spec.get('bytes'), spec.get('sha256')
    if (type(rel) is not str or '\\' in rel or '\x00' in rel or
            PurePosixPath(rel).is_absolute() or '..' in PurePosixPath(rel).parts or
            re.match(r'^[a-zA-Z]:', rel) or type(size) is not int or not 0 < size <= cap or
            type(expected) is not str or not re.fullmatch('[0-9a-f]{64}', expected)):
        raise ValueError('invalid bounded relative file receipt')
    return rel, size, expected


def _read_bounded(path, cap, expected_size=None):
    size = path.stat().st_size
    if not 0 < size <= cap or (expected_size is not None and size != expected_size):
        raise ValueError('file bounded size mismatch')
    with path.open('rb') as stream:
        raw = stream.read(cap+1)
    if len(raw) > cap or len(raw) != size:
        raise ValueError('file grew/shrank during bounded read')
    return raw


def _relative_read(root, spec, cap):
    rel, size, expected = _validate_spec(spec, cap)
    path = (root/rel).resolve()
    if not path.is_relative_to(root):
        raise ValueError('file outside corpus root')
    raw = _read_bounded(path, cap, size)
    if sha(raw) != expected:
        raise ValueError('file SHA256 mismatch')
    return raw


def _public_duplicate_audit(audit):
    """Fixed method plus vetted scalar counts/rates only; never copy metadata."""
    if type(audit) is not dict or audit.get('method') != 'NFKC_casefold_whitespace_and_word5' or audit.get('cross_split_duplicates') != 0:
        raise ValueError('pinned cross-split curation audit required')
    result = {'method': 'NFKC_casefold_whitespace_and_word5'}
    integer_fields = {'cross_split_duplicates': MAX_DOCUMENTS**2, 'comparisons': MAX_DOCUMENTS**2,
                      'minimum_shingles': 100000}
    rate_fields = ('jaccard_threshold', 'containment_threshold', 'max_jaccard', 'max_containment')
    for name, upper in integer_fields.items():
        if name in audit:
            value = audit[name]
            if type(value) is not int or not 0 <= value <= upper:
                raise ValueError('invalid public audit count')
            result[name] = value
    for name in rate_fields:
        if name in audit:
            value = audit[name]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('invalid public audit rate')
            result[name] = value
    return result


class Corpus:
    """Pinned public-literature corpus with explicit no-fit heldout boundaries."""
    @classmethod
    def load(cls, manifest_path, expected_manifest_sha256, corpus_root):
        path = Path(manifest_path).resolve()
        raw = _read_bounded(path, MAX_MANIFEST_BYTES)
        if type(expected_manifest_sha256) is not str or sha(raw) != expected_manifest_sha256:
            raise ValueError('explicit exact manifest pin required')
        obj = json.loads(raw)
        if obj.get('schema') != 'corpus_first_external_v1':
            raise ValueError('unsupported corpus schema')
        documents = obj.get('documents')
        if type(documents) is not list or not 3 <= len(documents) <= MAX_DOCUMENTS:
            raise ValueError('document count bound')
        seen_ids, raw_hashes, canonical_hashes, group_splits, author_splits = set(), set(), set(), {}, {}
        totals = {s: 0 for s in ('train', 'dev', 'test')}
        for row in documents:
            if type(row) is not dict or type(row.get('id')) is not str or row['id'] in seen_ids:
                raise ValueError('unique document IDs required')
            if not re.fullmatch(r'[a-z0-9]+(?:[-_][a-z0-9]+)*', row['id']) or len(row['id']) > 160:
                raise ValueError('document ID must be a bounded path-free slug')
            seen_ids.add(row['id'])
            _validate_spec(row, MAX_DOCUMENT_BYTES)
            split = row.get('split')
            if split not in totals or type(row.get('bytes')) is not int or not 0 < row['bytes'] <= MAX_DOCUMENT_BYTES:
                raise ValueError('split/size contract')
            totals[split] += row['bytes']
            for key, registry in (('group', group_splits), ('author', author_splits)):
                value = row.get(key)
                if type(value) is not str or not value or registry.setdefault(value, split) != split:
                    raise ValueError('source group or author crosses splits')
            for key, seen in (('sha256', raw_hashes), ('canonical_sha256', canonical_hashes)):
                value = row.get(key)
                if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value) or value in seen:
                    raise ValueError('duplicate/missing content identity')
                seen.add(value)
            if row.get('evaluation_allowed') is not True or type(row.get('model_fit_allowed')) is not bool:
                raise ValueError('explicit rights/use scope required')
            if split == 'train' and row['model_fit_allowed'] is not True:
                raise ValueError('no-fit document cannot enter training')
            if row.get('rights_status') not in ('publisher_US_PD_belief_CC0_contributions', 'CC0_original_fixture'):
                raise ValueError('unreviewed license scope')
            if not all(type(row.get(k)) is str and row[k] for k in ('license_id', 'source_id', 'shingles_sha256')):
                raise ValueError('source/license/fingerprint metadata required')
            if type(row.get('evidence')) is not list or not 1 <= len(row['evidence']) <= 8:
                raise ValueError('bounded rights/provenance evidence required')
            for evidence in row['evidence']:
                _validate_spec(evidence, MAX_EVIDENCE_BYTES)
        if min(totals.values()) <= 0 or sum(totals.values()) > MAX_CONTENT_BYTES:
            raise ValueError('all splits nonempty and bounded')
        if obj.get('content_bytes') != totals or obj.get('content_byte_cap') != sum(totals.values()):
            raise ValueError('exact declared exposure cap required')
        public_audit = _public_duplicate_audit(obj.get('duplicate_audit', {}))
        result = cls()
        result.manifest = obj
        result.manifest_sha256 = expected_manifest_sha256
        result.root = Path(corpus_root).resolve()
        result.content_bytes = totals
        result.public_duplicate_audit = public_audit
        result._cache = {}
        result._fingerprints = {}
        train_identity = [{k: r[k] for k in ('id', 'sha256', 'group', 'bytes')}
                          for r in documents if r['split'] == 'train']
        result.training_identity = _digest(train_identity)
        return result

    def documents(self, split):
        if split not in ('train', 'dev', 'test'):
            raise ValueError('unknown corpus split')
        for row in self.manifest['documents']:
            if row['split'] != split or row['id'] in self._cache:
                continue
            for evidence in row['evidence']:
                _relative_read(self.root, evidence, MAX_EVIDENCE_BYTES)
            raw = _relative_read(self.root, row, MAX_DOCUMENT_BYTES)
            fp = content_fingerprint(raw)
            if fp['canonical_sha256'] != row['canonical_sha256'] or fp['shingles_sha256'] != row['shingles_sha256']:
                raise ValueError('canonical/shingle curation receipt mismatch')
            for old_id, old_fp in self._fingerprints.items():
                old = next(r for r in self.manifest['documents'] if r['id'] == old_id)
                if old['split'] == split:
                    continue
                a, b = fp['shingles'], old_fp['shingles']
                if len(a) >= 20 and len(b) >= 20:
                    overlap = len(a & b)
                    if overlap/max(1, len(a | b)) >= 0.80 or overlap/min(len(a), len(b)) >= 0.90:
                        raise ValueError('loaded cross-split near duplicate')
            self._cache[row['id']] = {'id': row['id'], 'split': split, 'group': row['group'],
                'source_id': row['source_id'], 'license_id': row['license_id'],
                'raw': raw, 'sha256': row['sha256']}
            self._fingerprints[row['id']] = fp
        return tuple(dict(self._cache[r['id']]) for r in self.manifest['documents'] if r['split'] == split)

    @staticmethod
    def _window(doc, start, end):
        raw = doc['raw'][start:end]
        at_end = end == len(doc['raw'])
        ids = [BOS]+[BYTE_OFFSET+b for b in raw]+([EOS] if at_end else [])
        return {'ids': ids, 'doc_id': doc['id'], 'start': start, 'end': end,
                'content_targets': len(raw), 'synthetic_eos_targets': int(at_end)}

    def sample_batch(self, step, seed, batch_size, seq_len):
        """Train-only stateless weighted valid-byte-start sampling with replacement.

        Each window resets at BOS and contains exactly seq_len byte targets.
        Byte windows may start/end within a UTF8 scalar; the byte codec is exact.
        Heldout identity/content never affects the sampling hash or weights.
        """
        if not all(type(x) is int for x in (step, seed, batch_size, seq_len)) or not (0 <= step <= 10**9 and 1 <= batch_size <= 64 and 1 <= seq_len <= 2048):
            raise ValueError('bounded integer sampling recipe required')
        docs = self.documents('train')
        counts = [max(0, len(d['raw'])-seq_len+1) for d in docs]
        total = sum(counts)
        if not total:
            raise ValueError('no document supports requested byte window')
        result = []
        for draw in range(batch_size):
            index = int(_digest([self.training_identity, seed, step, draw, seq_len]), 16) % total
            for doc, count in zip(docs, counts):
                if index < count:
                    result.append(self._window(doc, index, index+seq_len))
                    break
                index -= count
        return result

    def evaluation_windows(self, split, seq_len, max_windows=None):
        """Fixed disjoint windows, round-robin across documents; explicit split IO."""
        if split not in ('dev', 'test') or type(seq_len) is not int or not 1 <= seq_len <= 2048:
            raise ValueError('heldout split and bounded byte context required')
        if max_windows is not None and (type(max_windows) is not int or not 1 <= max_windows <= 4096):
            raise ValueError('fixed bounded evaluation window count required')
        docs = self.documents(split)
        offsets = [0]*len(docs)
        windows = []
        while any(offset < len(doc['raw']) for offset, doc in zip(offsets, docs)):
            for i, doc in enumerate(docs):
                if offsets[i] >= len(doc['raw']):
                    continue
                end = min(offsets[i]+seq_len, len(doc['raw']))
                windows.append(self._window(doc, offsets[i], end))
                offsets[i] = end
                if max_windows is not None and len(windows) >= max_windows:
                    return windows
        return windows

    def receipt(self):
        """Path/text-free public identities and exposure metadata. No document IO."""
        # Free-form author/group/source/license strings are intentionally kept
        # local. Public reproduction uses pinned hashes, not arbitrary metadata.
        public_keys = ('id', 'split', 'bytes', 'sha256', 'rights_status',
                       'model_fit_allowed', 'evaluation_allowed')
        return {'manifest_sha256': self.manifest_sha256, 'training_identity': self.training_identity,
                'content_bytes': dict(self.content_bytes),
                'public_evaluation_not_blind': True, 'codec': 'raw_UTF8_bytes_offset8_vocab264',
                'documents': [{**{k: row[k] for k in public_keys},
                    'evidence_hashes': [{'bytes': e['bytes'], 'sha256': e['sha256']} for e in row['evidence']]}
                    for row in self.manifest['documents']],
                'duplicate_audit': dict(self.public_duplicate_audit),
                'sampling': 'stateless_train_only_valid_byte_starts_with_window_reset',
                'evaluation': 'disjoint_round_robin_document_windows_with_window_reset',
                'eos_scope': 'synthetic chapter-fragment termination, not natural book endings'}
