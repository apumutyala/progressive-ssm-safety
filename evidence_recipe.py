"""Train-only batch plans for ordinary bytes and matched fictional completions.

This module has no model, optimizer, certificate, remote action or holdout reader.
Corpus.load retains responsibility for approved rights/source intake. An EOS
attestation establishes a reviewed chapter boundary; it does not grant rights
or establish a book ending. Unattested fragment/window EOS tokens are removed.
Every objective term has unit token weight and the batch's ONE normalizer is
its total supervised-target count, including genuine source/answer EOS targets.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Callable

from corpus_io import Corpus, BOS, EOS, BYTE_OFFSET, VOCAB_SIZE

ARMS = ('true', 'shuffled')
MAX_GROUPS = 512
MAX_ORDINARY_RECORDS = 64


def _need(condition, message):
    if not condition: raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _sha(raw): return hashlib.sha256(raw).hexdigest()
def _pin(value): return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


@dataclass(frozen=True)
class PlannedRecord:
    id: str
    kind: str
    ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]  # aligned to ids[1:], not input positions
    prompt_len: int
    lineage: tuple[tuple[str, object], ...]

    @property
    def supervised_targets(self): return sum(self.loss_mask)

    def as_record(self):
        """Fresh plain containers for a later guarded runner; no model side inputs."""
        return {'id': self.id, 'kind': self.kind, 'ids': list(self.ids), 'loss_mask': list(self.loss_mask),
                'prompt_len': self.prompt_len, 'lineage': dict(self.lineage), 'events': None}


@dataclass(frozen=True)
class BatchPlan:
    arm: str
    step: int
    seed: int
    group_id: str
    records: tuple[PlannedRecord, ...]
    ordinary_byte_targets: int
    ordinary_source_eos_targets: int
    synthetic_answer_byte_targets: int
    synthetic_answer_eos_targets: int
    ordinary_draw_sha256: str
    synthetic_prompt_sha256: str
    synthetic_response_multiset_sha256: str
    plan_sha256: str

    @property
    def ordinary_targets(self): return self.ordinary_byte_targets+self.ordinary_source_eos_targets
    @property
    def synthetic_targets(self): return self.synthetic_answer_byte_targets+self.synthetic_answer_eos_targets
    @property
    def total_targets(self): return self.ordinary_targets+self.synthetic_targets
    @property
    def loss_normalizer(self): return self.total_targets

    def summary(self):
        """Counts and proof flags only; publication of identities is a separate decision."""
        return {'ordinary_byte_targets': self.ordinary_byte_targets,
            'ordinary_verified_chapter_eos_targets': self.ordinary_source_eos_targets,
            'synthetic_answer_byte_targets': self.synthetic_answer_byte_targets,
            'synthetic_answer_eos_targets': self.synthetic_answer_eos_targets,
            'synthetic_prompt_targets': 0, 'total_supervised_targets': self.total_targets,
            'ordinary_supervised_fraction': self.ordinary_targets/self.total_targets,
            'ordinary_byte_fraction_including_all_eos_in_denominator': self.ordinary_byte_targets/self.total_targets,
            'ordinary_records': sum(r.kind == 'ordinary' for r in self.records), 'synthetic_records': 9,
            'unit_weight_per_target': True, 'one_global_target_normalizer': self.loss_normalizer,
            'ordinary_eos_requires_reviewed_chapter_attestation': True,
            'answer_eos_means_complete_fictional_reply': True}

    def as_records(self): return [record.as_record() for record in self.records]


class EvidenceRecipe:
    """Deterministic paired plans: complete3x3 group plus >=75% real byte targets.

    group_source(split='train',seed=...,arm='true'|'shuffled') must yield matched
    groups {'id':str,'rows':[...9...]}. Rows need id,ids,prompt_len. The three
    unique prompts each repeat3 times; true labels are deterministic per prompt,
    while the shuffled Latin control gives every prompt each of3 replies once.
    Neither group selection nor ordinary sampling uses heldout information.
    """
    def __init__(self, corpus, group_source: Callable, *, chapter_boundaries=(), seq_len=128):
        _need(type(seq_len) is int and 1 <= seq_len <= 512, 'Bounded ordinary byte context required.')
        self.corpus = corpus; self.group_source = group_source; self.seq_len = seq_len; self._groups = {}
        docs = tuple(corpus.documents('train'))
        _need(0 < len(docs) <= 32, 'Nonempty bounded training source required.')
        self.docs = {}
        for doc in docs:
            _need(doc.get('split') == 'train' and type(doc.get('raw')) is bytes and doc['raw'], 'Training documents only.')
            _need(doc['id'] not in self.docs and _sha(doc['raw']) == doc['sha256'], 'Training document identity mismatch.')
            self.docs[doc['id']] = doc
        self.boundaries = {}
        for annotation in chapter_boundaries:
            _need(type(annotation) is dict, 'Boundary attestation must be an object.')
            doc = self.docs.get(annotation.get('doc_id'))
            _need(doc is not None, 'Boundary refers to a non-training document.')
            _need(annotation.get('doc_sha256') == doc['sha256'] and annotation.get('end') == len(doc['raw']),
                  'Boundary must bind the complete admitted source bytes.')
            _need(type(annotation.get('end')) is int and annotation.get('unit') == 'chapter'
                  and annotation.get('status') == 'verified_complete_chapter' and _pin(annotation.get('evidence_sha256')),
                  'Only reviewed complete-chapter provenance admits source EOS.')
            _need(doc['id'] not in self.boundaries, 'Duplicate chapter boundary attestation.')
            self.boundaries[doc['id']] = dict(annotation)

    @classmethod
    def from_manifest(cls, manifest_path, expected_manifest_sha256, corpus_root, group_source, *, boundary_provider=None, seq_len=128):
        """Future guarded-runner entrypoint; tests need not touch external sources.

        boundary_provider may be source_boundary.certify_training_chapters.
        Its reviewed XML replay is separate from corpus rights validation.
        """
        corpus = Corpus.load(manifest_path, expected_manifest_sha256, corpus_root=corpus_root)
        annotations = () if boundary_provider is None else boundary_provider(corpus)
        return cls(corpus, group_source, chapter_boundaries=annotations, seq_len=seq_len)

    @staticmethod
    def _synthetic(row, group):
        values = tuple(row['ids']); prompt = row['prompt_len']
        _need(type(prompt) is int and 1 < prompt < len(values)-1 and len(values) <= 1024,
              'Synthetic prompt and complete nonempty reply required.')
        _need(values[0] == BOS and values[-1] == EOS and all(type(t) is int and BYTE_OFFSET <= t < VOCAB_SIZE for t in values[1:-1]),
              'Fictional prompts/replies must use byte IDs and exactly one terminal EOS.')
        _need(len(values)-prompt <= 256, 'Synthetic reply exceeds bounded recipe.')
        mask = tuple(i >= prompt-1 for i in range(len(values)-1))
        return PlannedRecord(str(row['id']), 'synthetic', values, mask, prompt,
                             (('training_group', group), ('eos_unit', 'complete_fictional_reply')))

    def _load_groups(self, seed):
        if seed in self._groups: return self._groups[seed]
        _need(len(self._groups) < 4, 'Use a fresh planner after four distinct cached seeds.')
        result = {}
        for arm in ARMS:
            groups = {}
            for group in self.group_source(split='train', seed=seed, arm=arm):
                _need(len(groups) < MAX_GROUPS and type(group['id']) is str and group['id'] not in groups, 'Bounded unique training groups required.')
                _need(len(group['rows']) == 9, 'Each training group must contain all9 conditional cells.')
                rows = tuple(self._synthetic(r, group['id']) for r in group['rows'])
                _need(len({r.id for r in rows}) == 9, 'Training row IDs must be unique within group.')
                groups[group['id']] = tuple(sorted(rows, key=lambda r: r.id))
            _need(groups, 'Nonempty training groups required.'); result[arm] = groups
        _need(result['true'].keys() == result['shuffled'].keys(), 'Matched arms must have identical training group identities.')
        for key, real in result['true'].items():
            control = result['shuffled'][key]
            _need([r.id for r in real] == [r.id for r in control], 'Matched synthetic row identity mismatch.')
            for a, b in zip(real, control):
                _need(a.prompt_len == b.prompt_len and a.ids[:a.prompt_len] == b.ids[:b.prompt_len], 'Control changed the prompt bytes.')
            replies = lambda rows: Counter(r.ids[r.prompt_len:] for r in rows)
            prompts = Counter(r.ids[:r.prompt_len] for r in real)
            answer_counts = replies(real)
            _need(answer_counts == replies(control) and len(answer_counts) == 3 and set(answer_counts.values()) == {3},
                  'Control must preserve the complete reply/EOS/length multiset exactly.')
            _need(len(prompts) == 3 and set(prompts.values()) == {3}, 'Three identical-prompt replica triplets required.')
            for prompt_ids in prompts:
                real_replies = replies([r for r in real if r.ids[:r.prompt_len] == prompt_ids])
                null_replies = replies([r for r in control if r.ids[:r.prompt_len] == prompt_ids])
                _need(len(real_replies) == 1 and set(real_replies.values()) == {3}, 'Real source must have deterministic replies for identical prompts.')
                _need(null_replies == Counter({answer: 1 for answer in answer_counts}),
                      'Shuffled conditional must give every identical prompt all3 complete replies once.')
        self._groups[seed] = result
        return result

    def _ordinary(self, row, identifier):
        doc = self.docs.get(row['doc_id']); start, end = row['start'], row['end']
        _need(doc is not None and type(start) is int and type(end) is int and 0 <= start < end <= len(doc['raw']), 'Ordinary source span must stay in a training document.')
        raw = doc['raw'][start:end]; encoded = (BOS,)+tuple(BYTE_OFFSET+b for b in raw)
        supplied = tuple(row['ids'])
        _need(supplied in (encoded, encoded+(EOS,)) and row['content_targets'] == len(raw), 'Ordinary source-byte/target lineage mismatch.')
        attestation = self.boundaries.get(doc['id']) if end == len(doc['raw']) else None
        values = encoded+((EOS,) if attestation else ())
        lineage = {'doc_id': doc['id'], 'doc_sha256': doc['sha256'], 'start': start, 'end': end,
                   'source_id': doc.get('source_id'), 'license_id': doc.get('license_id'),
                   'source_eos_unit': 'chapter' if attestation else 'none'}
        if attestation: lineage['boundary_evidence_sha256'] = attestation['evidence_sha256']
        return PlannedRecord(identifier, 'ordinary', values, (True,)*(len(values)-1), 1, tuple(sorted(lineage.items())))

    def paired_plan(self, step, seed=12):
        _need(type(step) is int and 0 <= step <= 10**9 and type(seed) is int, 'Bounded integer step/seed required.')
        groups = self._load_groups(seed)
        keys = sorted(groups['true'])
        key = keys[int(_digest(['group', seed, step]), 16) % len(keys)]
        real = groups['true'][key]; control = groups['shuffled'][key]
        synthetic_targets = sum(r.supervised_targets for r in real)
        ordinary = []
        if self.boundaries:
            names = sorted(self.boundaries)
            name = names[int(_digest(['boundary', seed, step]), 16) % len(names)]
            doc = self.docs[name]; end = len(doc['raw']); start = max(0, end-self.seq_len)
            row = {'doc_id': name, 'start': start, 'end': end, 'content_targets': end-start,
                   'ids': [BOS]+[BYTE_OFFSET+b for b in doc['raw'][start:end]]+[EOS]}
            ordinary.append(self._ordinary(row, f'ordinary-{step}-boundary'))
        # Demand75% from ordinary BYTES even counting both kinds of EOS in the
        # denominator. This is stronger than75% ordinary-supervised targets.
        draws = max(0, math.ceil(3*synthetic_targets/self.seq_len)-len(ordinary))
        while True:
            _need(draws+len(ordinary) <= MAX_ORDINARY_RECORDS, 'Complete-group75% target floor exceeds bounded batch; never trim cells or alter weights.')
            sampled = (self.corpus.sample_batch(step=step, seed=seed, batch_size=draws, seq_len=self.seq_len)
                       if draws else [])
            _need(len(sampled) == draws, 'Corpus sampler changed the requested draw count.')
            random_records = [self._ordinary(r, f'ordinary-{step}-{i}') for i, r in enumerate(sampled)]
            records = tuple(ordinary+random_records)
            ordinary_bytes = sum(sum(t >= BYTE_OFFSET for t in r.ids[1:]) for r in records)
            ordinary_eos = sum(r.ids[-1] == EOS for r in records)
            total = ordinary_bytes+ordinary_eos+synthetic_targets
            if 4*ordinary_bytes >= 3*total: break
            draws += 1
        ordinary_pin = _digest([(r.id, r.ids, r.loss_mask, r.lineage) for r in records])
        prompt_pin = _digest([(r.id, r.ids[:r.prompt_len]) for r in real])
        response_pin = _digest(sorted(r.ids[r.prompt_len:] for r in real))
        plans = {}
        for arm, synthetic in (('true', real), ('shuffled', control)):
            combined = records+synthetic
            answers = sum(sum(t >= BYTE_OFFSET and keep for t, keep in zip(r.ids[1:], r.loss_mask)) for r in synthetic)
            eos = sum(r.ids[-1] == EOS and r.loss_mask[-1] for r in synthetic)
            payload = [(r.id, r.ids, r.loss_mask, r.kind) for r in combined]
            plans[arm] = BatchPlan(arm, step, seed, key, combined, ordinary_bytes, ordinary_eos, answers, eos,
                                   ordinary_pin, prompt_pin, response_pin, _digest([arm, step, seed, payload]))
            _need(plans[arm].total_targets == sum(r.supervised_targets for r in combined), 'Actual loss-mask target accounting mismatch.')
            _need(4*plans[arm].ordinary_targets >= 3*plans[arm].total_targets, 'Ordinary target floor failed.')
        return plans

    def plan(self, arm, step, seed=12):
        if arm == 'real': arm = 'true'
        _need(arm in ARMS, 'Use true/real or shuffled training arm.')
        return self.paired_plan(step, seed)[arm]
