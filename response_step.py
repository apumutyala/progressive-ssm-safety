"""In-memory collation/accounting and an UNEXECUTED lazy gradient adapter.

Pure Python helpers never import Torch or inspect a corpus. The future adapter
only accumulates gradients; it never clips them, chooses optimizer acceptance,
steps/zeros an optimizer, changes model mode, writes files or updates a ledger.
The caller must start with gradients None and separately perform its reviewed
acceptance/clipping/update checks. On any exception, partial gradients are
invalid and the caller must discard them. No rollback/acceptance is implied.

All supervised tokens have unit weight. Every microbatch divides its SUM by
the SAME whole-plan target count. Different accumulation/padding orders can
change floating-point rounding; mathematical equivalence is not bit parity.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import re

from evidence_recipe import BatchPlan, PlannedRecord, MAX_ORDINARY_RECORDS
from corpus_io import BOS, EOS, BYTE_OFFSET, VOCAB_SIZE

IGNORE_INDEX = -100
MAX_RECORDS = MAX_ORDINARY_RECORDS+9
MAX_SEQUENCE_IDS = 1024


def _need(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Accounting:
    ordinary_byte_targets: int
    ordinary_source_eos_targets: int
    synthetic_answer_byte_targets: int
    synthetic_answer_eos_targets: int
    ordinary_records: int
    synthetic_records: int

    @property
    def total_targets(self):
        return (self.ordinary_byte_targets+self.ordinary_source_eos_targets+
                self.synthetic_answer_byte_targets+self.synthetic_answer_eos_targets)

    def summary(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__} | {
            'total_targets': self.total_targets, 'synthetic_prompt_targets': 0,
            'global_loss_normalizer': self.total_targets,
            'ordinary_byte_fraction': self.ordinary_byte_targets/self.total_targets}


@dataclass(frozen=True)
class CollatedBatch:
    input_ids: tuple[tuple[int, ...], ...]
    targets: tuple[tuple[int, ...], ...]
    loss_mask: tuple[tuple[bool, ...], ...]
    input_lengths: tuple[int, ...]
    kinds: tuple[str, ...]  # loss-accounting metadata; NEVER a model input
    record_ids: tuple[str, ...]  # in-memory provenance; no export helper
    supervised_targets: int
    padding_positions: int


@dataclass(frozen=True)
class PreparedStep:
    accounting: Accounting
    microbatches: tuple[CollatedBatch, ...]
    global_normalizer: int


def _record_counts(record):
    _need(isinstance(record, PlannedRecord), 'Expected a frozen PlannedRecord.')
    ids, mask = record.ids, record.loss_mask
    _need(type(record.id) is str and record.id and record.kind in ('ordinary', 'synthetic'), 'Record identity/kind missing.')
    _need(type(ids) is tuple and 2 <= len(ids) <= MAX_SEQUENCE_IDS and ids[0] == BOS,
          'Bounded BOS-framed sequence required.')
    _need(all(type(t) is int and (BYTE_OFFSET <= t < VOCAB_SIZE or (t == EOS and i == len(ids)-1))
              for i, t in enumerate(ids[1:], 1)), 'Only ordinary bytes and one optional terminal EOS may follow BOS.')
    _need(type(mask) is tuple and len(mask) == len(ids)-1 and all(type(m) is bool for m in mask), 'Prediction-aligned Boolean mask required.')
    _need(type(record.prompt_len) is int, 'Integer prompt boundary required.')
    if record.kind == 'ordinary':
        _need(record.prompt_len == 1 and all(mask), 'All ordinary byte/EOS targets must be supervised.')
        if ids[-1] == EOS:
            lineage = dict(record.lineage)
            _need(lineage.get('source_eos_unit') == 'chapter'
                  and type(lineage.get('boundary_evidence_sha256')) is str
                  and re.fullmatch('[0-9a-f]{64}', lineage['boundary_evidence_sha256']), 'Ordinary EOS requires recipe chapter attestation.')
    else:
        p = record.prompt_len
        _need(1 < p < len(ids)-1 and ids[-1] == EOS, 'Complete nonempty synthetic reply required.')
        _need(mask == tuple(i >= p-1 for i in range(len(ids)-1)), 'Synthetic mask must exclude prompt and include full reply/EOS.')
    byte_count = sum(keep and t >= BYTE_OFFSET for t, keep in zip(ids[1:], mask))
    eos_count = sum(keep and t == EOS for t, keep in zip(ids[1:], mask))
    _need(byte_count > 0 and eos_count <= 1, 'A record must contribute ordinary byte targets.')
    return byte_count, eos_count


def validate_plan(plan):
    _need(isinstance(plan, BatchPlan), 'Expected EvidenceRecipe.BatchPlan.')
    _need(plan.arm in ('true', 'shuffled') and type(plan.step) is int and plan.step >= 0,
          'Plan arm/step invalid.')
    _need(type(plan.records) is tuple and 10 <= len(plan.records) <= MAX_RECORDS,
          'Bounded complete nine-row group plus ordinary source records required.')
    _need(len({r.id for r in plan.records}) == len(plan.records), 'Duplicate record identity.')
    counts = {'ordinary': [0, 0, 0], 'synthetic': [0, 0, 0]}
    for record in plan.records:
        byte_count, eos_count = _record_counts(record)
        row = counts[record.kind]
        row[0] += byte_count
        row[1] += eos_count
        row[2] += 1
    ordinary, synthetic = counts['ordinary'], counts['synthetic']
    measured = Accounting(ordinary[0], ordinary[1], synthetic[0], synthetic[1], ordinary[2], synthetic[2])
    _need(synthetic[2] == synthetic[1] == 9, 'Complete nine-row reply/EOS group required.')
    for key in ('ordinary_byte_targets', 'ordinary_source_eos_targets',
                'synthetic_answer_byte_targets', 'synthetic_answer_eos_targets'):
        _need(type(getattr(plan, key)) is int and getattr(plan, key) == getattr(measured, key),
              'Plan target totals differ from actual IDs/masks.')
    _need(plan.total_targets == measured.total_targets == plan.loss_normalizer > 0,
          'One exact global supervised-target denominator required.')
    _need(4*measured.ordinary_byte_targets >= 3*measured.total_targets, 'Ordinary-byte target floor failed.')
    return measured


def collate_records(records):
    """Pad only after valid inputs; mask prompt and padding targets to -100.

    No state, metadata, lineage, typed evidence or events enter input_ids.
    Padded EOS inputs can be computed by a recurrent model but their targets
    contribute zero loss. They occur only AFTER that row's valid computation.
    """
    _need(type(records) in (tuple, list) and 0 < len(records) <= MAX_RECORDS,
          'Bounded materialized nonempty record sequence required.')
    records = tuple(records)
    counts = [_record_counts(r) for r in records]
    lengths = tuple(len(r.ids)-1 for r in records)
    width = max(lengths)
    inputs, targets, masks = [], [], []
    for record, length in zip(records, lengths):
        pad = width-length
        inputs.append(record.ids[:-1]+(EOS,)*pad)
        targets.append(tuple(t if keep else IGNORE_INDEX for t, keep in zip(record.ids[1:], record.loss_mask))+(IGNORE_INDEX,)*pad)
        masks.append(record.loss_mask+(False,)*pad)
    total = sum(a+b for a, b in counts)
    _need(sum(t != IGNORE_INDEX for row in targets for t in row) == total,
          'Collation changed supervised-target count.')
    return CollatedBatch(tuple(inputs), tuple(targets), tuple(masks), lengths,
                         tuple(r.kind for r in records), tuple(r.id for r in records),
                         total, sum(width-n for n in lengths))


def prepare_step(plan, *, microbatch_records=1, group_by_length=True):
    accounting = validate_plan(plan)
    _need(type(microbatch_records) is int and 1 <= microbatch_records <= MAX_RECORDS
          and type(group_by_length) is bool, 'Bounded microbatch size/Boolean grouping required.')
    records = list(plan.records)
    if group_by_length:
        records.sort(key=lambda r: len(r.ids))  # stable, deterministic
    batches = tuple(collate_records(records[i:i+microbatch_records])
                    for i in range(0, len(records), microbatch_records))
    _need(sum(b.supervised_targets for b in batches) == accounting.total_targets,
          'Microbatch partition changed global target count.')
    _need(sorted(r for b in batches for r in b.record_ids) == sorted(r.id for r in plan.records),
          'Microbatch partition dropped/duplicated a row.')
    return PreparedStep(accounting, batches, accounting.total_targets)


def normalized_sum(loss_sums, global_normalizer):
    """Stdlib accounting analogue; no per-microbatch averaging/normalization."""
    _need(type(global_normalizer) is int and global_normalizer > 0, 'Positive global target count required.')
    values = tuple(loss_sums)
    _need(values and all(type(v) in (int, float) and math.isfinite(v) for v in values),
          'Finite scalar microbatch loss sums required.')
    return math.fsum(values)/global_normalizer


def accumulate_gradients(model, plan, *, device=None, microbatch_records=1,
                         group_by_length=True, check=None):
    """FUTURE EXPLICIT CALL ONLY: finite fp32 CE accumulation, no update.

    Torch imports occur only here. This adapter has not been numerically tested
    by its stdlib test suite. Caller owns model admission, resource supervision,
    dynamics/certificate audits, mode selection, gradient clipping and optimizer
    acceptance. Existing gradients are rejected; no implicit clearing happens.
    check(), if supplied, is the caller's read-only deadline/resource guard.
    """
    prepared = prepare_step(plan, microbatch_records=microbatch_records,
                            group_by_length=group_by_length)
    import torch
    import torch.nn.functional as functional

    parameters = tuple(model.parameters())
    _need(parameters and all(p.grad is None for p in parameters), 'Caller must explicitly clear gradients to None before accumulation.')
    _need(all(torch.isfinite(p).all().item() for p in parameters), 'Nonfinite admitted model parameter.')
    if device is None:
        device = parameters[0].device
    nll_sums, category_sums = [], [0., 0., 0., 0.]
    observed_counts = [0, 0, 0, 0]
    for batch in prepared.microbatches:
        if check is not None:
            check()
        x = torch.tensor(batch.input_ids, dtype=torch.long, device=device)
        y = torch.tensor(batch.targets, dtype=torch.long, device=device)
        logits = model(x)  # ONLY raw token inputs; no plan labels/events/metadata.
        _need(torch.is_tensor(logits) and logits.shape == (*x.shape, VOCAB_SIZE)
              and logits.dtype == torch.float32 and torch.isfinite(logits).all().item(),
              'Finite complete-vocabulary fp32 logits required.')
        losses = functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1),
            reduction='none', ignore_index=IGNORE_INDEX).reshape_as(y)
        loss_sum = losses.sum()
        _need(torch.isfinite(loss_sum).item(), 'Nonfinite masked loss.')
        valid = y != IGNORE_INDEX
        _need(int(valid.sum().item()) == batch.supervised_targets, 'Tensor target accounting changed after collation.')
        ordinary = torch.tensor([kind == 'ordinary' for kind in batch.kinds], dtype=torch.bool, device=device)[:, None]
        selectors = (ordinary & (y >= BYTE_OFFSET), ordinary & (y == EOS),
                     ~ordinary & (y >= BYTE_OFFSET), ~ordinary & (y == EOS))
        for i, selector in enumerate(selectors):
            observed_counts[i] += int(selector.sum().item())
            category_sums[i] += float(losses[selector].detach().sum().item())
        nll_sums.append(float(loss_sum.detach().item()))
        (loss_sum/prepared.global_normalizer).backward()
        _need(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in parameters),
              'Nonfinite accumulated gradient; caller must discard partial gradients.')
    a = prepared.accounting
    _need(observed_counts == [a.ordinary_byte_targets, a.ordinary_source_eos_targets,
                             a.synthetic_answer_byte_targets, a.synthetic_answer_eos_targets],
          'Accumulated tensor categories differ from whole-plan accounting.')
    return {'accounting': a.summary(), 'microbatches': len(prepared.microbatches),
            'loss_sum': math.fsum(nll_sums), 'loss': normalized_sum(nll_sums, prepared.global_normalizer),
            'ordinary_byte_nll_sum': category_sums[0], 'ordinary_chapter_eos_nll_sum': category_sums[1],
            'synthetic_answer_byte_nll_sum': category_sums[2], 'synthetic_answer_eos_nll_sum': category_sums[3],
            'optimizer_step_performed': False, 'gradient_clipping_performed': False,
            'optimizer_acceptance_decided': False}
