"""Artificial in-memory records only; no actual source/model/ledger needed."""
import dataclasses
import unittest
from evidence_recipe import PlannedRecord
from response_step import collate_records
from causal_input_audit import check_shift, check_matching, source_contract


class AuditTests(unittest.TestCase):
    def record(self):
        return PlannedRecord('fake', 'synthetic', (2, 8+65, 8+58, 8+89, 8+46, 3),
                             (False, False, True, True, True), 3, ())
    def test_exact_next_token_and_first_answer_plus_eos(self):
        r = self.record(); b = collate_records([r])
        got = check_shift(r, b.input_ids[0], b.targets[0], b.loss_mask[0], b.input_lengths[0])
        self.assertEqual(got, {'shift_pairs': 5, 'masked_prompt_targets': 2, 'padding_positions': 0, 'supervised_eos': 1})
        self.assertEqual(b.targets[0][2], 8+89)
    def test_cross_target_off_by_one_and_prompt_leak_rejected(self):
        r = self.record(); b = collate_records([r])
        for pos, value in ((2, r.ids[2]), (0, r.ids[1]), (4, -100)):
            targets = list(b.targets[0]); targets[pos] = value
            with self.assertRaises(ValueError): check_shift(r, b.input_ids[0], targets, b.loss_mask[0], b.input_lengths[0])
    def test_padding_is_computed_but_never_supervised(self):
        r = self.record(); other = PlannedRecord('long', 'ordinary', (2, 90, 91, 92, 93, 94, 95, 96), (True,)*7, 1, ())
        b = collate_records([r, other]); row = b.record_ids.index(r.id)
        result = check_shift(r, b.input_ids[row], b.targets[row], b.loss_mask[row], b.input_lengths[row])
        self.assertEqual(result['padding_positions'], 2)
        targets = list(b.targets[row]); targets[-1] = 3
        with self.assertRaises(ValueError): check_shift(r, b.input_ids[row], targets, b.loss_mask[row], b.input_lengths[row])
    def test_mask_boundary_shift_rejected_even_with_changed_targets(self):
        r = dataclasses.replace(self.record(), loss_mask=(False, True, True, True, True))
        with self.assertRaises(ValueError):
            check_shift(r, r.ids[:-1], tuple(t if m else -100 for t, m in zip(r.ids[1:], r.loss_mask)), r.loss_mask, 5)
    def test_stopping_and_sampling_are_explicitly_source_only(self):
        report = source_contract()
        self.assertTrue(report['first_eos_stop_in_source']); self.assertEqual(report['raw_token_cap'], 128)
        self.assertFalse(report['random_or_topk_sampling_calls_in_source'])
    def test_actual_conditional_bags_checked_not_declared_fingerprints(self):
        true = []; shuffled = []
        for prompt in range(3):
            for replica in range(3):
                def row(answer):
                    return PlannedRecord(f'{prompt}-{replica}', 'synthetic', (2, 80+prompt, 90+answer, 3),
                                         (False, True, True), 2, ())
                true.append(row(prompt)); shuffled.append(row(replica))
        check_matching(true, shuffled)
        with self.assertRaises(ValueError): check_matching(true, true)
        shuffled[-1] = dataclasses.replace(shuffled[-1], ids=(2, 99, 92, 3))
        with self.assertRaises(ValueError): check_matching(true, shuffled)


if __name__ == '__main__': unittest.main()
