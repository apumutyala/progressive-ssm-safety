"""Stdlib panel/gate tests; no numerical model or final-case access."""
import copy
import sys
import unittest

_BEFORE = set(sys.modules)
import pilot_evaluation as pilot
_IMPORTED = set(sys.modules)-_BEFORE


def fake_rows(*, correct=True, priors=(3., -1., 5.), signal=0.):
    result = []
    for case in pilot.development_panel():
        label = case['expected']['label']; chosen = label if correct else 'unknown'
        text = pilot._reference(chosen, case['expected']['entity'], case['expected']['action'])
        ids = pilot._encode(text)+[3]
        logits = [v+(signal if label == pilot.LABELS[i] else 0) for i, v in enumerate(priors)]
        scored = pilot.score_raw_response(case, ids, capped=False)
        result.append({'id': case['id'], 'raw_u16le_b64': pilot._u16(ids), 'raw_tokens': len(ids),
            **scored, 'eos': True, 'capped': False,
            'first_candidate_logits': logits})
    return result


def fake_result(correct):
    rows = fake_rows(correct=correct, signal=2. if correct else 0.)
    summary, influence = pilot.summarize(rows)
    authority = {key: {'summary': {'ordering_errors': 0, 'unsupported_raw_confirms': 0,
        'supported_positions': 1, 'supported_raw_confirms': 1}} for key in ('fitted', 'adverse_copy')}
    return {'raw_rows': rows, 'summary': summary, 'influence': influence, 'authority': authority}


class PilotEvaluationTests(unittest.TestCase):
    def test_stdlib_only_import(self):
        self.assertNotIn('torch', _IMPORTED)
        self.assertNotIn('corpus_model', _IMPORTED)

    def test_public_factorial_and_independent_actual_prompt_oracle(self):
        cases = pilot.development_panel()
        self.assertEqual(len(cases), 24)
        self.assertEqual(len({r['prompt'] for r in cases}), 24)
        for dossier in range(4):
            selected = [c for c in cases if c['dossier'] == dossier]
            for context in (0, 1):
                self.assertEqual({c['expected']['label'] for c in selected if c['context'] == context}, set(pilot.LABELS))
            for case in selected: self.assertEqual(pilot.oracle(case), case['expected'])

    def test_irrelevant_edit_only_changes_other_record(self):
        cases = pilot.development_panel()
        for dossier in range(4):
            for label in pilot.LABELS:
                left, right = [c for c in cases if c['dossier'] == dossier and c['expected']['label'] == label]
                self.assertEqual(left['expected'], right['expected'])
                import re
                match = [m for m in re.finditer(pilot._PATTERNS[left['world']], left['prompt']) if m['name'] == left['other_entity']][-1]
                a, b = match.span('state')
                self.assertEqual(left['prompt'][:a]+right['prompt'][a:b]+left['prompt'][b:], right['prompt'])

    def test_relevant_edits_only_change_latest_target_fact(self):
        import re
        cases = pilot.development_panel()
        for dossier in range(4):
            for context in (0, 1):
                selected = [c for c in cases if c['dossier'] == dossier and c['context'] == context]
                first = selected[0]
                match = [m for m in re.finditer(pilot._PATTERNS[first['world']], first['prompt']) if m['name'] == first['expected']['entity']][-1]
                a, b = match.span('state')
                for other in selected[1:]:
                    self.assertEqual(first['prompt'][:a]+other['prompt'][a:b]+first['prompt'][b:], other['prompt'])

    def test_canonical_refs_and_length_matched_entity_action_foils(self):
        from response_eval import score_case
        for case in pilot.development_panel():
            self.assertTrue(score_case(case, pilot._encode(case['reference'])+[3])['content_correct'])
            for foil in pilot.foils(case):
                self.assertFalse(score_case(case, pilot._encode(foil['reference'])+[3])['content_correct'])
                if foil['kind'] in ('wrong_entity', 'wrong_action'): self.assertTrue(foil['same_byte_length'])

    def test_cross_fact_contrast_cancels_fixed_answer_priors(self):
        _, zero = pilot.summarize(fake_rows(priors=(50., -20., 7.), signal=0.))
        self.assertEqual(zero['mean_cross_fact_contrast'], 0.)
        _, influenced = pilot.summarize(fake_rows(priors=(50., -20., 7.), signal=2.))
        self.assertEqual(influenced['mean_cross_fact_contrast'], 2.)
        self.assertEqual(len(influenced['cross_fact_contrasts']), 24)

    def test_gate_requires_grounded_triplets_eos_and_matched_initials(self):
        good, null = fake_result(True), fake_result(False)
        self.assertEqual(good['summary']['dossiers_all_six_content_correct'], 4)
        self.assertEqual(null['summary']['dossiers_all_six_content_correct'], 0)
        self.assertFalse(pilot.compare_treatments(good, null)['directional_replication_trigger'])
        gate = pilot.compare_treatments(good, null, initial_true=null, initial_shuffled=copy.deepcopy(null))
        self.assertTrue(gate['directional_replication_trigger'])
        self.assertFalse(pilot.compare_treatments(good, null, initial_true=good, initial_shuffled=copy.deepcopy(good))['directional_replication_trigger'])
        mismatch = copy.deepcopy(null); mismatch['raw_rows'][0]['first_candidate_logits'][0] += 1
        self.assertFalse(pilot.compare_treatments(good, null, initial_true=null, initial_shuffled=mismatch)['directional_replication_trigger'])
        no_eos = copy.deepcopy(good)
        no_eos['summary']['dossiers_all_six_correct_and_eos'] = 2
        self.assertFalse(pilot.compare_treatments(no_eos, null, initial_true=null, initial_shuffled=null)['directional_replication_trigger'])
        invalid_initial = copy.deepcopy(null)
        invalid_initial['authority']['fitted']['summary']['ordering_errors'] = 1
        self.assertFalse(pilot.compare_treatments(good, null, initial_true=invalid_initial, initial_shuffled=invalid_initial)['directional_replication_trigger'])

    def test_authority_includes_positive_grant_and_revocation_all_positions(self):
        from authority_cases import build_cases, oracle
        cases = build_cases()
        self.assertEqual(len(cases), 56)
        self.assertEqual(sum(len(c.ids) for c in cases), 7700)
        self.assertTrue(any(any(oracle(c)) for c in cases))
        for case in cases:
            truth = oracle(case)
            if case.mode == 'grant_late': self.assertFalse(truth[-2]); self.assertTrue(truth[-1])
            if case.mode == 'revoked': self.assertTrue(truth[-2]); self.assertFalse(truth[-1])
        result = fake_result(True); result['authority']['adverse_copy']['summary']['unsupported_raw_confirms'] = 1
        self.assertFalse(pilot._absolute(result))

    def test_missing_case_rejected_and_public_admission_sizes(self):
        with self.assertRaises(ValueError): pilot.summarize(fake_rows()[:-1])
        manifest = pilot.admission_manifest()
        self.assertLessEqual(manifest['max_prompt_bytes'], 512)
        self.assertEqual(manifest['authority_positions_per_audit']*2*10, 154000)
        self.assertTrue(manifest['development_only'])

    def test_no_query_action_echo_and_exact_vs_semantic_score(self):
        for case in pilot.development_panel():
            question = case['prompt'].split('\n')[-1]
            self.assertNotIn('pack', question); self.assertNotIn('sort', question)
            text = case['reference']
            score = pilot.score_raw_response(case, pilot._encode(text), capped=True)
            self.assertTrue(score['exact_content']); self.assertFalse(score['exact_content_and_eos'])
            score = pilot.score_raw_response(case, pilot._encode(text)+[3], capped=False)
            self.assertTrue(score['exact_content_and_eos'])

    def test_worst_legal_raw_wire_does_not_expand_renderings(self):
        rows = []
        for case in pilot.development_panel():
            raw = [6]*128
            row = {**case, 'raw_u16le_b64': pilot._u16(raw), 'raw_tokens': 128,
                'eos': False, 'capped': True, **pilot.score_raw_response(case, raw, capped=True),
                'first_id': 6, 'gold_first_id': 97, 'gold_first_rank': 264,
                'gold_first_probability': .000012345678901234, 'gold_first_logit': -12.34567890123456,
                'gold_first_log_odds': -25.34567890123456, 'first_candidate_logits': [-12.34567890123456]*3,
                'reference_likelihood': {'tokens_including_eos': 99, 'full_log_likelihood': -1234.567890123456,
                    'mean_log_likelihood': -12.34567890123456, 'same_byte_length': True},
                'foil_likelihoods': [{'kind': f['kind'], 'tokens_including_eos': 99,
                    'full_log_likelihood': -1234.567890123456, 'mean_log_likelihood': -12.34567890123456,
                    'same_byte_length': f['same_byte_length']} for f in pilot.foils(case)]}
            self.assertNotIn('raw_text', row)
            self.assertEqual(pilot.render_raw(row), '<query_a>'*128)
            rows.append(row)
        self.assertLessEqual(len(pilot.canonical(rows)), pilot.SEMANTIC_RAW_CAP)

    def test_query_only_pairs_and_assignment_counterbalancing(self):
        pairs = pilot.query_only_pairs()
        self.assertEqual(len(pairs), 8)
        self.assertEqual(sum(a['expected']['label'] != b['expected']['label'] for a, b in pairs), 4)
        for a, b in pairs:
            self.assertEqual(a['prompt'].rsplit('\n', 1)[0], b['prompt'].rsplit('\n', 1)[0])
            self.assertNotEqual(a['expected']['entity'], b['expected']['entity'])
            self.assertNotEqual(a['expected']['action'], b['expected']['action'])
        cases = pilot.development_panel()
        for entity in ('Arlo', 'Bela'):
            self.assertEqual({c['expected']['action'] for c in cases if c['expected']['entity'] == entity}, {'pack', 'sort'})


if __name__ == '__main__': unittest.main()
