import copy
from fractions import Fraction
from itertools import product
import unittest
import checker
import intervals as iv
import learner
import model
from runtime import Engine, licensed_interval
import spec
from wire import encode, decode, digest, Reject


def fitted(revised=False):
    raw = learner.acquire(revised)
    return raw, checker.certificate(raw)


def positive_engine(tokens):
    e = Engine(*fitted())
    for token in tokens:
        e.consume(token)
    for subject in (1, 2):
        e.grant(subject, True)
        e.evidence(subject, 1, 1)
    return e


class Qualification(unittest.TestCase):
    def test_all_signed_evidence_contract_cases(self):
        evidence = ((-1, -1), (-1, 0), (-1, 1), (0, 0), (0, 1), (1, 1))
        count = 0
        for state, query, grant, bounds, proof_epoch, epoch, budget in product(
                range(3), spec.QUERIES, (False, True), evidence, range(3), range(3), range(2)):
            subject = spec.referent(state, query)
            status = 1 if bounds[0] > 0 else (2 if bounds[1] < 0 else 0)
            self.assertEqual(licensed_interval(subject, grant, bounds, proof_epoch, epoch, budget),
                             spec.licensed(subject, grant, status, proof_epoch, epoch, budget))
            count += 1
        self.assertEqual(count, 1296)

    def test_coefficients_actually_fitted(self):
        raw, proof = fitted()
        obj = checker.admit(raw, proof)
        self.assertEqual(obj['transitions']['set_ab'][0], [32, 208])
        self.assertEqual(obj['transitions']['set_ba'][0], [32, -208])
        self.assertEqual(obj['transitions']['swap'][0], [-384, 0])
        self.assertEqual(obj['heads']['left']['w'], [256])
        self.assertEqual(obj['heads']['right']['w'], [-256])

    def test_numeric_rounding_matches_exact_fraction(self):
        for numerator in range(-1536, 1537):
            self.assertEqual(iv.rnd(numerator), round(Fraction(numerator, 256)))

    def test_all_quantized_states_parameter_neighbors_errors_enclosed(self):
        raw, _ = fitted()
        obj = decode(raw)
        for state, region in enumerate(iv.REGIONS):
            for token in spec.CONTEXT:
                a, b = obj['transitions'][token][0]
                bound = iv.advance(region, a, b)
                target = spec.step(state, token)
                self.assertTrue(iv.inside(bound, iv.REGIONS[target]))
                for h in range(region[0], region[1] + 1):
                    for aa, bb, error in product(range(a - 1, a + 2), range(b - 1, b + 2), (-1, 0, 1)):
                        point = model.point_step(h, aa, bb, error)
                        self.assertTrue(bound[0] <= point <= bound[1])

    def test_signed_head_bounds_cover_all_grid_points_and_weight_neighbors(self):
        raw, _ = fitted()
        obj = decode(raw)
        for state, region in enumerate(iv.REGIONS):
            for query in spec.QUERIES:
                head = obj['heads'][query]
                bound = iv.readout((region,), head['w'], head['mask'], head['b'])
                self.assertEqual(iv.classify(bound), spec.referent(state, query))
                w, bias = head['w'][0], head['b']
                for h in range(region[0], region[1] + 1):
                    for ww, bb in product(range(w - 1, w + 2), range(bias - 1, bias + 2)):
                        score = model.point_score((h,), (ww,), (1,), bb)
                        self.assertTrue(bound[0] <= score <= bound[1])

    def test_two_differing_histories_have_correct_positive_effects(self):
        first = positive_engine(('set_ab',))
        second = positive_engine(('set_ab', 'swap'))
        self.assertEqual(first.commit(first.request('left'))[1], 1)
        self.assertEqual(second.commit(second.request('left'))[1], 2)

    def test_current_context_not_request_time_freezing(self):
        e = positive_engine(('set_ab',))
        stale = e.request('left')
        e.consume('swap')
        with self.assertRaisesRegex(Reject, 'fresh-issued-binding'):
            e.commit(stale)
        fresh = e.request('left')
        self.assertEqual(decode(fresh)['subject'], 2)
        self.assertEqual(e.commit(fresh)[1], 2)

    def test_same_result_context_edit_still_invalidates_old_source(self):
        e = positive_engine(('set_ab',))
        stale = e.request('left')
        e.consume('swap')
        e.consume('swap')
        with self.assertRaises(Reject):
            e.commit(stale)

    def test_ambiguous_numeric_binding_and_unbound_cannot_dispatch(self):
        self.assertEqual(iv.classify((-256, 256)), 0)
        e = Engine(*fitted())
        for subject in (1, 2):
            e.grant(subject, True)
            e.evidence(subject, 1, 1)
        with self.assertRaisesRegex(Reject, 'ambiguous-referent'):
            e.commit(e.request('left'))

    def test_ambiguous_evidence_and_replacement_and_revocation(self):
        for bounds in ((-1, 1), (0, 1), (-1, -1)):
            e = Engine(*fitted())
            e.consume('set_ab')
            e.grant(1, True)
            e.evidence(1, *bounds)
            with self.assertRaises(Reject):
                e.commit(e.request('left'))
        for mode in ('replace', 'revoke', 'authority'):
            e = positive_engine(('set_ab',))
            p = e.request('left')
            if mode == 'replace':
                e.evidence(1, 1, 1)
            elif mode == 'revoke':
                e.evidence(1, -1, 1)
            else:
                e.grant(1, False)
            with self.assertRaises(Reject):
                e.commit(p)

    def test_rejected_update_does_not_replace_good_source(self):
        raw, proof = fitted()
        e = positive_engine(('set_ab',))
        cache = e.cache()
        obj = decode(raw)
        obj['heads']['left']['w'] = [-256]
        bad = encode(obj)
        change = checker.change_object(raw, bad, 'update-v1')
        with self.assertRaisesRegex(Reject, 'signed-readout-refinement'):
            e.change(bad, proof, proof, change)
        self.assertEqual(e.cache(), cache)
        self.assertEqual(e.commit(e.request('left'))[1], 1)

    def test_update_replays_numeric_state_rejects_old_cache_and_output(self):
        raw, proof = fitted()
        new, new_proof = fitted(True)
        e = positive_engine(('set_ab',))
        stale = e.cache()
        p = e.request('left')
        old_h = decode(stale)['hidden']
        e.change(new, proof, new_proof, checker.change_object(raw, new, 'update-v1'))
        self.assertNotEqual(decode(e.cache())['hidden'], old_h)
        with self.assertRaises(Reject):
            e.restore_cache(stale)
        forged = decode(stale)
        forged['model_sha'] = digest(new)
        forged['generation'] = 1
        with self.assertRaises(Reject):
            e.restore_cache(encode(forged))
        with self.assertRaises(Reject):
            e.commit(p)
        self.assertEqual(e.commit(e.request('left'))[1], 1)

    def test_growth_preserves_point_and_interval_readout_but_invalidates_old_source(self):
        raw, proof = fitted()
        grown = learner.grow(raw)
        gp = checker.certificate(grown)
        old_p, new_p = model.Program.from_admitted(raw), model.Program.from_admitted(grown)
        h, box = old_p.initial()
        hh, bb = new_p.initial()
        for token in ('set_ab', 'swap', 'swap', 'set_ba'):
            h, box = old_p.step(h, box, token, (0,))
            hh, bb = new_p.step(hh, bb, token, (0, 0))
            for query in spec.QUERIES:
                self.assertEqual(old_p.select(h, box, query), new_p.select(hh, bb, query))
        e = positive_engine(('set_ab', 'swap'))
        stale, proposal = e.cache(), e.request('left')
        e.change(grown, proof, gp, checker.change_object(raw, grown, 'growth-v1'))
        with self.assertRaises(Reject):
            e.restore_cache(stale)
        with self.assertRaises(Reject):
            e.commit(proposal)
        self.assertEqual(e.commit(e.request('left'))[1], 2)

    def test_postgrowth_weights_can_change_scores_without_changing_referent(self):
        raw, proof = fitted()
        grown = learner.grow(raw)
        gp = checker.certificate(grown)
        active = learner.activate_aux(grown)
        ap = checker.certificate(active)
        e = positive_engine(('set_ab',))
        e.change(grown, proof, gp, checker.change_object(raw, grown, 'growth-v1'))
        old_bounds = decode(e.request('left'))['score_interval']
        old_point = e._program.select(e._hidden, e._box, 'left')[1]
        e.change(active, gp, ap, checker.change_object(grown, active, 'update-v1'))
        new_bounds = decode(e.request('left'))['score_interval']
        new_point = e._program.select(e._hidden, e._box, 'left')[1]
        self.assertNotEqual(old_bounds, new_bounds)
        self.assertNotEqual(old_point, new_point)
        self.assertEqual(e.commit(e.request('left'))[1], 1)

    def test_certificate_binding_complete_and_no_unsupported_paths(self):
        raw, proof = fitted()
        obj = decode(proof)
        obj['checks'].pop()
        with self.assertRaises(Reject):
            checker.admit(raw, encode(obj))
        obj = decode(raw)
        obj['residual'] = 1
        with self.assertRaises(Reject):
            checker.certificate(encode(obj))
        obj = decode(raw)
        obj['transitions']['swap'][0][0] = -128  # shrinking forever loses robust region
        with self.assertRaisesRegex(Reject, 'transition-region-refinement'):
            checker.certificate(encode(obj))

    def test_noise_histories_stay_inside_bounds(self):
        raw, proof = fitted()
        for errors in product((-1, 0, 1), repeat=4):
            e = Engine(raw, proof)
            for token, error in zip(('set_ab', 'swap', 'swap', 'swap'), errors):
                e.consume(token, (error,))
            self.assertEqual(decode(e.request('left'))['subject'], 2)

    def test_overflow_fault_and_bad_context_cannot_keep_positive(self):
        e = positive_engine(('set_ab',))
        e.evidence(1, 1, 1)
        p = e.request('left')
        with self.assertRaises(Reject):
            e.evidence(1, -1, 1)
        with self.assertRaises(Reject):
            e.commit(p)
        f = positive_engine(('set_ab',))
        p = f.request('left')
        with self.assertRaises(Reject):
            f.consume('unmodeled-context')
        with self.assertRaises(Reject):
            f.commit(p)

    def test_without_certificate_wrong_numeric_binder_passes_generic_guard(self):
        raw, proof = fitted()
        obj = decode(raw)
        obj['heads']['left']['w'] = [-256]
        bad = encode(obj)
        with self.assertRaises(Reject):
            Engine(bad, proof)
        mutant = object.__new__(Engine)
        mutant._initialize_unchecked(bad)  # explicit TCB/admission bypass mutant
        mutant.consume('set_ab')
        for subject in (1, 2):
            mutant.grant(subject, True)
            mutant.evidence(subject, 1, 1)
        self.assertEqual(mutant.commit(mutant.request('left'))[1], 2)
        self.assertEqual(spec.referent(1, 'left'), 1)

    def test_shared_weight_alias_and_cache_cannot_restore_authority(self):
        raw, proof = fitted()
        producer = decode(raw)
        e = Engine(encode(producer), proof)
        producer['heads']['left']['w'][0] = -256
        e.consume('set_ab')
        e.grant(1, True)
        e.evidence(1, 1, 1)
        cache = e.cache()
        proposal = e.request('left')
        self.assertEqual(decode(proposal)['subject'], 1)
        e.grant(1, False)
        e.restore_cache(cache)
        with self.assertRaises(Reject):
            e.commit(proposal)

    def test_signed_correlation_is_not_replaced_by_absolute_magnitude(self):
        raw, _ = fitted()
        program = model.Program.from_admitted(raw)
        h, box = program.initial()
        h, box = program.step(h, box, 'set_ab', (0,))
        left, _, lb = program.select(h, box, 'left')
        right, _, rb = program.select(h, box, 'right')
        self.assertEqual((left, right), (1, 2))
        self.assertGreater(lb[0], 64)
        self.assertLess(rb[1], -64)
