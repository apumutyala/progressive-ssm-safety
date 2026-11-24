import itertools
import json
import threading
import unittest
from dataclasses import replace
from routed_cases import Event, request_for
from routed_state import State as LegacyState, candidate as legacy_candidate
from transaction import AtomicHost, Reject, bookkeeping_invariant, commit_step, event_step

A = ('Ada', 'pack', 'Cedar', 1)
B = ('Ben', 'sort', 'Birch', 2)
WITNESSES = {}


def ready(domain='atomic-one', budget=4):
    h = AtomicHost(domain, budget)
    for route in (A, B):
        h.apply(Event('observe', route, 1, True))
        h.apply(Event('grant', route, 1))
    return h


class TransactionTests(unittest.TestCase):
    def test_useful_positive_commits_two_distinct_referents(self):
        h = ready()
        receipts = []
        for route in (A, B):
            request = request_for(route)
            raw = h.issue(request, route)
            r = h.commit(raw, request)
            self.assertEqual(r.route, route)
            self.assertEqual(r.budget_after, r.budget_before-1)
            receipts.append({'nonce': r.nonce, 'route': list(r.route), 'intent': raw.decode('ascii')})
        self.assertEqual(h.inspect().budget, 2)
        self.assertTrue(bookkeeping_invariant(h.inspect()))
        WITNESSES['positive_commits'] = receipts

    def test_restored_learned_cache_never_restores_current_authority(self):
        h = ready()
        legacy = LegacyState()
        legacy.apply(Event('observe', A, 1, True))
        legacy.apply(Event('grant', A, 1))
        cached = legacy.snapshot()
        raw = h.issue(request_for(A), A)
        h.apply(Event('revoke', A, 2))
        old = LegacyState.restore(cached)
        proposed = legacy_candidate(old, A)
        self.assertEqual(proposed.kind, 'ALLOW')
        protected = h.inspect()
        with self.assertRaisesRegex(Reject, 'current-frontier'):
            h.commit(raw, request_for(A))
        with self.assertRaisesRegex(Reject, 'current-license'):
            h.issue(request_for(A), proposed.route)
        self.assertEqual(protected, h.inspect())
        WITNESSES['restored_cache'] = 'Legacy ALLOW remains only a route proposal; current revoked host refuses issue and commit'

    def test_revoke_regrant_prevents_old_intent_ABA_revival(self):
        h = ready()
        request = request_for(A)
        old = h.issue(request, A)
        h.apply(Event('revoke', A, 2))
        h.apply(Event('grant', A, 3))
        with self.assertRaises(Reject):
            h.commit(old, request)
        new = h.issue(request, A)
        self.assertNotEqual(old, new)
        self.assertEqual(h.commit(new, request).row.grant_revision, 3)

    def test_exact_bytes_duplicate_forgery_and_actual_request_binding(self):
        h = ready()
        request = request_for(A)
        raw = h.issue(request, A)
        forged = json.loads(raw)
        forged['cost'] = 0
        altered = json.dumps(forged, sort_keys=True, separators=(',', ':')).encode('ascii')
        for bad in (altered, raw+b' ', bytearray(raw)):
            with self.assertRaises(Reject):
                h.commit(bad, request)
        with self.assertRaises(Reject):
            h.commit(raw, request_for(B))
        h.commit(raw, request)
        state = h.inspect()
        with self.assertRaisesRegex(Reject, 'already-consumed'):
            h.commit(raw, request)
        self.assertEqual(h.inspect(), state)

    def test_all_six_commit_commit_revoke_serial_interleavings(self):
        summaries = []
        for order in itertools.permutations(('commit1', 'commit2', 'revoke')):
            h = ready()
            raw = h.issue(request_for(A), A)
            outcomes = []
            for op in order:
                if op == 'revoke':
                    h.apply(Event('revoke', A, 2))
                    outcomes.append('revoked')
                else:
                    try:
                        h.commit(raw, request_for(A))
                        outcomes.append('committed')
                    except Reject:
                        outcomes.append('rejected')
                self.assertTrue(bookkeeping_invariant(h.inspect()))
            expected = 0 if order[0] == 'revoke' else 1
            self.assertEqual(outcomes.count('committed'), expected)
            summaries.append([list(order), outcomes])
        WITNESSES['all_six_serial_orders'] = summaries

    def test_two_actual_threads_cannot_consume_same_intent_twice(self):
        h = ready()
        raw = h.issue(request_for(A), A)
        barrier = threading.Barrier(3)
        outcomes = []
        def attempt():
            barrier.wait(timeout=2)
            try:
                h.commit(raw, request_for(A))
                outcomes.append('committed')
            except Reject:
                outcomes.append('rejected')
        workers = [threading.Thread(target=attempt) for _ in range(2)]
        for worker in workers:
            worker.start()
        barrier.wait(timeout=2)
        for worker in workers:
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
        self.assertCountEqual(outcomes, ['committed', 'rejected'])
        self.assertTrue(bookkeeping_invariant(h.inspect()))
        WITNESSES['two_thread_duplicate'] = sorted(outcomes)

    def test_split_check_then_use_counterexample_is_not_atomic_commit(self):
        h = ready()
        raw = h.issue(request_for(A), A)
        _, prechecked = commit_step(h.inspect(), raw, request_for(A))
        h.apply(Event('revoke', A, 2))
        external_simulated_sends = [prechecked.route]  # mutant acts on stale approval
        self.assertEqual(external_simulated_sends, [A])
        with self.assertRaises(Reject):
            h.commit(raw, request_for(A))
        WITNESSES['split_check_counterexample'] = 'A saved approval can be used after revocation; no real external send was made'

    def test_model_update_or_growth_revision_invalidates_pending(self):
        h = ready()
        raw = h.issue(request_for(A), A)
        h.revise(1)
        with self.assertRaisesRegex(Reject, 'current-frontier'):
            h.commit(raw, request_for(A))
        self.assertEqual(h.commit(h.issue(request_for(A), A), request_for(A)).model_revision, 1)

    def test_wrong_route_unknown_negative_and_missing_grant_cannot_issue(self):
        h = AtomicHost('negative')
        with self.assertRaises(Reject):
            h.issue(request_for(A), A)
        h.apply(Event('observe', A, 1, False))
        h.apply(Event('grant', A, 1))
        with self.assertRaises(Reject):
            h.issue(request_for(A), A)
        h.apply(Event('observe', A, 2, True))
        with self.assertRaisesRegex(Reject, 'request-route'):
            h.issue(request_for(B), A)
        h.apply(Event('revoke', A, 2))
        with self.assertRaises(Reject):
            h.issue(request_for(A), A)

    def test_stale_event_noop_and_unrelated_new_event_conservative_invalidation(self):
        h = ready()
        raw = h.issue(request_for(A), A)
        state = h.inspect()
        self.assertFalse(h.apply(Event('revoke', A, 1)))
        self.assertEqual(h.inspect(), state)
        h.apply(Event('observe', B, 2, True))
        with self.assertRaises(Reject):
            h.commit(raw, request_for(A))

    def test_caps_and_budget_failure_do_not_mutate_authority_or_refund(self):
        h = ready(budget=1)
        one = h.issue(request_for(A), A)
        two = h.issue(request_for(B), B)
        h.commit(one, request_for(A))
        exhausted = h.inspect()
        with self.assertRaisesRegex(Reject, 'budget-exhausted'):
            h.commit(two, request_for(B))
        self.assertEqual(exhausted, h.inspect())
        s = AtomicHost('cap')
        for revision in range(1, 65):
            s.apply(Event('grant', A, revision))
        full = s.inspect()
        with self.assertRaisesRegex(Reject, 'event-cap'):
            s.apply(Event('revoke', A, 65))
        self.assertEqual(s.inspect(), replace(full, halted=True))
        with self.assertRaisesRegex(Reject, 'halted'):
            s.issue(request_for(A), A)

    def test_event_cap_fault_invalidates_already_issued_intent(self):
        h = ready()
        for revision in range(2, 62):
            h.apply(Event('observe', B, revision, True))
        self.assertEqual(h.inspect().accepted_events, 64)
        raw = h.issue(request_for(A), A)
        with self.assertRaisesRegex(Reject, 'event-cap'):
            h.apply(Event('revoke', A, 2))
        with self.assertRaisesRegex(Reject, 'halted'):
            h.commit(raw, request_for(A))
        self.assertEqual(h.inspect().consumed, ())

    def test_cross_domain_rejects_but_restart_same_domain_needs_durability(self):
        first, other = ready('first'), ready('other')
        raw = first.issue(request_for(A), A)
        other.issue(request_for(A), A)
        with self.assertRaisesRegex(Reject, 'not-exact-issued'):
            other.commit(raw, request_for(A))
        first.commit(raw, request_for(A))
        restarted = ready('first')
        self.assertEqual(raw, restarted.issue(request_for(A), A))
        restarted.commit(raw, request_for(A))
        WITNESSES['restart_limit'] = 'Recreating same domain/frontier/nonce permits reacceptance: no crash-durable/global exactly-once claim'
