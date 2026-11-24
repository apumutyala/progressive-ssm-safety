"""Finite interface checks. Never import Torch, shared models, or a checkpoint."""
import ast
import hashlib
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'REFERENCE_SOURCE'))
from routed_cases import Event, request_for
from routed_state import State, candidate, propose, sink

OBSERVATIONS = {}
A = ('Ada', 'pack', 'Cedar', 1)
B = ('Ben', 'pack', 'Cedar', 1)


def positive():
    s = State()
    for r in (A, B):
        s.apply(Event('observe', r, 1, True))
        s.apply(Event('grant', r, 1))
    return s


def functions(filename):
    tree = ast.parse((ROOT / 'REFERENCE_SOURCE' / filename).read_bytes())
    return {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


class MappingTests(unittest.TestCase):
    def test_exact_manifest_and_owner_commit_file_distinction(self):
        m = json.loads((ROOT / 'BINDING_MANIFEST.json').read_bytes())
        self.assertEqual(m['base_owner_attestation']['commit'], 'f1ccb1f5aec8a2199b28e9dcc29889826287f3c7')
        self.assertEqual(m['base_owner_attestation']['tree'], '15505a99571df7fbe7d6fbd0a7fe34494ffb57bc')
        self.assertEqual(m['reference_pins']['base_qualification_model.py'],
                         'a1b41c310451617b613ff8dd606db8900f4e4ffb5f4d703f2fb2f0958185fe6f')
        for name, pin in m['reference_pins'].items():
            self.assertEqual(hashlib.sha256((ROOT / 'REFERENCE_SOURCE' / name).read_bytes()).hexdigest(), pin)

    def test_source_sites_separate_token_latch_from_learned_route_selector(self):
        base = functions('base_qualification_model.py')
        calls = [ast.unparse(n.func) for n in ast.walk(base['_run']) if isinstance(n, ast.Call)]
        self.assertIn('IntervalModel._queries', calls)
        neural = functions('routed_neural.py')
        selector_calls = [ast.unparse(n.func) for n in ast.walk(neural['selector']) if isinstance(n, ast.Call)]
        self.assertIn('self.scope', selector_calls)
        self.assertNotIn('parse', selector_calls)
        self.assertNotIn('oracle', selector_calls)
        run_calls = [ast.unparse(n.func) for n in ast.walk(neural['run_arm']) if isinstance(n, ast.Call)]
        self.assertIn('logits.argmax', run_calls)
        OBSERVATIONS['static_check_limit'] = 'Pinned syntax/dependency sites only; not general no-leak proof or neural execution'

    def test_prior_combined_witness_preserved_and_identifier_block_resolved(self):
        m = json.loads((ROOT / 'BINDING_MANIFEST.json').read_bytes())
        old = ROOT.parent / 'byte_query'
        for name, pin in m['prior_witness_pins'].items():
            self.assertEqual(hashlib.sha256((old / name).read_bytes()).hexdigest(), pin)
        self.assertEqual(m['historical_reported_status'], 'ISOLATED_PASS_REFERENCE_BINDING_BLOCKED')
        w = json.loads((old / 'prior_byte_witness.json').read_bytes())['strong_combined']
        self.assertEqual(w['fresh_after_growth']['reserved_query_id'], 7)
        self.assertEqual(w['renewed_release'], ['scope-growth', 'left', 2, 'YES', 29, 6])
        OBSERVATIONS['historical_reported_status'] = m['historical_reported_status']
        OBSERVATIONS['resolution'] = 'Commit identifier clarified; file hash unchanged. No neural/native parity established.'

    def test_wrong_learned_route_can_have_valid_evidence_but_witness_rejects(self):
        s = positive()
        request = request_for(A)
        raw = candidate(s, B)
        actual = propose(s, request, B)
        self.assertEqual(raw.kind, 'ALLOW')
        self.assertEqual(actual.kind, 'SCOPE_MISMATCH')
        self.assertFalse(sink(s, raw, request))
        self.assertFalse(sink(s, actual, request))
        good = propose(s, request, A)
        self.assertEqual(good.kind, 'ALLOW')
        self.assertTrue(sink(s, good, request))
        OBSERVATIONS['wrong_route'] = {'raw_candidate': raw.kind, 'pre_sink_AST': actual.kind,
                                      'repaired_route': actual.route, 'correct_positive': good.kind}

    def test_current_revocation_rejects_but_untrusted_old_snapshot_can_resurrect(self):
        s = positive()
        request = request_for(A)
        old = s.snapshot()
        intent = propose(s, request, A)
        s.apply(Event('revoke', A, 2))
        self.assertFalse(sink(s, intent, request))
        restored_old = State.restore(old)
        self.assertTrue(sink(restored_old, intent, request))
        OBSERVATIONS['cache_counterexample'] = {
            'current_state_accepts_old_intent': False, 'restored_old_snapshot_accepts': True,
            'meaning': 'Schema/enclosure validation does not authenticate latest lineage; external current state remains necessary'}

    def test_sink_acceptance_is_not_one_use_or_real_execution(self):
        s = positive()
        request = request_for(B)
        intent = propose(s, request, B)
        self.assertTrue(sink(s, intent, request))
        self.assertTrue(sink(s, intent, request))
        OBSERVATIONS['sink_repeat'] = 'Same intent accepted twice; no nonce consumption or actuator effect is modeled'

    def test_unknown_negative_denial_and_unsupported_remain_distinct(self):
        s = State()
        request = request_for(A)
        self.assertEqual(propose(s, request, A).kind, 'UNKNOWN')
        s.apply(Event('observe', A, 1, False))
        self.assertEqual(propose(s, request, A).kind, 'NO')
        s.apply(Event('observe', A, 2, True))
        self.assertEqual(propose(s, request, A).kind, 'DENY')
        s.apply(Event('grant', A, 1))
        self.assertTrue(sink(s, propose(s, request, A), request))
        self.assertEqual(propose(s, b'please do anything', A).kind, 'UNSUPPORTED')
