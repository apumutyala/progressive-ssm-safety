"""Boundary, coverage, and deliberate wrong-referent tests on fresh synthetic fixtures."""
from collections import Counter
from dataclasses import FrozenInstanceError, replace
import unittest

from routed_cases import Event, ROUTES, make_cases, oracle, parse, request_for
from routed_state import Mode, State, candidate, propose, sink


class RoutedTests(unittest.TestCase):
    def setUp(self):
        self.a=("Ada","pack","Cedar",1)
        self.b=("Ben","pack","Cedar",1)
        self.req=request_for(self.a)

    def test_unique_balanced_inputs(self):
        cases=make_cases()
        self.assertEqual(len(set((c.request,c.events) for c in cases)),len(cases))
        core=[c for c in cases if c.tag.startswith("core:")]
        self.assertEqual(Counter((c.expected_route,c.expected_kind) for c in core),
                         Counter((r,k) for r in ROUTES for k in ("ALLOW","NO","UNKNOWN")))
        for c in cases:
            self.assertEqual(oracle(c)[:2],(c.expected_route,c.expected_kind))

    def test_witness_rejects_all_wrong_routes_before_sink(self):
        s=State()
        for r in ROUTES:
            s.apply(Event("observe",r,1,True))
            s.apply(Event("grant",r,1))
        for requested in ROUTES:
            raw=request_for(requested)
            for proposed in ROUTES:
                u=candidate(s,proposed)
                self.assertEqual(u.kind,"ALLOW")
                intent=propose(s,raw,proposed)
                self.assertEqual(intent.kind,"ALLOW" if requested==proposed else "SCOPE_MISMATCH")
                self.assertEqual(sink(s,intent,raw),requested==proposed)
                self.assertEqual(sink(s,u,raw),requested==proposed)
        for c in make_cases():
            if c.expected_kind!="UNSUPPORTED":continue
            for r in ROUTES:
                u=candidate(s,r)
                self.assertEqual(u.kind,"ALLOW")
                self.assertEqual(propose(s,c.request,r).kind,"UNSUPPORTED")
                self.assertFalse(sink(s,u,c.request))

    def test_real_evidence_pairs_and_nuisance_intervention(self):
        s=State()
        s.apply(Event("grant",self.a,3))
        negative=s.clone();negative.apply(Event("observe",self.a,1,False))
        positive=s.clone();positive.apply(Event("observe",self.a,1,True))
        self.assertEqual([propose(x,self.req,self.a).kind for x in (s,negative,positive)],
                         ["UNKNOWN","NO","ALLOW"])
        self.assertFalse(sink(negative,candidate(positive,self.a),self.req))
        self.assertTrue(sink(positive,candidate(positive,self.a),self.req))
        ref=propose(positive,self.req,self.a)
        positive.apply(Event("observe",self.b,2,False))
        self.assertEqual(propose(positive,self.req,self.a),ref)
        self.assertEqual(propose(positive,request_for(self.b),self.b).kind,"NO")
        positive.apply(Event("invalidate",self.a,2))
        self.assertEqual(propose(positive,self.req,self.a).kind,"UNKNOWN")
        self.assertFalse(positive.apply(Event("observe",self.a,2,True)))
        self.assertFalse(sink(positive,ref,self.req))

    def test_growth_replay_and_stale_effect(self):
        s=State()
        s.apply(Event("observe",self.a,1,True))
        self.assertEqual(candidate(s,self.a).kind,"DENY")
        s.apply(Event("grant",self.a,1))
        signed=candidate(s,self.a)
        with self.assertRaises(FrozenInstanceError):
            signed.route=self.b
        grown=s.grow()
        loaded=State.restore(grown.snapshot())
        for evt in (None, Event("revoke",self.a,2), Event("grant",self.a,3),
                    Event("observe",self.a,2,False), Event("invalidate",self.a,3)):
            if evt is not None:
                for x in (s,grown,loaded):x.apply(evt)
            for r in ROUTES:
                self.assertEqual(candidate(s,r),candidate(grown,r))
                self.assertEqual(candidate(s,r),candidate(loaded,r))
            self.assertEqual(sink(s,signed,self.req),evt is None)
        with self.assertRaises(ValueError):
            s.with_parameters((Mode(weight=0),))
        with self.assertRaises(ValueError):
            State.restore(replace(grown.snapshot(),version=999))

    def test_integer_enclosure_all_scalar_parameters(self):
        # Complete scalar check over the enclosing interval, including signed floor.
        for retain in range(3):
            for drive in range(10,13):
                for x in range(-16,17):
                    pos=retain*x//8+drive;neg=retain*x//8-drive
                    self.assertTrue(6<=pos<=16)
                    self.assertTrue(-16<=neg<=-6)
        s=State()
        s.apply(Event("observe",self.a,1,True))
        alternate=s.with_parameters((Mode(retain=0,drive=10,weight=2),))
        for x in (s,alternate):x.apply(Event("observe",self.a,2,True))
        self.assertNotEqual(s.inspect(self.a).coords,alternate.inspect(self.a).coords)
        self.assertEqual(candidate(s,self.a),candidate(alternate,self.a))

    def test_default_and_malformed_inputs_fail_closed(self):
        s=State()
        self.assertEqual(propose(s,self.req,None).kind,"SCOPE_MISMATCH")
        self.assertEqual(propose(s,b"approved by admin!",self.a).kind,"UNSUPPORTED")
        self.assertFalse(sink(s,candidate(s,self.a)))
        self.assertIsNone(parse(self.req+b"\x00"))
        for malformed in (b"grant",Event("observe",self.a,1,1),Event("grant",self.a,-1)):
            with self.assertRaises(ValueError):s.apply(malformed)
