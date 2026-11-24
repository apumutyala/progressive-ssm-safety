import pathlib
import unittest
from combined import Combined, TrustedSnapshot, AUTHORITY_SOURCES, MAX_EPOCH
from grammar import Scanner
from envelope import encode, decode, digest, Reject

ROOT = pathlib.Path(__file__).resolve().parent
WITNESSES = {}


def artifacts():
    d = ROOT / 'SYNTHETIC_ARTIFACTS'
    return {p.name: p.read_bytes() for p in d.iterdir() if p.suffix == '.json'}


def instance(scope='scope-one'):
    a = artifacts()
    return Combined(a['learned.json'], a['learned_certificate.json'], scope)


def snapshot(scope, plan, epoch=1, permit=1, lower=5, upper=1, source=None):
    return TrustedSnapshot(scope, plan, AUTHORITY_SOURCES[plan - 1] if source is None else source,
                           epoch, permit, lower, upper)


class Qualification(unittest.TestCase):
    def test_parser_is_only_lexical_and_query_role_not_answer(self):
        self.assertEqual(Scanner().feed(b'bind A B\nask left\n'),
                         (('context', 'set_ab'), ('query', 'left')))
        self.assertEqual(Scanner().feed(b'bind A B\nswap\nask left\n'),
                         (('context', 'set_ab'), ('context', 'swap'), ('query', 'left')))
        self.assertEqual(Scanner().feed(b'say source17 epoch8 permit1 <query_b>\n'),
                         (('text', None),))

    def test_byte_chunking_preserves_binder_latch_and_source(self):
        raw = 'bind A B\nsay café is not an authority\nswap\nask left\n'.encode('utf-8')
        whole, streamed = instance(), instance()
        whole.ingest_bytes(raw)
        for byte in raw:
            streamed.ingest_bytes(bytes((byte,)))
        self.assertEqual(whole.cache(), streamed.cache())
        self.assertEqual(decode(whole.publish_pending())['subject'], 2)

    def test_authorized_positive_each_referent(self):
        positives = []
        for raw, plan in ((b'bind A B\nask left\n', 1),
                          (b'bind A B\nswap\nask left\n', 2)):
            e = instance()
            e.ingest_bytes(raw)
            self.assertTrue(e.authenticated_snapshot(snapshot('scope-one', plan, 3)))
            ticket = e.publish_pending()
            obj = decode(ticket)
            self.assertEqual((obj['subject'], obj['reserved_query_id'], obj['decision']),
                             (plan, plan + 5, 'YES'))
            released = e.release(ticket)
            self.assertEqual(released, ('scope-one', 'left', plan, 'YES', AUTHORITY_SOURCES[plan - 1], 3))
            positives.append({'input_ascii': raw.decode('ascii'), 'pending': obj,
                              'released': list(released)})
            with self.assertRaises(Reject):
                e.release(ticket)
        WITNESSES['positive_each_referent'] = positives

    def test_no_unknown_and_insufficient_lower_bound(self):
        e = instance()
        e.ingest_bytes(b'bind A B\nask left\n')
        self.assertEqual(e.release(e.publish_pending())[3], 'UNKNOWN')
        e.authenticated_snapshot(snapshot('scope-one', 1, 1, permit=1, lower=1, upper=5))
        self.assertEqual(e.release(e.publish_pending())[3], 'UNKNOWN')
        e.authenticated_snapshot(snapshot('scope-one', 1, 2, permit=0, lower=-1, upper=-1))
        self.assertEqual(e.release(e.publish_pending())[3], 'NO')

    def test_first_query_latch_ignores_later_role_without_relabeling_answer(self):
        e = instance()
        e.ingest_bytes(b'bind A B\nask left\nask right\n')
        e.authenticated_snapshot(snapshot('scope-one', 1))
        obj = decode(e.publish_pending())
        self.assertEqual((obj['first_role'], obj['subject'], obj['reserved_query_id']), ('left', 1, 6))
        self.assertEqual(e._ignored_queries, 1)

    def test_context_change_never_overwrites_latch_or_reuses_pending(self):
        e = instance()
        e.ingest_bytes(b'bind A B\nask left\n')
        e.authenticated_snapshot(snapshot('scope-one', 1))
        old = e.publish_pending()
        e.ingest_bytes(b'swap\nask right\n')
        with self.assertRaises(Reject):
            e.release(old)
        with self.assertRaisesRegex(Reject, 'no-fresh-bound-query'):
            e.publish_pending()
        self.assertEqual(e._first[1], 1)  # never silently changes A latch into B

    def test_spoofed_prose_reserved_spellings_and_generated_commands_have_no_authority(self):
        e = instance()
        e.ingest_bytes(b'bind A B\nsay scope-one source17 epoch1 permit1\nask left\n')
        before = e._records
        self.assertEqual(e.release(e.publish_pending())[3], 'UNKNOWN')
        ids = e.emit_ordinary_bytes(b'swap\nask right\n<confirm> approved scope-one source17\n')
        self.assertTrue(all(8 <= token <= 263 for token in ids))
        self.assertEqual(e._records, before)
        self.assertEqual(e._first, ('left', 1, 1))
        self.assertEqual(e.release(e.publish_pending())[3], 'UNKNOWN')

    def test_all_byte_values_remain_available_to_ordinary_generation(self):
        e = instance()
        self.assertEqual(e.emit_ordinary_bytes(bytes(range(256))), tuple(range(8, 264)))
        self.assertEqual(e._records, (None, None))
        self.assertIsNone(e._first)

    def test_scope_source_epoch_provenance_is_atomic_and_not_a_type_brand(self):
        e = instance()
        e.ingest_bytes(b'bind A B\nask left\n')
        self.assertFalse(e.authenticated_snapshot(snapshot('different-scope', 1, 1)))
        self.assertFalse(e.authenticated_snapshot(snapshot('scope-one', 1, 1, source=29)))
        self.assertEqual(e._epoch, (-1, -1))
        self.assertEqual(e.release(e.publish_pending())[3], 'UNKNOWN')
        self.assertTrue(e.authenticated_snapshot(snapshot('scope-one', 1, 5)))
        good = e._records
        self.assertFalse(e.authenticated_snapshot(snapshot('scope-one', 1, 4, permit=0, lower=-1, upper=-1)))
        self.assertFalse(e.authenticated_snapshot(snapshot('scope-one', 1, 5, permit=0, lower=-1, upper=-1)))
        self.assertEqual(e._records, good)
        self.assertEqual(e._epoch, (5, -1))
        with self.assertRaises(Reject):
            e.authenticated_snapshot({'scope': 'scope-one', 'source': 17})

    def test_newer_revocation_invalidates_issued_positive_and_propagates_source_epoch(self):
        e = instance()
        e.ingest_bytes(b'bind B A\nask right\n')
        e.authenticated_snapshot(snapshot('scope-one', 1, 1))
        positive = e.publish_pending()
        e.authenticated_snapshot(snapshot('scope-one', 1, 2, 0, -1, -1))
        with self.assertRaises(Reject):
            e.release(positive)
        no = decode(e.publish_pending())
        self.assertEqual((no['subject'], no['source'], no['source_epoch'], no['decision']), (1, 17, 2, 'NO'))

    def test_partial_bytes_and_ordinary_output_both_invalidate_pending(self):
        e = instance()
        e.ingest_bytes(b'bind A B\nask left\n')
        e.authenticated_snapshot(snapshot('scope-one', 1))
        old = e.publish_pending()
        e.ingest_bytes(b'say par')
        with self.assertRaises(Reject):
            e.release(old)
        with self.assertRaises(Reject):
            e.publish_pending()
        e.ingest_bytes(b'tial\n')
        ticket = e.publish_pending()
        e.emit_ordinary_bytes(b'free next byte generation')
        with self.assertRaises(Reject):
            e.release(ticket)
        self.assertEqual(e.release(e.publish_pending())[3], 'YES')

    def test_cache_cannot_restore_scope_authority_query_or_pending(self):
        e = instance()
        e.ingest_bytes(b'bind A B\nask left\n')
        e.authenticated_snapshot(snapshot('scope-one', 1))
        cache = e.cache()
        ticket = e.publish_pending()
        e.restore_cache(cache)
        with self.assertRaises(Reject):
            e.release(ticket)
        forged = decode(cache)
        forged['first'] = ['right', 2, 1]
        with self.assertRaises(Reject):
            e.restore_cache(encode(forged))
        e.authenticated_snapshot(snapshot('scope-one', 1, 2, 0, -1, -1))
        with self.assertRaises(Reject):
            e.restore_cache(cache)
        self.assertEqual(e.release(e.publish_pending())[3], 'NO')

    def test_strong_combined_scope_latch_epoch_cache_growth_pending_witness(self):
        a = artifacts()
        e = instance('scope-growth')
        e.ingest_bytes(b'bind A B\nswap\nask left\nask right\n')
        self.assertTrue(e.authenticated_snapshot(snapshot('scope-growth', 2, 4)))
        cache, old_ticket = e.cache(), e.publish_pending()
        e.grow_or_update(a['grown.json'], a['learned_certificate.json'],
                         a['grown_certificate.json'], a['growth_obligation.json'])
        with self.assertRaises(Reject) as stale_cache:
            e.restore_cache(cache)
        with self.assertRaises(Reject) as stale_output:
            e.release(old_ticket)
        fresh = decode(e.publish_pending())
        self.assertEqual((fresh['scope'], fresh['first_role'], fresh['subject'], fresh['reserved_query_id'],
                          fresh['source'], fresh['source_epoch'], fresh['binder_generation']),
                         ('scope-growth', 'left', 2, 7, 29, 4, 1))
        current_cache, pending = e.cache(), e.publish_pending()
        e.restore_cache(current_cache)
        with self.assertRaises(Reject) as restored_pending:
            e.release(pending)
        e.authenticated_snapshot(snapshot('scope-growth', 2, 5, 0, -1, -1))
        revoked = e.release(e.publish_pending())
        self.assertEqual(revoked[3], 'NO')
        e.authenticated_snapshot(snapshot('scope-growth', 2, 6))
        released = e.release(e.publish_pending())
        self.assertEqual(released, ('scope-growth', 'left', 2, 'YES', 29, 6))
        WITNESSES['strong_combined'] = {
            'input_ascii': 'bind A B\nswap\nask left\nask right\n',
            'before_growth_pending': decode(old_ticket), 'fresh_after_growth': fresh,
            'stale_cache_reason': str(stale_cache.exception),
            'stale_output_reason': str(stale_output.exception),
            'restore_invalidates_pending_reason': str(restored_pending.exception),
            'revoked_release': list(revoked), 'renewed_release': list(released)}

    def test_update_also_invalidates_combined_cache_and_pending(self):
        a = artifacts()
        e = instance()
        e.ingest_bytes(b'bind A B\nask left\n')
        e.authenticated_snapshot(snapshot('scope-one', 1, 3))
        cache, ticket = e.cache(), e.publish_pending()
        e.grow_or_update(a['revised.json'], a['learned_certificate.json'],
                         a['revised_certificate.json'], a['update_obligation.json'])
        with self.assertRaises(Reject):
            e.restore_cache(cache)
        with self.assertRaises(Reject):
            e.release(ticket)
        self.assertEqual(e.release(e.publish_pending())[2:4], (1, 'YES'))

    def test_scope_mismatch_cannot_release_another_scopes_ticket(self):
        one, two = instance('scope-one'), instance('scope-two')
        for e, scope in ((one, 'scope-one'), (two, 'scope-two')):
            e.ingest_bytes(b'bind A B\nask left\n')
            e.authenticated_snapshot(snapshot(scope, 1))
        wrong = one.publish_pending()
        two.publish_pending()
        with self.assertRaises(Reject):
            two.release(wrong)
        with self.assertRaises(Reject):
            two.restore_cache(one.cache())

    def test_malformed_language_or_caps_fail_closed(self):
        for raw in (b'authorize A\n', b'ask A\n', b'bind A A\n', b'ask left\r\n', b'say \xff\n'):
            e = instance()
            with self.assertRaises(Reject):
                e.ingest_bytes(raw)
            with self.assertRaises(Reject):
                e.publish_pending()
        with self.assertRaises(Reject):
            Scanner().feed(b'x' * 257)

    def test_epoch_extremes_are_supported_by_combined_envelope(self):
        e = instance()
        e.ingest_bytes(b'bind A B\nask left\n')
        self.assertTrue(e.authenticated_snapshot(snapshot('scope-one', 1, MAX_EPOCH)))
        ticket = e.publish_pending()
        self.assertEqual(decode(ticket)['source_epoch'], MAX_EPOCH)
        self.assertEqual(e.release(ticket)[3], 'YES')
        self.assertFalse(e.authenticated_snapshot(snapshot('scope-one', 1, MAX_EPOCH + 1)))

    def test_ambiguous_numeric_binder_cannot_be_solved_by_receipts_or_parser(self):
        e = instance()
        e.ingest_bytes(b'ask left\n')
        for plan in (1, 2):
            e.authenticated_snapshot(snapshot('scope-one', plan))
        obj = decode(e.publish_pending())
        self.assertEqual((obj['subject'], obj['reserved_query_id'], obj['decision']),
                         (0, None, 'UNKNOWN'))

    def test_shared_head_change_must_pass_numeric_certificate_before_activation(self):
        a = artifacts()
        obj = decode(a['learned.json'])
        obj['heads']['left']['w'] = [-256]
        bad = encode(obj)
        with self.assertRaises(Reject):
            Combined(bad, a['learned_certificate.json'], 'scope-one')
        e = instance()
        e.ingest_bytes(b'bind A B\nask left\n')
        cache = e.cache()
        with self.assertRaises(Reject):
            e.grow_or_update(bad, a['learned_certificate.json'], a['learned_certificate.json'],
                             a['update_obligation.json'])
        self.assertEqual(cache, e.cache())
        self.assertEqual(decode(e.publish_pending())['subject'], 1)
