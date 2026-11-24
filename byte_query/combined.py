"""Isolated combined witness, NOT imported into or merged with the GPU model.

Typed provenance is an assumed authenticated caller channel. A Python dataclass
or source number is not cryptographic authentication. Byte parser never creates
these events. Query identity comes from exact learned quantized recurrence.
"""
from dataclasses import dataclass
from runtime import Engine as NumericBinder
from grammar import Scanner
from envelope import encode, decode, digest, need, Reject

AUTHORITY_SOURCES = (17, 29)
MAX_EPOCH = 2**31 - 1


@dataclass(frozen=True)
class TrustedSnapshot:
    scope: str
    plan: int
    source: int
    epoch: int
    permit: int
    lower: int
    upper: int


class Combined:
    def __init__(self, model_bytes, certificate, scope):
        need(type(scope) is str and 1 <= len(scope) <= 32
             and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in scope), 'scope-schema')
        self._scope = scope
        self._binder = NumericBinder(model_bytes, certificate)
        self._scanner = Scanner()
        self._input = b''
        self._output = b''
        self._first = None
        self._binding_stale = False
        self._pending = None
        self._records = (None, None)
        self._epoch = (-1, -1)
        self._fault = False
        self._budget = 1
        self._ticket = 0
        self._ignored_queries = 0

    def ingest_bytes(self, raw):
        self._pending = None  # even a partial new byte invalidates old output
        try:
            need(not self._fault and type(raw) is bytes, 'input-type-or-fault')
            events = self._scanner.feed(raw)
            self._input += raw
            for kind, value in events:
                if kind == 'context':
                    self._binder.consume(value)
                    if self._first is not None:
                        self._binding_stale = True
                elif kind == 'query':
                    if self._first is None:
                        binding = decode(self._binder.request(value))
                        self._first = (value, binding['subject'], binding['context'])
                    else:
                        self._ignored_queries += 1
                # Text is intentionally NOT interpreted as context or authority.
            return events
        except Reject:
            self._fault = True
            raise

    def authenticated_snapshot(self, event):
        self._pending = None
        need(type(event) is TrustedSnapshot, 'trusted-channel-schema')
        valid = (type(event.scope) is str and event.scope == self._scope
                 and type(event.plan) is int and event.plan in (1, 2)
                 and type(event.source) is int and type(event.epoch) is int
                 and 0 <= event.epoch <= MAX_EPOCH
                 and type(event.permit) is int and event.permit in (0, 1)
                 and type(event.lower) is int and type(event.upper) is int)
        if not valid:
            return False
        index = event.plan - 1
        bounds = 0 <= event.lower <= 1_000_000 and 0 <= event.upper <= 1_000_000
        revoke = event.permit == 0 and event.lower == -1 and event.upper == -1
        if (event.source != AUTHORITY_SOURCES[index] or event.epoch <= self._epoch[index]
                or not (bounds or revoke)):
            return False
        epochs = list(self._epoch)
        records = list(self._records)
        epochs[index] = event.epoch
        records[index] = (event.scope, event.source, event.epoch,
                          event.permit, event.lower, event.upper)
        self._epoch, self._records = tuple(epochs), tuple(records)
        return True

    def emit_ordinary_bytes(self, raw):
        need(not self._fault and type(raw) is bytes and len(self._output) + len(raw) <= 1024,
             'ordinary-output-cap-or-fault')
        self._output += raw
        self._pending = None
        # Literal bytes including reserved-looking text remain ordinary IDs.
        return tuple(8 + byte for byte in raw)

    def grow_or_update(self, new_bytes, old_certificate, new_certificate, obligation):
        need(not self._fault, 'fault')
        self._binder.change(new_bytes, old_certificate, new_certificate, obligation)
        self._pending = None

    def _cache_bytes(self):
        return encode({'schema': 'controlled-byte-query-cache-v1', 'scope': self._scope,
                       'input_sha': digest(self._input), 'input_bytes': len(self._input),
                       'output_sha': digest(self._output), 'output_bytes': len(self._output),
                       'parser_pending_sha': digest(self._scanner.pending),
                       'parser_pending_bytes': len(self._scanner.pending),
                       'parser_records': self._scanner.records,
                       'first': self._first, 'binding_stale': self._binding_stale,
                       'authority_sources': list(AUTHORITY_SOURCES),
                       'epochs': self._epoch, 'records': self._records,
                       'binder': decode(self._binder.cache())})

    def cache(self):
        return self._cache_bytes()

    def restore_cache(self, raw):
        self._pending = None
        need(type(raw) is bytes and raw == self._cache_bytes(), 'combined-cache-not-current-owned')
        # No trusted scope, latch, receipt or epoch is restored from the blob.
        self._binder.restore_cache(self._binder.cache())

    def publish_pending(self):
        need(not self._fault and not self._scanner.pending and self._first is not None
             and not self._binding_stale and self._ticket < 16, 'no-fresh-bound-query')
        role, subject, bound_context = self._first
        binding = decode(self._binder.request(role))
        need(binding['subject'] == subject and binding['context'] == bound_context,
             'latched-query-context-mismatch')
        row = self._records[subject - 1] if subject in (1, 2) else None
        decision = 'UNKNOWN'
        if row is not None:
            if row[3] == 0:
                decision = 'NO'
            elif 0 <= row[5] <= row[4]:
                decision = 'YES'
        self._ticket += 1
        self._pending = encode({'schema': 'pending-typed-query-v1', 'ticket': self._ticket,
                                'scope': self._scope, 'first_role': role, 'subject': subject,
                                'reserved_query_id': subject + 5 if subject else None,
                                'decision': decision, 'source': row[1] if row else None,
                                'source_epoch': row[2] if row else -1,
                                'source_snapshot_sha': digest(encode(row)),
                                'combined_state_sha': digest(self._cache_bytes()),
                                'binder_generation': binding['generation'],
                                'binder_model_sha': binding['model_sha'],
                                'status': 'PENDING_NOT_RELEASED'})
        return self._pending

    def release(self, raw):
        need(not self._fault and type(raw) is bytes and self._pending is not None
             and raw == self._pending, 'pending-output-not-current')
        ticket = decode(raw)
        need(ticket['combined_state_sha'] == digest(self._cache_bytes())
             and ticket['scope'] == self._scope and not self._binding_stale,
             'release-freshness')
        if ticket['decision'] == 'YES':
            need(self._budget == 1, 'one-use-budget')
            self._budget = 0
        self._pending = None
        return (ticket['scope'], ticket['first_role'], ticket['subject'], ticket['decision'],
                ticket['source'], ticket['source_epoch'])
