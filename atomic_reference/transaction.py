"""Immutable transition specification plus serialized in-process reference host.

No network, disk commit, actuator, model, learned selector, or authority cache.
Trusted typed events and actual immutable request bytes are caller premises.
"""
from dataclasses import dataclass, replace
import json
import threading
from routed_cases import Event, ROUTES, parse


class Reject(ValueError):
    pass


def need(ok, reason):
    if not ok:
        raise Reject(reason)


def route_ok(route):
    return (type(route) is tuple and len(route) == 4
            and all(type(x) is str for x in route[:3])
            and type(route[3]) is int and route in ROUTES)


@dataclass(frozen=True)
class Row:
    evidence: bool | None = None
    evidence_revision: int = 0
    grant: bool = False
    grant_revision: int = 0


@dataclass(frozen=True)
class Issued:
    nonce: int
    raw: bytes
    request: bytes
    route: tuple
    frontier: int
    model_revision: int
    row: Row


@dataclass(frozen=True)
class Receipt:
    nonce: int
    intent: bytes
    route: tuple
    frontier: int
    model_revision: int
    row: Row
    budget_before: int
    budget_after: int


@dataclass(frozen=True)
class Reference:
    domain: str
    initial_budget: int
    budget: int
    rows: tuple
    frontier: int = 0
    model_revision: int = 0
    accepted_events: int = 0
    issued: tuple = ()
    consumed: tuple = ()
    receipts: tuple = ()
    halted: bool = False


def initial(domain, budget=4):
    need(type(domain) is str and 1 <= len(domain) <= 32 and domain.isascii(), 'domain')
    need(type(budget) is int and 1 <= budget <= 16, 'budget')
    return Reference(domain, budget, budget, tuple(Row() for _ in ROUTES))


def event_step(state, event):
    """One trusted current event. Stale well-formed revisions hold state."""
    need(not state.halted, 'halted')
    need(type(event) is Event and route_ok(event.route), 'event-schema')
    need(type(event.revision) is int and 1 <= event.revision <= 2**31-1, 'event-revision')
    need(type(event.kind) is str and event.kind in ('observe', 'invalidate', 'grant', 'revoke'), 'event-kind')
    need((type(event.value) is bool) if event.kind == 'observe' else event.value is None, 'event-value')
    index = ROUTES.index(event.route)
    old = state.rows[index]
    previous = old.grant_revision if event.kind in ('grant', 'revoke') else old.evidence_revision
    if event.revision <= previous:
        return state, False
    need(state.accepted_events < 64, 'event-cap')
    if event.kind in ('grant', 'revoke'):
        row = replace(old, grant=event.kind == 'grant', grant_revision=event.revision)
    else:
        row = replace(old, evidence=event.value if event.kind == 'observe' else None,
                      evidence_revision=event.revision)
    rows = state.rows[:index] + (row,) + state.rows[index+1:]
    return replace(state, rows=rows, frontier=state.frontier+1,
                   accepted_events=state.accepted_events+1), True


def revision_step(state, revision):
    """Invalidation only, NOT a certificate for newly installed model semantics."""
    need(not state.halted, 'halted')
    need(type(revision) is int and revision == state.model_revision+1 and revision <= 8,
         'model-revision')
    return replace(state, model_revision=revision, frontier=state.frontier+1)


def issue_step(state, request, guessed_route):
    """Reject-only binding. The grammar never repairs a selector's wrong guess."""
    need(not state.halted, 'halted')
    need(type(request) is bytes and len(request) <= 512, 'request-bytes')
    need(route_ok(guessed_route) and parse(request) == guessed_route, 'request-route')
    row = state.rows[ROUTES.index(guessed_route)]
    need(row.evidence is True and row.grant is True, 'current-license')
    need(state.budget > 0 and len(state.issued) < 16, 'issue-cap-or-budget')
    nonce = len(state.issued)+1
    obj = dict(schema='atomic-reference-intent-v1', domain=state.domain,
               request_hex=request.hex(), route=guessed_route, nonce=nonce, cost=1,
               frontier=state.frontier, model_revision=state.model_revision,
               evidence_revision=row.evidence_revision, grant_revision=row.grant_revision)
    raw = json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')
    need(len(raw) <= 2048, 'intent-size')
    record = Issued(nonce, raw, request, guessed_route, state.frontier, state.model_revision, row)
    return replace(state, issued=state.issued+(record,)), raw


def commit_step(state, raw, actual_request):
    """Single atomic modeled transition: validate current state AND consume once.

    Exact raw equality with an owned issued record, not a client digest or cache,
    determines membership. No parser output or parsed field from raw is trusted.
    """
    need(not state.halted, 'halted')
    need(type(raw) is bytes and len(raw) <= 2048, 'intent-bytes')
    need(type(actual_request) is bytes and len(actual_request) <= 512, 'actual-request')
    record = next((item for item in state.issued if item.raw == raw), None)
    need(record is not None, 'not-exact-issued-intent')
    need(record.nonce not in state.consumed, 'already-consumed')
    need(record.request == actual_request and parse(actual_request) == record.route, 'actual-request-binding')
    need(record.frontier == state.frontier and record.model_revision == state.model_revision,
         'current-frontier')
    row = state.rows[ROUTES.index(record.route)]
    need(row == record.row and row.evidence is True and row.grant is True, 'current-license')
    need(state.budget > 0, 'budget-exhausted')
    receipt = Receipt(record.nonce, record.raw, record.route, state.frontier,
                      state.model_revision, row, state.budget, state.budget-1)
    return replace(state, consumed=state.consumed+(record.nonce,),
                   receipts=state.receipts+(receipt,), budget=state.budget-1), receipt


def bookkeeping_invariant(state):
    """Executable necessary invariant, not a universal theorem/native proof."""
    return (type(state) is Reference and state.budget >= 0
            and state.budget+len(state.consumed) == state.initial_budget
            and len(set(state.consumed)) == len(state.consumed) == len(state.receipts)
            and tuple(item.nonce for item in state.issued) == tuple(range(1, len(state.issued)+1))
            and all(nonce in range(1, len(state.issued)+1) for nonce in state.consumed)
            and all(r.nonce == nonce and r.row.evidence is True and r.row.grant is True
                    and r.budget_before-r.budget_after == 1
                    and r.intent == state.issued[nonce-1].raw
                    for nonce, r in zip(state.consumed, state.receipts))
            and state.frontier == state.accepted_events+state.model_revision)


class AtomicHost:
    """One trusted process/lineage. All protected operations use the SAME lock.

    No restore-authority API. Old learned caches may propose a route only.
    Python private names are conventions, not protection against hostile code.
    """
    def __init__(self, domain, budget=4):
        self._state = initial(domain, budget)
        self._lock = threading.Lock()

    def inspect(self):
        with self._lock:
            return self._state

    def apply(self, event):
        with self._lock:
            try:
                self._state, accepted = event_step(self._state, event)
            except Reject:
                self._state = replace(self._state, halted=True)
                raise
            return accepted

    def revise(self, revision):
        with self._lock:
            try:
                self._state = revision_step(self._state, revision)
            except Reject:
                self._state = replace(self._state, halted=True)
                raise

    def issue(self, request, guessed_route):
        with self._lock:
            self._state, raw = issue_step(self._state, request, guessed_route)
            return raw

    def commit(self, raw, actual_request):
        with self._lock:
            self._state, receipt = commit_step(self._state, raw, actual_request)
            return receipt
