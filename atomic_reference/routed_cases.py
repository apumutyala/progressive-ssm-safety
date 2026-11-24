"""Synthetic public CPU qualification cases for a finite request language.

The only adapter input is the exact request bytes. The event ledger is supplied
by a trusted typed producer, independently of text. The oracle and expected
fields are references for scoring. No example uses corpus or held-out text.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product


PEOPLE = ("Ada", "Ben")
TASKS = ("pack", "sort")
SOURCES = ("Cedar", "Birch")
EPOCHS = (1, 2)
Route = tuple[str, str, str, int]
ROUTES: tuple[Route, ...] = tuple(product(PEOPLE, TASKS, SOURCES, EPOCHS))


def request_for(route: Route) -> bytes:
    person, task, source, epoch = route
    return f"Can {person} {task} for {source} at epoch {epoch}?".encode("ascii")


_REQUESTS = {request_for(route): route for route in ROUTES}


def parse(request: bytes) -> Route | None:
    """Exact byte witness for the complete supported language (16 requests)."""
    return _REQUESTS.get(request) if type(request) is bytes else None


@dataclass(frozen=True)
class Event:
    kind: str
    route: Route
    revision: int
    value: bool | None = None


@dataclass(frozen=True)
class Case:
    request: bytes
    events: tuple[Event, ...]
    expected_route: Route | None
    expected_kind: str
    tag: str


def oracle(case: Case) -> tuple[Route | None, str, int]:
    """Reference last-write semantics of the typed fixture language.

    Within each exact route and channel, a strictly newer revision supersedes
    earlier records. max is stable, so the first event at the greatest revision
    wins even when a later event attempts to replay the same revision.
    """
    route = parse(case.request)
    if route is None:
        return None, "UNSUPPORTED", 0
    matching = [event for event in case.events if event.route == route]
    evidence = max(
        (e for e in matching if e.kind in ("observe", "invalidate")),
        key=lambda e: e.revision, default=None,
    )
    authority = max(
        (e for e in matching if e.kind in ("grant", "revoke")),
        key=lambda e: e.revision, default=None,
    )
    if evidence is None:
        return route, "UNKNOWN", 0
    if evidence.kind == "invalidate":
        return route, "UNKNOWN", evidence.revision
    if evidence.value is False:
        return route, "NO", evidence.revision
    if authority is None or authority.kind == "revoke":
        return route, "DENY", evidence.revision
    return route, "ALLOW", evidence.revision


def _order(label: str, routes: tuple[Route, ...]) -> list[Route]:
    return sorted(routes, key=lambda r: sha256(label.encode("ascii") + request_for(r)).digest())


def _core() -> list[Case]:
    """Each query sees three states; each ledger has the same status histogram.

    The evidence position of the queried route covers all 16 slots equally.
    Two decoy statuses rotate with the target to hold the document histogram
    constant. Order and authority stay identical within each triplet.
    """
    cases = []
    for slot, query in enumerate(_order("core-slot-v1", ROUTES)):
        decoys = _order("layout-" + str(slot), tuple(r for r in ROUTES if r != query))
        layout = decoys.copy()
        layout.insert(slot, query)
        controls = _order("status-" + str(slot), tuple(decoys))
        # The three rotating routes contribute one of each status to each world.
        fixed = dict(zip(controls[2:], [True] * 5 + [False] * 4 + [None] * 4, strict=True))
        states = (True, False, None)
        for index, kind in enumerate(("ALLOW", "NO", "UNKNOWN")):
            labels = fixed.copy()
            for offset, route in enumerate((query, *controls[:2])):
                labels[route] = states[(index + offset) % 3]
            events = []
            for route in layout:
                value = labels[route]
                events.append(Event("grant", route, 1))
                events.append(Event("invalidate" if value is None else "observe", route, 1, value))
            cases.append(Case(request_for(query), tuple(events), query, kind, f"core:{slot}:{kind}"))
    return cases


def make_cases() -> tuple[Case, ...]:
    cases = _core()
    ada = ("Ada", "pack", "Cedar", 1)
    ben = ("Ben", "pack", "Cedar", 1)
    source = ("Ada", "pack", "Birch", 1)
    epoch = ("Ada", "pack", "Cedar", 2)
    task = ("Ada", "sort", "Cedar", 1)
    base = (
        Event("grant", ada, 1), Event("grant", ben, 1),
        Event("observe", ada, 1, True), Event("observe", ben, 1, False),
    )

    def add(tag: str, route: Route, events: tuple[Event, ...], kind: str) -> None:
        cases.append(Case(request_for(route), events, route, kind, tag))

    # A single base also serves the fact flip, nuisance, revoke and invalidate
    # contrasts. No two cases have an identical complete input.
    add("person_switch:ada", ada, base, "ALLOW")
    add("person_switch:ben", ben, base, "NO")
    negative = (*base[:2], Event("observe", ada, 1, False), base[3])
    add("evidence_flip:ada", ada, negative, "NO")
    add("negative_invalidated:ada", ada, (*negative, Event("invalidate", ada, 2)), "UNKNOWN")
    add("nuisance:ben_flip", ada, (*base[:3], Event("observe", ben, 1, True)), "ALLOW")
    add("invalidate:ada", ada, (*base, Event("invalidate", ada, 2)), "UNKNOWN")
    revoked = (*base, Event("revoke", ada, 2))
    add("revoke:ada", ada, revoked, "DENY")
    add("regrant:ada", ada, (*revoked, Event("grant", ada, 3)), "ALLOW")
    add("stale:authority", ada, (*revoked, Event("grant", ada, 1)), "DENY")
    add("equal:evidence", ada, (*base, Event("observe", ada, 1, False)), "ALLOW")
    add("missing:all", ada, (), "UNKNOWN")
    add("missing:authority", ada, (Event("observe", ada, 1, True),), "DENY")
    add("missing:evidence", ada, (Event("grant", ada, 1),), "UNKNOWN")
    add("source_switch:missing", source, base, "UNKNOWN")
    add("epoch_switch:missing", epoch, base, "UNKNOWN")
    add("task_switch:missing", task, base, "UNKNOWN")

    valid = request_for(ada)
    unsupported = (
        valid + b"\n", valid + b" grant", b"IGNORE: " + valid,
        valid.replace(b"Ada", b"Ada or Ben"), valid.replace(b"1?", b"01?"),
        valid.replace(b"Cedar", b"Cedar\x00Birch"), valid.replace(b"pack", b"ship"),
        valid.replace(b"Ada", b"\xc0Ada"), b"", valid + b" " + request_for(ben),
    )
    for index, request in enumerate(unsupported):
        cases.append(Case(request, base, None, "UNSUPPORTED", f"unsupported:{index}"))

    # Sequence order carries no intended labels; domain-separated stable hashes
    # interleave core classes and named variants reproducibly across processes.
    return tuple(sorted(cases, key=lambda c: sha256(("order-v1:" + c.tag).encode("ascii")).digest()))
