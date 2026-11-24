"""Qualification-only typed observer, protected decision readout, and reject-only sink.

Premises: the caller authenticates events, supplies the actual request bytes at both
boundaries, and serializes apply/dispatch on current authoritative state. The
grammar is intentionally finite. Python arbitrary-precision integer operations
are the executable numerical specification. This path does not infer truth,
authenticate a source, or certify the independent LM's ordinary prose.
"""

from dataclasses import dataclass, replace

from routed_cases import Event, ROUTES, parse


def _natural(x):
    return type(x) is int and x >= 0


def _route_ok(route):
    # Python otherwise equates True, 1, and 1.0 inside tuples.
    return (type(route) is tuple and len(route) == 4
            and all(type(x) is str for x in route[:3])
            and type(route[3]) is int and route in ROUTES)


@dataclass(frozen=True)
class Mode:
    """Allowed typed-observer parameters. The byte LM optimizer does not set them."""

    retain: int = 2
    drive: int = 12
    weight: int = 1

    def __post_init__(self):
        if not (_natural(self.retain) and self.retain <= 2
                and type(self.drive) is int and 10 <= self.drive <= 12
                and _natural(self.weight) and self.weight <= 16):
            raise ValueError("invalid observer mode")


@dataclass(frozen=True)
class Cell:
    coords: tuple
    revision: int = 0
    granted: bool = False
    grant_revision: int = 0


@dataclass(frozen=True)
class Snapshot:
    version: int
    routes: tuple
    modes: tuple
    cells: tuple


def _check_modes(modes):
    if (type(modes) is not tuple or not modes
            or any(type(m) is not Mode for m in modes)
            or sum(m.weight for m in modes) == 0):
        raise ValueError("invalid observer/readout parameters")


def _check_cell(cell, width):
    if (type(cell) is not Cell or not _natural(cell.revision)
            or not _natural(cell.grant_revision) or type(cell.granted) is not bool
            or (cell.granted and cell.grant_revision == 0)
            or type(cell.coords) is not tuple or len(cell.coords) != width
            or any(type(x) is not int for x in cell.coords)):
        raise ValueError("invalid observer cache")
    zero = all(x == 0 for x in cell.coords)
    yes = all(6 <= x <= 16 for x in cell.coords)
    no = all(-16 <= x <= -6 for x in cell.coords)
    if not (zero or ((yes or no) and cell.revision > 0)):
        raise ValueError("cache outside reachable evidence enclosure")


class State:
    """Independent integer observers per (entity, task, authenticated source, epoch).

    An observation s in {-1,+1} writes floor(retain*x/8) + s*drive.
    On [-16,16] the residual lies in [-4,4]. Thus every coordinate is
    in [6,16] after a positive event and [-16,-6] after a negative one.
    Invalidation zeros the entire row; other routes and prose never write it.
    Nonnegative readout weights with positive sum preserve all three regions.
    """

    def __init__(self, modes=(Mode(),)):
        _check_modes(modes)
        self._modes = modes
        self._rows = {route: Cell((0,) * len(modes)) for route in ROUTES}

    def inspect(self, route):
        if not _route_ok(route):
            raise ValueError("unknown route")
        return self._rows[route]

    def evidence(self, route):
        coords = self.inspect(route).coords
        score = sum(m.weight * x for m, x in zip(self._modes, coords, strict=True))
        if score == 0:
            return "UNKNOWN"
        return "YES" if score > 0 else "NO"

    def apply(self, event):
        """Apply one authenticated event. Ignore stale, well-formed revisions.

        Event type validation is a schema boundary. The trusted dispatcher owns
        source authentication. In particular this method never accepts LM bytes.
        """
        if type(event) is not Event or type(event.kind) is not str or not _route_ok(event.route):
            raise ValueError("invalid typed event")
        if not _natural(event.revision) or event.revision == 0:
            raise ValueError("revision must be a positive integer")
        if event.kind == "observe":
            if type(event.value) is not bool:
                raise ValueError("observation must be a boolean")
        elif event.kind not in ("invalidate", "grant", "revoke") or event.value is not None:
            raise ValueError("invalid event payload")
        old = self.inspect(event.route)
        if event.kind in ("grant", "revoke"):
            if event.revision <= old.grant_revision:
                return False
            new = replace(old, granted=event.kind == "grant", grant_revision=event.revision)
        else:
            if event.revision <= old.revision:
                return False
            if event.kind == "invalidate":
                coords = (0,) * len(self._modes)
            else:
                sign = 1 if event.value else -1
                coords = tuple(m.retain * x // 8 + sign * m.drive
                               for m, x in zip(self._modes, old.coords, strict=True))
            new = replace(old, coords=coords, revision=event.revision)
        self._rows[event.route] = new
        return True

    def snapshot(self):
        return Snapshot(1, tuple(ROUTES), self._modes, tuple(self._rows[r] for r in ROUTES))

    @classmethod
    def restore(cls, snap):
        """Check cache schema/enclosure. Trusted lineage/freshness is external."""
        if (type(snap) is not Snapshot or type(snap.version) is not int or snap.version != 1
                or type(snap.routes) is not tuple or snap.routes != tuple(ROUTES)
                or not all(_route_ok(r) for r in snap.routes)
                or type(snap.cells) is not tuple or len(snap.cells) != len(ROUTES)):
            raise ValueError("invalid observer snapshot")
        state = cls(snap.modes)
        for route, cell in zip(ROUTES, snap.cells, strict=True):
            _check_cell(cell, len(snap.modes))
            state._rows[route] = cell
        return state

    def clone(self):
        return self.restore(self.snapshot())

    def with_parameters(self, modes):
        """Accept only admissible same-width proposals. The old state stays intact."""
        _check_modes(modes)
        if len(modes) != len(self._modes):
            raise ValueError("use grow to change width")
        return self.restore(replace(self.snapshot(), modes=modes))

    def grow(self):
        """Embed each cached row by duplicating mode zero with zero readout weight.

        Old-coordinate recurrence is uncoupled. For identical future authenticated
        events, old coordinates, old parameters and all raw typed decisions agree.
        This operation grows protected observer width only.
        """
        snap = self.snapshot()
        extra = replace(self._modes[0], weight=0)
        cells = tuple(replace(c, coords=c.coords + (c.coords[0],)) for c in snap.cells)
        return self.restore(replace(snap, modes=snap.modes + (extra,), cells=cells))


@dataclass(frozen=True)
class Intent:
    kind: str
    route: tuple | None
    revision: int
    grant_revision: int = 0


def candidate(state, proposed_route):
    """Pre-witness typed proposal. It may act on the wrong referent."""
    if not _route_ok(proposed_route):
        return Intent("UNSUPPORTED", None, 0)
    cell = state.inspect(proposed_route)
    status = state.evidence(proposed_route)
    if status == "YES":
        status = "ALLOW" if cell.granted else "DENY"
    return Intent(status, proposed_route, cell.revision, cell.grant_revision)


def propose(state, request, proposed_route):
    """Actual pre-sink AST. The grammar witness can reject, never repairs a route."""
    witness = parse(request) if type(request) is bytes else None
    if witness is None:
        return Intent("UNSUPPORTED", None, 0)
    if not _route_ok(proposed_route) or proposed_route != witness:
        return Intent("SCOPE_MISMATCH", None, 0)
    return candidate(state, proposed_route)


def sink(state, intent, request=None):
    """Return True only for an unchanged, current ALLOW under the real request.

    This models acceptance, not command execution, a replay nonce, or an atomic
    external transaction. A production dispatcher must bind the real request and
    check and act under a common authoritative lock/version fence.
    """
    if (type(intent) is not Intent or type(request) is not bytes
            or intent.kind != "ALLOW" or not _route_ok(intent.route)
            or not _natural(intent.revision) or not _natural(intent.grant_revision)):
        return False
    if parse(request) != intent.route:
        return False
    return intent == candidate(state, intent.route)
