"""Synthetic native interval binder/effect engine; no actuator or text oracle.

Strict CURRENT-context policy: context changes invalidate a previous request.
Model changes recheck certificates, replay owned history, and change generation.
"""
import checker
from model import Program
from wire import encode, decode, digest, need, Reject


def licensed_interval(subject, grant, evidence, proof_epoch, current_epoch, budget):
    return (subject in (1, 2) and grant and evidence[0] > 0
            and proof_epoch == current_epoch and budget == 1)


class Engine:
    def __init__(self, raw, certificate):
        checker.admit(raw, certificate)
        self._initialize_unchecked(raw)

    def _initialize_unchecked(self, raw):
        # Private TCB initialization; only the explicit bridge-removal mutant
        # bypasses public admission. Private names are not hostile-code security.
        self._raw, self._program = raw, Program.from_admitted(raw)
        self._hidden, self._box = self._program.initial()
        self._history = ()
        self._generation = 0
        self._context = 0
        self._issued = None
        self._requests = 0
        self._faulted = False
        self._grants = [False, False]
        self._evidence = [(-1, 1), (-1, 1)]
        self._epochs = [0, 0]
        self._budget = 1

    def consume(self, token, errors=None):
        self._issued = None
        try:
            need(not self._faulted and len(self._history) < 32, 'context-cap-or-fault')
            errors = (0,) * self._program.width if errors is None else errors
            hidden, box = self._program.step(self._hidden, self._box, token, errors)
        except Reject:
            self._faulted = True
            raise
        self._hidden, self._box = hidden, box
        self._history += ((token, errors),)
        self._context += 1

    def _replay(self, program):
        hidden, box = program.initial()
        for token, errors in self._history:
            need(len(errors) <= program.width, 'replay-width')
            # New coordinates did not exist in old records: declared warm replay
            # uses zero synthetic error, not a fabricated measured hardware trace.
            expanded = errors + (0,) * (program.width - len(errors))
            hidden, box = program.step(hidden, box, token, expanded)
        return hidden, box

    def _source_bytes(self):
        hidden, box = self._replay(self._program)
        need((hidden, box) == (self._hidden, self._box), 'source-replay-mismatch')
        return encode({'schema': 'quantized-binder-source-v1', 'model_sha': digest(self._raw),
                       'generation': self._generation, 'context': self._context,
                       'history_sha': digest(encode(self._history)),
                       'hidden': list(hidden), 'box': [list(x) for x in box]})

    def cache(self):
        return self._source_bytes()

    def restore_cache(self, raw):
        need(type(raw) is bytes and raw == self._source_bytes(), 'stale-or-forged-source-cache')
        self._hidden, self._box = self._replay(self._program)
        # Authority/evidence/budget never restored from a neural cache.

    def change(self, new_raw, old_proof, new_proof, change):
        need(not self._faulted and self._generation < 8, 'model-generation-cap-or-fault')
        checker.admit_change(self._raw, old_proof, new_raw, new_proof, change)
        program = Program.from_admitted(new_raw)
        hidden, box = self._replay(program)
        # All staging/checking above precedes this single-threaded transaction.
        self._raw, self._program = new_raw, program
        self._hidden, self._box = hidden, box
        self._generation += 1
        self._issued = None

    def grant(self, subject, granted):
        valid = type(subject) is int and subject in (1, 2) and type(granted) is bool
        if not valid:
            self._faulted, self._issued = True, None
        need(valid, 'grant-schema')
        self._grants[subject - 1] = granted

    def evidence(self, subject, lo, hi):
        valid = (type(subject) is int and subject in (1, 2) and type(lo) is int and type(hi) is int
                 and -1 <= lo <= hi <= 1)
        if not valid:
            self._faulted, self._issued = True, None
        need(valid, 'evidence-interval')
        i = subject - 1
        if self._epochs[i] >= 2:
            self._evidence[i] = (-1, 1)
            need(False, 'epoch-exhausted-unknown')
        self._epochs[i] += 1
        self._evidence[i] = (lo, hi)

    def request(self, query):
        need(not self._faulted and self._requests < 16, 'request-cap-or-fault')
        source = self._source_bytes()
        subject, score, bounds = self._program.select(self._hidden, self._box, query)
        self._requests += 1
        self._issued = encode({'request': self._requests, 'query': query, 'subject': subject,
                               'model_sha': digest(self._raw), 'generation': self._generation,
                               'context': self._context, 'source_sha': digest(source),
                               'score_interval': list(bounds), 'operation': 'dispatch', 'cost': 1,
                               'token': 'synthetic-session',
                               'evidence_epoch': self._epochs[subject - 1] if subject else 0})
        return self._issued

    def commit(self, raw):
        need(not self._faulted, 'faulted')
        p = decode(raw)
        need(type(raw) is bytes and self._issued is not None and raw == self._issued, 'fresh-issued-binding')
        need(p['context'] == self._context and p['generation'] == self._generation
             and p['model_sha'] == digest(self._raw)
             and p['source_sha'] == digest(self._source_bytes()), 'fresh-source-binding')
        subject = p['subject']
        need(subject in (1, 2), 'ambiguous-referent')
        i = subject - 1
        # Only a strictly positive lower support bound establishes evidence.
        need(licensed_interval(subject, self._grants[i], self._evidence[i],
                               p['evidence_epoch'], self._epochs[i], self._budget),
             'ambiguous-revoked-stale-or-unlicensed')
        self._budget = 0
        return (p['request'], subject, 'dispatch', 1, p['evidence_epoch'])
