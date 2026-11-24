"""Independent declared semantics, never imported by candidate inference.

Abstract states0=unbound,1=left A/right B,2=left B/right A.
Evidence status0=unknown,1=supported,2=contradicted. Epochs0..2 never wrap.
"""
CONTEXT = ('set_ab', 'set_ba', 'swap')
QUERIES = ('left', 'right')
CONTRACT_ID = 'two-referents-current-evidence-v1'


def step(state, token):
    if token == 'set_ab':
        return 1
    if token == 'set_ba':
        return 2
    if token == 'swap':
        return (0, 2, 1)[state]
    raise ValueError('abstract-context')


def referent(state, query):
    if state == 0:
        return 0
    if query == 'left':
        return state
    if query == 'right':
        return 3 - state
    raise ValueError('abstract-query')


def licensed(subject, grant, status, proof_epoch, current_epoch, budget):
    return (subject in (1, 2) and grant and status == 1
            and proof_epoch == current_epoch and budget == 1)
