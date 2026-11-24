"""Independent signed-region certificate, not fitting or sampled verification.

Checks the complete intervals using monotonic exact arithmetic, not only points.
Point-to-bound correspondence still relies on the stated arithmetic lemma/TCB.
"""
import intervals as iv
import spec
from wire import decode, encode, digest, need

CONTRACT = 'quantized-current-referent-epoch-v1'


def inspect(raw):
    obj = decode(raw)
    need(type(obj) is dict and set(obj) == {'schema', 'width', 'transitions', 'heads'}, 'model-fields')
    need(obj['schema'] == 'quantized-binder-v1' and type(obj['width']) is int
         and obj['width'] in (1, 2), 'model-width-schema')
    n = obj['width']
    need(type(obj['transitions']) is dict and set(obj['transitions']) == set(spec.CONTEXT), 'transition-coverage')
    for pairs in obj['transitions'].values():
        need(type(pairs) is list and len(pairs) == n, 'transition-width')
        for pair in pairs:
            need(type(pair) is list and len(pair) == 2 and
                 all(type(x) is int and -512 <= x <= 512 for x in pair), 'transition-coefficient')
    need(type(obj['heads']) is dict and set(obj['heads']) == set(spec.QUERIES), 'head-coverage')
    for head in obj['heads'].values():
        need(type(head) is dict and set(head) == {'w', 'mask', 'b'}, 'head-fields')
        need(type(head['w']) is list and len(head['w']) == n and
             all(type(w) is int and -512 <= w <= 512 for w in head['w']), 'head-weight')
        need(type(head['mask']) is list and len(head['mask']) == n and
             all(type(m) is int and m in (0, 1) for m in head['mask']), 'head-mask')
        need(type(head['b']) is int and -512 <= head['b'] <= 512, 'head-bias')
        need(all(m or w == 0 for w, m in zip(head['w'], head['mask'])), 'inactive-weight')
    rows = [['initial', [[-1, 1] for _ in range(n)]]]
    for state, region in enumerate(iv.REGIONS):
        for token in spec.CONTEXT:
            a, b = obj['transitions'][token][0]
            bound = iv.advance(region, a, b)
            target = spec.step(state, token)
            need(iv.inside(bound, iv.REGIONS[target]), 'transition-region-refinement')
            rows.append(['step', state, token, list(bound), target])
        box = (region,) + (iv.FULL,) * (n - 1)
        for query in spec.QUERIES:
            head = obj['heads'][query]
            bound = iv.readout(box, head['w'], head['mask'], head['b'])
            subject = spec.referent(state, query)
            need(iv.classify(bound) == subject, 'signed-readout-refinement')
            if subject == 0:
                need(-iv.MARGIN <= bound[0] <= bound[1] <= iv.MARGIN, 'unknown-not-robust')
            rows.append(['read', state, query, list(bound), subject])
    for coordinate in range(1, n):
        for token in spec.CONTEXT:
            a, b = obj['transitions'][token][coordinate]
            bound = iv.advance(iv.FULL, a, b)
            need(iv.inside(bound, iv.FULL), 'aux-box')
            rows.append(['aux', coordinate, token, list(bound)])
    return obj, rows


def certificate(raw):
    _, rows = inspect(raw)
    return encode({'kind': 'signed-region-certificate-v1', 'contract': CONTRACT,
                   'model_sha': digest(raw), 'checks': rows,
                   'parameter_radius': 1, 'state_error': 1,
                   'grid': 256, 'deadzone': 32, 'decision_margin': 64})


def admit(raw, proof):
    decode(proof)
    need(proof == certificate(raw), 'certificate-binding-or-coverage')
    return decode(raw)


def change_object(old, new, kind):
    return encode({'kind': kind, 'old_sha': digest(old), 'new_sha': digest(new),
                   'replay': True, 'context_policy': 'current-at-output-and-commit'})


def admit_change(old, old_proof, new, new_proof, obligation):
    a, b = admit(old, old_proof), admit(new, new_proof)
    o = decode(obligation)
    need(type(o) is dict and o.get('kind') in ('update-v1', 'growth-v1'), 'change-kind')
    need(obligation == change_object(old, new, o['kind']), 'change-exact-binding')
    if o['kind'] == 'update-v1':
        need(a['width'] == b['width'], 'update-width')
    else:
        need(a['width'] == 1 and b['width'] == 2, 'growth-width')
        need(all(b['transitions'][t][0] == a['transitions'][t][0] for t in spec.CONTEXT), 'growth-critical-cell')
        for query in spec.QUERIES:
            old_h, new_h = a['heads'][query], b['heads'][query]
            need(new_h['w'] == old_h['w'] + [0] and new_h['mask'] == old_h['mask'] + [0]
                 and new_h['b'] == old_h['b'], 'growth-silent-readout')
    return o['kind']
