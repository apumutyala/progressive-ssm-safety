"""Exact least-squares fitting of affine recurrent/readout coefficients.

Supervised numeric preactivation examples; no spec/checker/reference inference.
Domain and latent supervision are supplied, not discovered language semantics.
"""
from fractions import Fraction
from wire import encode, decode, need

CORE = {
    'set_ab': ((-256, 176), (0, 208), (256, 240)),
    'set_ba': ((-256, -240), (0, -208), (256, -176)),
    'swap': ((-128, 192), (0, 0), (128, -192)),
}
REVISED_CORE = {
    'set_ab': ((-256, 176), (0, 192), (256, 208)),
    'set_ba': ((-256, -208), (0, -192), (256, -176)),
    'swap': CORE['swap'],
}
AUX = {
    'set_ab': ((-256, 48), (0, 64), (256, 80)),
    'set_ba': ((-256, -80), (0, -64), (256, -48)),
    'swap': ((-128, 64), (0, 0), (128, -64)),
}
HEADS = {'left': ((-192, -192), (0, 0), (192, 192)),
         'right': ((-192, 192), (0, 0), (192, -192))}


def fit(points):
    n = len(points)
    sx, sy = sum(x for x, _ in points), sum(y for _, y in points)
    sxx, sxy = sum(x * x for x, _ in points), sum(x * y for x, y in points)
    den = n * sxx - sx * sx
    need(den > 0, 'fit-underdetermined')
    a = round(Fraction(256 * (n * sxy - sx * sy), den))
    b = round(Fraction(sy, n) - Fraction(a * sx, 256 * n))
    return [a, b]


def acquire(revised=False):
    examples = REVISED_CORE if revised else CORE
    heads = {}
    for name, points in HEADS.items():
        w, b = fit(points)
        heads[name] = {'w': [w], 'mask': [1], 'b': b}
    return encode({'schema': 'quantized-binder-v1', 'width': 1,
                   'transitions': {name: [fit(points)] for name, points in examples.items()},
                   'heads': heads})


def grow(raw):
    obj = decode(raw)
    need(obj['width'] == 1, 'one-growth-stage')
    obj['width'] = 2
    for name, pairs in obj['transitions'].items():
        pairs.append(fit(AUX[name]))
    for head in obj['heads'].values():
        head['w'].append(0)
        head['mask'].append(0)
    return encode(obj)


def activate_aux(raw):
    # A declared post-growth parameter proposal, NOT a measured optimizer run.
    obj = decode(raw)
    need(obj['width'] == 2, 'aux-width')
    obj['heads']['left']['w'][1] = 4
    obj['heads']['right']['w'][1] = -4
    for head in obj['heads'].values():
        head['mask'][1] = 1
    return encode(obj)
