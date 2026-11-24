"""Exact signed interval arithmetic; integer bounds, no floating rounding.

Bounds use parameter tolerance +/-1 and state error +/-1 quantum. These are
declared uncertainty sets, not statistical confidence or a hardware-error proof.
"""
from wire import need

Q, DEAD, MARGIN, ERROR = 256, 32, 64, 1
REGIONS = ((-16, 16), (128, 256), (-256, -128))
FULL = (-Q, Q)


def valid(iv):
    return (type(iv) in (tuple, list) and len(iv) == 2 and
            all(type(v) is int for v in iv) and iv[0] <= iv[1])


def rnd(n):
    # Nearest integer, ties to even, including negative numerators.
    quotient, remainder = divmod(n, Q)
    return quotient + int(2 * remainder > Q or
                          (2 * remainder == Q and quotient % 2 != 0))


def clip(v):
    return min(Q, max(-Q, v))


def dead(v):
    return 0 if -DEAD <= v <= DEAD else v


def product(a, b):
    need(valid(a) and valid(b), 'interval-shape')
    values = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(values), max(values)


def advance(iv, a, b):
    lo, hi = product(iv, (a - 1, a + 1))
    lo, hi = rnd(lo) + b - 1, rnd(hi) + b + 1
    # clip and deadzone are monotone; endpoint images enclose the full interval.
    lo, hi = dead(clip(lo)), dead(clip(hi))
    return clip(lo - ERROR), clip(hi + ERROR)


def readout(box, weights, mask, bias):
    lo = hi = 0
    for iv, w, live in zip(box, weights, mask):
        if live:
            low, high = product(iv, (w - 1, w + 1))
            lo, hi = lo + low, hi + high
    return rnd(lo) + bias - 1, rnd(hi) + bias + 1


def classify(iv):
    need(valid(iv), 'score-interval')
    if iv[0] > MARGIN:
        return 1
    if iv[1] < -MARGIN:
        return 2
    return 0


def inside(iv, region):
    return region[0] <= iv[0] <= iv[1] <= region[1]
