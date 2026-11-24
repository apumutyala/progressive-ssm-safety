"""Sufficient free next-byte bound conditional on a fitted final hidden box.

Supply a box per block enclosing every final hidden state after the *same*
ordinary suffix, for example from the actual monotone certified recurrence run
from -1 and +1 at the same shape. Supply the fitted raw dense weights/biases
as Python lists of their exact FP32 values. No text, model IO or tensors are
required here. The query must be unbound and the protected records unchanged.

Bound the REAL function of those hidden values: affine mix, tanh, affine
vocabulary head, per-block clipping, then clipped block additions in order.
Every coefficient and affine operation is evaluated using exact rationals.
Decimal's correctly rounded exp, with neighboring representable endpoints,
encloses real tanh. Different features' extrema need not be jointly reachable.
Positive lower margin proves a unique specified ordinary byte over the whole
box for this readout. Zero/negative is inconclusive. For actual FP32 readout
the same conclusion would require a separate uniform error bound below half
the certified gap. Two observed corner logits do not establish such an error
bound. No conclusion here applies to sampling or a different suffix.
"""
from decimal import Context, Decimal, ROUND_HALF_EVEN
from fractions import Fraction
import math


def _finite(x):
    if type(x) not in (int, float) or not math.isfinite(x):
        raise ValueError("Expected finite binary floating-point weights or endpoints")
    return Fraction(x)


def _clip(x, limit):
    return max(-limit, min(limit, x))


def _affine(weights, biases, box):
    result = []
    for row, bias in zip(weights, biases, strict=True):
        lower = upper = _clip(_finite(bias), 1)
        for raw, (left, right) in zip(row, box, strict=True):
            weight = _clip(_finite(raw), 1)
            if weight < 0:
                left, right = right, left
            lower += weight * left
            upper += weight * right
        result.append((lower, upper))
    return result


def _tanh_edge(x, upper):
    if not x:
        return Fraction(0)
    # exp and (e-1)/(e+1) are increasing. Enclose division rounding in 2*x,
    # then exp rounding. Evaluate the ratio exactly, including near zero.
    ctx = Context(prec=40, rounding=ROUND_HALF_EVEN)
    edge = ctx.next_plus if upper else ctx.next_minus
    arg = edge(ctx.divide(Decimal(2 * x.numerator), Decimal(x.denominator)))
    exp_edge = Fraction(edge(ctx.exp(arg)))
    return (exp_edge - 1) / (exp_edge + 1)


def _down(x):
    value = float(x)
    return math.nextafter(value, -math.inf) if Fraction(value) > x else value


def real_suffix_box(retentions, drives):
    """Optional real recurrence box, driven by a suffix's fixed effective rows.

    Start at [-1,1]^width. Treat the supplied gate/drive values as exact model
    constants. Per step, enclose the real affine update and round the bounds
    outward to binary64. This does not model the FP32 multiply/add sequence.
    For executed states, pass independently verified actual endpoint boxes to
    bound_expected_byte instead.
    """
    lower = upper = None
    for gate_row, drive_row in zip(retentions, drives, strict=True):
        if lower is None:
            if not len(gate_row):
                raise ValueError("Empty state width")
            lower, upper = [-1.0] * len(gate_row), [1.0] * len(gate_row)
        for j, (a, u, lo, hi) in enumerate(zip(gate_row, drive_row, lower, upper, strict=True)):
            a, u = _finite(a), _finite(u)
            if not 0 <= a <= 1 or not -1 <= u <= 1:
                raise ValueError("Expected nonnegative bounded retention and bounded drive")
            injection = (1 - a) * u
            lower[j] = _down(a * Fraction(lo) + injection)
            upper[j] = -_down(-(a * Fraction(hi) + injection))
    if lower is None:
        raise ValueError("Expected a nonempty suffix")
    return lower, upper


def bound_expected_byte(expected_id, blocks):
    """Each block dict needs hidden_lower/upper, mix_weight/bias, head_weight/bias.

    IDs 0..7 are reserved, IDs 8..263 ordinary. Returns aggregates only.
    """
    if type(expected_id) is not int or not 8 <= expected_id < 264:
        raise ValueError("Expected one ordinary byte-token ID")
    total = None
    for block in blocks:
        hidden = [(_finite(lo), _finite(hi)) for lo, hi in
                  zip(block["hidden_lower"], block["hidden_upper"], strict=True)]
        if not hidden or any(not -1 <= lo <= hi <= 1 for lo, hi in hidden):
            raise ValueError("Expected an ordered certified hidden box inside [-1,1]")
        mix = _affine(block["mix_weight"], block["mix_bias"], hidden)
        if len(mix) != len(hidden):
            raise ValueError("Mix dimension differs from hidden width")
        transformed = [(_tanh_edge(lo, False), _tanh_edge(hi, True)) for lo, hi in mix]
        head = _affine(block["head_weight"], block["head_bias"], hidden + transformed)
        if len(head) != 264:
            raise ValueError("Expected the full byte vocabulary")
        head = [(_clip(lo, 16), _clip(hi, 16)) for lo, hi in head]
        if total is None:
            total = head
        else:
            total = [(_clip(a + c, 16), _clip(b + d, 16))
                     for (a, b), (c, d) in zip(total, head)]
    if total is None:
        raise ValueError("Expected at least one fitted block")
    # The unbound prior is -3. The replacement CONFIRM score lies strictly
    # below DEFER for every state, so DEFER already dominates that competitor.
    competitors = [hi for i, (_, hi) in enumerate(total) if i not in (4, expected_id)]
    margin = total[expected_id][0] - max(competitors)
    return {
        "real_readout_margin_lower": _down(margin),
        "real_readout_unique_expected_byte": margin > 0,
        "uniform_logit_error_must_be_strictly_below": _down(max(0, margin) / Fraction(2)),
    }
