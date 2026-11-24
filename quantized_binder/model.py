"""Actual learned quantized recurrence, independent of abstract spec/checker.

Point execution uses Fraction's exact rounding, separate from interval rnd().
Interval readout is part of this module, not an after-the-fact authority filter.
"""
from dataclasses import dataclass
from fractions import Fraction
import intervals as iv
from wire import decode, need


def point_step(h, a, b, error=0):
    need(type(h) is int and -256 <= h <= 256 and type(error) is int
         and -1 <= error <= 1, 'point-state-or-error')
    value = round(Fraction(a * h, 256)) + b
    value = min(256, max(-256, value))
    value = 0 if -32 <= value <= 32 else value
    return min(256, max(-256, value + error))


def point_score(hidden, weights, mask, bias):
    return round(Fraction(sum(w * h for w, h, live in zip(weights, hidden, mask) if live), 256)) + bias


@dataclass(frozen=True)
class Program:
    width: int
    transitions: tuple
    heads: tuple

    @staticmethod
    def from_admitted(raw):
        # Not itself an admission API: owner must call independent checker first.
        obj = decode(raw)
        return Program(obj['width'],
                       tuple((name, tuple(tuple(pair) for pair in pairs))
                             for name, pairs in obj['transitions'].items()),
                       tuple((name, tuple(h['w']), tuple(h['mask']), h['b'])
                             for name, h in obj['heads'].items()))

    def initial(self):
        return (0,) * self.width, ((-1, 1),) * self.width

    def step(self, hidden, box, token, errors):
        need(type(token) is str and token in dict(self.transitions), 'context')
        need(type(hidden) is tuple and type(box) is tuple and type(errors) is tuple
             and len(hidden) == len(box) == len(errors) == self.width, 'state-width')
        for h, region in zip(hidden, box):
            need(type(h) is int and iv.valid(region) and region[0] <= h <= region[1]
                 and iv.inside(region, iv.FULL), 'state-enclosure')
        params = dict(self.transitions)[token]
        new_hidden = tuple(point_step(h, a, b, error)
                           for h, (a, b), error in zip(hidden, params, errors))
        new_box = tuple(iv.advance(region, a, b) for region, (a, b) in zip(box, params))
        need(all(low <= h <= high for h, (low, high) in zip(new_hidden, new_box)), 'point-outside-certified-box')
        return new_hidden, new_box

    def select(self, hidden, box, query):
        need(type(query) is str, 'query-type')
        choices = {name: (weights, mask, bias) for name, weights, mask, bias in self.heads}
        need(query in choices, 'query')
        weights, mask, bias = choices[query]
        score = point_score(hidden, weights, mask, bias)
        bounds = iv.readout(box, weights, mask, bias)
        need(bounds[0] <= score <= bounds[1], 'readout-enclosure')
        return iv.classify(bounds), score, bounds
