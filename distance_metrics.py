"""Token- and field-level diagnostics on caller-supplied sequences."""


def first_error(prediction, gold):
    for index, (actual, expected) in enumerate(zip(prediction, gold)):
        if actual != expected:
            return str(index)
    return "exact" if len(prediction) == len(gold) else str(min(len(prediction), len(gold)))


def correct_fields(prediction, gold, positions):
    return all(len(prediction) > i and prediction[i] == gold[i] for i in positions)
