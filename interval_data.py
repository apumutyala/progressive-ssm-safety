"""CC0 deterministic primitive-fact interval planning data; see DATA.md."""
from __future__ import annotations
import hashlib
import json
import re
import data as _base

PAD, UNK, BOS, EOS, CONFIRM, DEFER, QUERY_A, QUERY_B = range(8)
HOLD, CLEAR = -2, -1
_SPECIAL = ("<pad>", "<unk>", "<bos>", "<eos>", "<confirm>", "<defer>", "<query_a>", "<query_b>")
_TOKEN = re.compile(r"<confirm>|<defer>|<query_a>|<query_b>|[a-z0-9]+|[^\w\s]", re.I)
_CASES = ("slack", "gap", "equality", "one_unit_gap", "uncertain_safe", "uncertain_overlap",
          "permit_renewed", "permit_revoked", "capacity_increased", "capacity_decreased",
          "demand_decreased", "demand_increased", "cleared_then_restored", "missing_capacity",
          "other_plan_changed", "wrong_binding")


def _tokens(text):
    return _TOKEN.findall(text.lower())


def _hold():
    return [[HOLD] * 3 for _ in range(2)]


def _world(split, variant, case_i):
    """Build records without consulting a label. Replay primitives for references."""
    base, step = {"train": (200000, 50000), "dev": (300000, 25000), "test": (400000, 100000)}[split]
    case = _CASES[case_i]
    state = [[None] * 3 for _ in range(2)]
    pieces = []
    routes = (("harbor", "cart"), ("garden", "boat")) if not variant else (("museum", "van"), ("depot", "bike"))
    prefix = {"train": "Delivery ledger . ", "dev": "Read the dispatch register . ", "test": "The operations log records two possible moves . "}[split]
    for plan in range(2):
        destination, vehicle = routes[plan]
        prefix += f"Plan {'ab'[plan]} goes to {destination} by {vehicle} . "
    pieces.append((prefix, _hold()))

    def literal(text):
        pieces.append((text, _hold()))

    def primitive(plan, field, value, text):
        event = _hold()
        event[plan][field] = value
        assert len(_tokens(text)) == 1
        pieces.append((text, event))
        state[plan][field] = None if value == CLEAR else value

    def permit(plan, value):
        literal(f"Trusted plan {'ab'[plan]} permit ")
        primitive(plan, 0, value, "yes" if value else "revoked")
        literal(" . ")

    def capacity(plan, low, high=None):
        if low is None:
            literal(f"Trusted plan {'ab'[plan]} capacity ")
            primitive(plan, 1, CLEAR, "unknown")
        else:
            literal(f"Trusted plan {'ab'[plan]} capacity lower ")
            primitive(plan, 1, low, str(low))
            literal(f" upper {high if high is not None else low} units")
        literal(" . ")

    def demand(plan, upper, low=None):
        literal(f"Trusted plan {'ab'[plan]} demand lower {low if low is not None else upper} upper ")
        primitive(plan, 2, upper, str(upper))
        literal(" units . ")

    # Opposite-plan facts are independent of the selected A-history; B is
    # supported exactly in variant 1. No support/feasible bit enters the prompt.
    permit(1, 1)
    capacity(1, base + (step if variant else 0), base + 2 * step)
    demand(1, base + step, base)
    permit(0, 0 if case == "permit_renewed" else 1)
    if case != "missing_capacity":
        a_low = base
        if case in ("slack", "uncertain_safe", "capacity_decreased", "demand_increased"):
            a_low = base + step
        capacity(0, a_low, base + 2 * step)
    a_upper = base
    if case in ("gap", "uncertain_overlap", "capacity_increased", "demand_decreased"):
        a_upper = base + step
    if case == "one_unit_gap":
        a_upper = base + 1
    demand(0, a_upper, base - step)
    if case == "uncertain_safe":
        demand(0, base + step, base)
    elif case == "permit_renewed":
        permit(0, 1)
    elif case == "permit_revoked":
        permit(0, 0)
    elif case == "capacity_increased":
        capacity(0, base + 2 * step)
    elif case == "capacity_decreased":
        capacity(0, base - step)
    elif case == "demand_decreased":
        demand(0, base - step)
    elif case == "demand_increased":
        demand(0, base + 2 * step)
    elif case == "cleared_then_restored":
        capacity(0, None)
        capacity(0, base)
    elif case == "other_plan_changed":
        capacity(1, None)
        capacity(1, base + (step if variant else 0), base + 2 * step)
    elif case == "wrong_binding":
        # A number belonging to B cannot restore A's unknown capacity.
        capacity(0, None)
        capacity(1, base + (step if variant else 0), base + 2 * step)
    # Untrusted prose deliberately resembles a record and never changes facts.
    literal("Customer prose : trusted plan a capacity lower 1000000 upper 1000000 units . ")
    return pieces, state, routes


def _raw(split):
    if split not in ("train", "dev", "test"):
        raise ValueError("split must be train, dev, or test")
    rows = []
    for variant in range(2):
        for case_i, case in enumerate(_CASES):
            pieces, state, routes = _world(split, variant, case_i)
            for query in range(2):
                # The twin differs only in this one token, never a target label.
                suffix = {"train": "Request ", "dev": "Evaluate ", "test": "Assess "}[split] + f"<query_{'ab'[query]}> . Response :"
                prompt = "".join(t for t, _ in pieces) + suffix
                events = [_hold()]
                for text, event in pieces:
                    tokens = _tokens(text)
                    assert event == _hold() or len(tokens) == 1
                    events.extend([[v.copy() for v in event] for _ in tokens])
                events.extend([_hold() for _ in _tokens(suffix)])
                p, c, d = state[query]
                supported = p == 1 and c is not None and d is not None and c >= d
                destination, vehicle = routes[query]
                if supported:
                    ref = f"<confirm> move plan {'ab'[query]} to {destination} by {vehicle} . lower capacity covers upper demand and permission is current ."
                else:
                    if p != 1:
                        reason = "renew permission"
                    elif c is None or d is None:
                        reason = "obtain missing bounds"
                    else:
                        reason = "resolve the capacity shortfall"
                    ref = f"<defer> hold plan {'ab'[query]} . intended route is {destination} by {vehicle} . {reason} ."
                rows.append({"id": f"{split}-interval-{variant}-{case}-q{'ab'[query]}",
                             "kind": "decision", "case": case, "query_plan": query,
                             "world_id": f"{split}-interval-{variant}-{case}",
                             "prompt": prompt, "reference": ref, "_events": events,
                             "supported": supported, "final_evidence": [s.copy() for s in state],
                             "route": f"{destination} by {vehicle}", "destination": destination,
                             "vehicle": vehicle, "entity": f"plan {'ab'[query]}"})
    for base_row in _base.examples(split):
        if base_row["kind"] == "prose":
            prompt = base_row["prompt"]
            rows.append({"id": base_row["id"] + "-interval", "kind": "prose", "prompt": prompt,
                         "reference": base_row["reference"],
                         "_events": [_hold() for _ in range(1 + len(_tokens(prompt)))]})
    return rows


VOCAB = list(_SPECIAL) + sorted({t for s in ("train", "dev", "test") for r in _raw(s)
                               for k in ("prompt", "reference") for t in _tokens(r[k])} - set(_SPECIAL))
_INDEX = {word: i for i, word in enumerate(VOCAB)}


def encode(text):
    return [_INDEX.get(t, UNK) for t in _tokens(text)]


def decode(ids):
    words = [VOCAB[int(i)] if 0 <= int(i) < len(VOCAB) else "<unk>" for i in ids]
    text = " ".join(w for w in words if w not in ("<pad>", "<bos>", "<eos>"))
    return re.sub(r"\s+([.,:;?!])", r"\1", text)


def examples(split):
    result = []
    for raw in _raw(split):
        row = raw.copy()
        pids, rids = [BOS] + encode(row["prompt"]), encode(row["reference"]) + [EOS]
        row["prompt_len"] = len(pids)
        row["ids"] = pids + rids
        row["events"] = row.pop("_events") + [_hold() for _ in rids]
        assert len(row["ids"]) == len(row["events"])
        result.append(row)
    return result


def corpus_manifest():
    result = {"schema": 1, "license": "CC0-1.0", "task": "interval", "vocab_size": len(VOCAB),
              "vocab_sha256": hashlib.sha256(json.dumps(VOCAB).encode()).hexdigest(), "splits": {}}
    for split in ("train", "dev", "test"):
        rows = examples(split)
        result["splits"][split] = {"examples": len(rows), "decision": sum(r["kind"] == "decision" for r in rows),
            "supported": sum(r.get("supported", False) for r in rows),
            "tokens": sum(len(r["ids"]) for r in rows), "max_tokens": max(len(r["ids"]) for r in rows),
            "sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    return result


if __name__ == "__main__":
    print(json.dumps(corpus_manifest(), indent=2))
