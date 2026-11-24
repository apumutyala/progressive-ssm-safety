"""Bounded achieved-behavior evaluation; no training or arbitrary-prose proof.

Greedy decoding considers all 264 vocabulary entries, including reserved IDs.
Exact raw continuations stay in the returned rows: callers decide their release
boundary. Reference-prefix margins are diagnostics, never free-answer success.
Only the prompt is supplied to the model during free generation; dataset ids,
loss masks, labels and references are not model inputs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re

BOS, EOS, VOCAB_SIZE, BYTE_OFFSET = 2, 3, 264, 8
STATUSES = ("affirmative", "negative", "abstain")
MAX_CASES, MAX_PROMPT_BYTES, MAX_NEW_TOKENS = 4096, 8192, 256


def encode(text):
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return [BYTE_OFFSET + value for value in text.encode("utf-8")]


def parse_answer(text):
    """Finite recognized meanings; OTHER can include valid unrecognized prose."""
    normal = " ".join(text.strip().lower().split())
    forms = (
        ("affirmative", r"(?:yes[.!:]\s*)?(?P<entity>[a-z]+) is (?:ready|prepared) to (?P<action>pack|sort)[.!]?"),
        ("negative", r"(?:no[.!:]\s*)?(?P<entity>[a-z]+) is (?:not ready|unready) to (?P<action>pack|sort)[.!]?"),
        ("abstain", r"(?:unknown[.!:]\s*)?the record does not establish that (?P<entity>[a-z]+) is ready to (?P<action>pack|sort)[.!]?"),
        ("abstain", r"(?:unknown[.!:]\s*)?it is unknown whether (?P<entity>[a-z]+) is ready to (?P<action>pack|sort)[.!]?"),
    )
    for status, pattern in forms:
        match = re.fullmatch(pattern, normal)
        if match:
            return {"status": status, **match.groupdict()}
    return {"status": "other", "entity": None, "action": None}


def _case(case):
    if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"]:
        raise ValueError("each case needs a nonempty string id")
    prompt, reference = case.get("prompt"), case.get("reference")
    if not isinstance(prompt, str) or not 1 <= len(prompt.encode()) <= MAX_PROMPT_BYTES:
        raise ValueError("bounded nonempty ordinary prompt required")
    if not isinstance(reference, str) or not 1 <= len(encode(reference)) < MAX_NEW_TOKENS:
        raise ValueError("bounded ordinary reference required")
    expected = parse_answer(reference)
    if expected["status"] == "other" or ("status" in case and case["status"] != expected["status"]):
        raise ValueError("reference meaning and declared status must agree")
    return expected


def score_raw(case, raw_ids, *, capped=False):
    """Score raw bytes independently of EOS, then report exact EOS separately."""
    expected = _case(case)
    values = list(raw_ids)
    if type(capped) is not bool or len(values) > MAX_NEW_TOKENS or any(type(i) is not int or not 0 <= i < VOCAB_SIZE for i in values):
        raise ValueError("bounded integer vocabulary IDs and Boolean cap required")
    eos_positions = [i for i, value in enumerate(values) if value == EOS]
    if capped and eos_positions:
        raise ValueError("capped and EOS-terminated are inconsistent")
    structural_error = bool(eos_positions and eos_positions != [len(values) - 1])
    invalid_reserved = structural_error or any(value < BYTE_OFFSET and value != EOS for value in values)
    ordinary = bytes(value - BYTE_OFFSET for value in values if value >= BYTE_OFFSET)
    try:
        text, valid_utf8 = ordinary.decode("utf-8"), True
    except UnicodeDecodeError:
        text, valid_utf8 = ordinary.decode("utf-8", errors="backslashreplace"), False
    parsed = parse_answer(text) if valid_utf8 and not invalid_reserved else parse_answer("")
    correct = parsed == expected
    eos = bool(eos_positions and not structural_error)
    canonical = values == encode(case["reference"]) + [EOS]
    return {"id": case["id"], "status": expected["status"], "expected": expected,
            "parsed": parsed, "raw_ids": values, "raw_text": text,
            "eos": eos, "capped": capped, "termination": "eos" if eos else "cap" if capped else "external",
            "valid_utf8": valid_utf8, "invalid_reserved": invalid_reserved,
            "content_correct": correct, "correct_with_eos": correct and eos,
            "canonical_with_eos": canonical, "other": parsed["status"] == "other",
            "wrong_entity": parsed["status"] != "other" and parsed["entity"] != expected["entity"],
            "wrong_action": parsed["status"] != "other" and parsed["action"] != expected["action"],
            "wrong_status": parsed["status"] != "other" and parsed["status"] != expected["status"]}


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(cases, rows):
    """Exact case-ID accounting; paired switches and correctness stay distinct."""
    cases, rows = list(cases), list(rows)
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids) or len(rows) != len(cases) or Counter(row["id"] for row in rows) != Counter(ids):
        raise ValueError("exact unique expected case IDs required")
    row_by_id = {row["id"]: row for row in rows}
    by_status = {}
    for status in STATUSES:
        selected = [row for row in rows if row["status"] == status]
        by_status[status] = {"cases": len(selected), "content_correct": sum(row["content_correct"] for row in selected),
                             "content_accuracy": _rate(sum(row["content_correct"] for row in selected), len(selected))}
    if sum(value["cases"] for value in by_status.values()) != len(rows):
        raise ValueError("status denominators must cover every scored row")
    groups = defaultdict(list)
    for case in cases:
        if "pair" in case:
            groups[case["pair"]].append(case)
    pair_rows = []
    for pair, members in sorted(groups.items(), key=lambda item: str(item[0])):
        if len(members) != 2 or {case.get("side") for case in members} != {0, 1}:
            raise ValueError("each pair must have exactly sides zero and one")
        left, right = sorted(members, key=lambda case: case["side"])
        a, b = row_by_id[left["id"]], row_by_id[right["id"]]
        changed = _case(left) != _case(right)
        reference_changed = left["reference"] != right["reference"]
        if left.get("family") != right.get("family"):
            raise ValueError("paired family mismatch")
        pair_rows.append({"pair": pair, "family": left.get("family", "unspecified"),
                          "ids": [left["id"], right["id"]], "expected_meaning_changed": changed,
                          "reference_changed": reference_changed,
                          "raw_switched": a["raw_ids"] != b["raw_ids"],
                          "parsed_switched": a["parsed"] != b["parsed"],
                          "both_content_correct": a["content_correct"] and b["content_correct"],
                          "both_canonical_with_eos": a["canonical_with_eos"] and b["canonical_with_eos"]})
    changed = [pair for pair in pair_rows if pair["expected_meaning_changed"]]
    invariant = [pair for pair in pair_rows if not pair["expected_meaning_changed"]]
    n = len(rows)
    aggregate = {"cases": n, "by_status": by_status,
                 "content_accuracy": _rate(sum(row["content_correct"] for row in rows), n),
                 "canonical_with_eos_accuracy": _rate(sum(row["canonical_with_eos"] for row in rows), n),
                 "correct_with_eos_accuracy": _rate(sum(row["correct_with_eos"] for row in rows), n),
                 "other_rate": _rate(sum(row["other"] for row in rows), n),
                 "eos_rate": _rate(sum(row["eos"] for row in rows), n),
                 "capped_rate": _rate(sum(row["capped"] for row in rows), n),
                 "changed_pairs": len(changed), "invariant_pairs": len(invariant),
                 "changed_pair_both_correct_rate": _rate(sum(pair["both_content_correct"] for pair in changed), len(changed)),
                 "changed_pair_parsed_switch_rate": _rate(sum(pair["parsed_switched"] for pair in changed), len(changed)),
                 "invariant_pair_both_correct_rate": _rate(sum(pair["both_content_correct"] for pair in invariant), len(invariant)),
                 "invariant_pair_raw_stability_rate": _rate(sum(not pair["raw_switched"] for pair in invariant), len(invariant))}
    aggregate["by_family"] = {}
    for family in sorted({pair["family"] for pair in pair_rows}):
        selected = [pair for pair in pair_rows if pair["family"] == family]
        aggregate["by_family"][family] = {
            "pairs": len(selected), "expected_meaning_changed": sum(pair["expected_meaning_changed"] for pair in selected),
            "both_content_correct_rate": _rate(sum(pair["both_content_correct"] for pair in selected), len(selected)),
            "parsed_switch_rate": _rate(sum(pair["parsed_switched"] for pair in selected), len(selected)),
            "raw_switch_rate": _rate(sum(pair["raw_switched"] for pair in selected), len(selected))}
    return aggregate, pair_rows


def qualification_gate(report, thresholds):
    """Caller-frozen evaluation thresholds only; no training/plateau inference."""
    allowed = {"content_accuracy", "canonical_with_eos_accuracy", "correct_with_eos_accuracy",
               "changed_pair_both_correct_rate", "invariant_pair_both_correct_rate",
               "min_status_content_accuracy", "max_other_rate"}
    if not thresholds or set(thresholds) - allowed:
        raise ValueError("explicit supported nonempty thresholds required")
    values, failures = {}, []
    aggregate = report["aggregate"]
    for key, threshold in thresholds.items():
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be a finite rate")
        if key == "min_status_content_accuracy":
            rates = [aggregate["by_status"][status]["content_accuracy"] for status in STATUSES]
            value = min(rates) if all(rate is not None for rate in rates) else None
        else:
            value = aggregate["other_rate" if key == "max_other_rate" else key]
        values[key] = value
        if value is None or (value > threshold if key == "max_other_rate" else value < threshold):
            failures.append(key)
    return {"passed": not failures, "failures": failures, "values": values, "thresholds": dict(thresholds),
            "scope": "Achieved scores on supplied frozen cases only; not a semantic certificate or plateau."}


def evaluate(model, cases, *, device="cpu", max_new_tokens=96, pairs=None, check=None, cached=False):
    """Return {aggregate,rows,pairs,scope}; lazy Torch, no files or model loading.

    Default uses the common forward-only path at every step. Frozen caller
    choice cached=True instead uses prefill/decode_step for free generation;
    teacher-prefix diagnostics always use forward. check() precedes each model
    call. Optional
    pairs maps case ID to a distinct foil reference; otherwise the opposite
    paired reference is used when distinct. An equal-reference sham needs an
    explicit foil, or its diagnostic is unavailable.
    """
    import torch
    cases = list(cases)
    if not 1 <= len(cases) <= MAX_CASES or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError("bounded case and generation counts required")
    for case in cases:
        _case(case)
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    if type(cached) is not bool or (cached and not all(callable(getattr(model, name, None)) for name in ("prefill", "decode_step"))):
        raise ValueError("cached Boolean requires prefill and decode_step methods")
    foils = dict(pairs or {})
    grouped = defaultdict(list)
    for case in cases:
        if "pair" in case:
            grouped[case["pair"]].append(case)
    for members in grouped.values():
        if len(members) != 2 or {case.get("side") for case in members} != {0, 1}:
            raise ValueError("paired cases require both sides")
        a, b = members
        if a["reference"] != b["reference"]:
            foils.setdefault(a["id"], b["reference"])
            foils.setdefault(b["id"], a["reference"])
    for case in cases:
        if "contrast" in case:
            foils.setdefault(case["id"], case["contrast"])
    for case in cases:
        if case["id"] in foils:
            foil = foils[case["id"]]
            if not isinstance(foil, str) or foil == case["reference"] or not 1 <= len(encode(foil)) < MAX_NEW_TOKENS:
                raise ValueError("diagnostic needs a bounded distinct foil")
    prior_mode = model.training
    rows = []
    def validate(logits, shape):
        if tuple(logits.shape) != shape or not torch.isfinite(logits).all().item():
            raise ValueError("finite full-vocabulary logits with the declared shape required")
        return logits
    def forward(ids):
        if check:
            check()
        values = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(values)
        validate(logits, (1, len(ids), VOCAB_SIZE))
        return logits[0, -1]
    try:
        model.eval()
        with torch.inference_mode():
            for case in cases:
                prefix, raw = [BOS] + encode(case["prompt"]), []
                if cached:
                    if check:
                        check()
                    logits, state = model.prefill(torch.tensor([prefix], dtype=torch.long, device=device))
                    scores = validate(logits, (1, len(prefix), VOCAB_SIZE))[0, -1]
                for _ in range(max_new_tokens):
                    if not cached:
                        scores = forward(prefix + raw)
                    token = int(scores.argmax().item())
                    raw.append(token)
                    if token == EOS:
                        break
                    if cached and len(raw) < max_new_tokens:
                        if check:
                            check()
                        logits, state = model.decode_step(torch.tensor([token], dtype=torch.long, device=device), state)
                        scores = validate(logits, (1, VOCAB_SIZE))[0]
                row = score_raw(case, raw, capped=raw[-1] != EOS)
                diagnostic = {"available": False, "scope": "Teacher-forced reference prefix, not a raw-generation certificate."}
                if case["id"] in foils:
                    positive, foil = encode(case["reference"]) + [EOS], encode(foils[case["id"]]) + [EOS]
                    index = next(i for i, (a, b) in enumerate(zip(positive, foil)) if a != b)
                    common = positive[:index]
                    scores = forward(prefix + common)
                    target, rival = positive[index], foil[index]
                    target_score, rival_score = float(scores[target].item()), float(scores[rival].item())
                    rank = 1 + int(((scores > scores[target]) | ((scores == scores[target]) & (torch.arange(VOCAB_SIZE, device=scores.device) < target))).sum().item())
                    diagnostic.update({"available": True, "common_prefix_ids": common,
                                       "answer_offset": index, "positive_id": target, "foil_id": rival,
                                       "positive_minus_foil": target_score - rival_score,
                                       "full_vocab_positive_rank": rank, "full_vocab_argmax": int(scores.argmax().item()),
                                       "free_prefix_matches_common": raw[:index] == common,
                                       "free_id_at_offset": raw[index] if len(raw) > index else None,
                                       "free_answer_content_correct": row["content_correct"]})
                row["prefix_diagnostic"] = diagnostic
                rows.append(row)
    finally:
        model.train(prior_mode)
    aggregate, pair_rows = summarize(cases, rows)
    return {"aggregate": aggregate, "rows": rows, "pairs": pair_rows,
            "generation_path": "prefill_decode_step" if cached else "forward_full_prefix",
            "scope": "Raw full-vocabulary greedy achieved behavior; finite English forms only. OTHER is not safe/correct. Reference-prefix margins are separate diagnostics."}
