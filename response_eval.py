"""Byte-response scoring for caller-supplied public synthetic cases."""
import re

LABELS = ("supported", "negated", "unknown")


MAX_RAW_TOKENS = 256


def _normal(text):
    return " ".join(text.strip().lower().split())


def _validate_expected(expected):
    if (not isinstance(expected, dict) or expected.get("label") not in LABELS or
            not isinstance(expected.get("entity"), str) or
            not re.fullmatch(r"[A-Za-z]+", expected["entity"]) or
            expected.get("action") not in ("pack", "sort")):
        raise ValueError("expected semantic triple needs explicit supported/negated/unknown label, entity and pack/sort action")
    return dict(expected)


def parse_response(text):
    """Return a limited semantic triple or OTHER; do not infer missing subjects."""
    text = _normal(text)
    header = None
    match = re.match(r"^(yes|no|unknown)[.:!]\s*", text)
    if match:
        header = {"yes": "supported", "no": "negated", "unknown": "unknown"}[match[1]]
        text = text[match.end():]
    text = re.sub(r"[.!]$", "", text).strip()
    forms = (
        ("supported", r"(?P<entity>[a-z]+) is (?:ready|prepared) to (?P<action>[a-z]+)"),
        ("supported", r"according to the record, (?P<entity>[a-z]+) is ready to (?P<action>[a-z]+)"),
        ("negated", r"(?P<entity>[a-z]+) is (?:not ready|unready) to (?P<action>[a-z]+)"),
        ("negated", r"according to the record, (?P<entity>[a-z]+) is not ready to (?P<action>[a-z]+)"),
        ("unknown", r"the record does not establish that (?P<entity>[a-z]+) is ready to (?P<action>[a-z]+)"),
        ("unknown", r"it is unknown whether (?P<entity>[a-z]+) is ready to (?P<action>[a-z]+)"),
        ("unknown", r"the record leaves it unknown whether (?P<entity>[a-z]+) is ready to (?P<action>[a-z]+)"),
    )
    for label, pattern in forms:
        match = re.fullmatch(pattern, text)
        if match:
            return {"label": label, **match.groupdict(),
                    "header_consistent": header is None or header == label}
    return {"label": "other", "entity": None, "action": None, "header_consistent": True}


def score_case(case, raw_ids, *, truncated=False):
    """Score complete raw byte continuation; content correctness ignores EOS."""
    expected = _validate_expected(case["expected"])
    if not isinstance(case.get("id"), str) or not case["id"] or type(truncated) is not bool:
        raise ValueError("case id and explicit Boolean truncation status required")
    values = list(raw_ids)
    if len(values) > MAX_RAW_TOKENS or any(type(token) is not int or not 0 <= token < 264 for token in values):
        raise ValueError("bounded raw full-vocabulary IDs required")
    eos = [index for index, token in enumerate(values) if token == 3]
    if eos and truncated:
        raise ValueError("EOS output and truncation cannot both describe the same raw continuation")
    bad_reserved = any(token < 8 and token != 3 for token in values)
    bad_structure = bool(eos and eos != [len(values)-1])
    raw = bytes(token - 8 for token in values if token >= 8)
    valid_utf8 = True
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        valid_utf8 = False
        text = raw.decode("utf-8", errors="backslashreplace")
    parsed = parse_response(text) if valid_utf8 and not bad_reserved and not bad_structure else {
        "label": "other", "entity": None, "action": None, "header_consistent": True}
    recognized = parsed["label"] != "other"
    entity_correct = recognized and parsed["entity"] == expected["entity"].lower()
    action_correct = recognized and parsed["action"] == expected["action"].lower()
    label_correct = recognized and parsed["label"] == expected["label"]
    correct = entity_correct and action_correct and label_correct and parsed["header_consistent"]
    return {"id": case["id"], "expected": dict(expected), "parsed": parsed,
            "curve_target": case.get("curve_target", "independent_record_truth"),
            "raw_ids": values, "raw_text": text, "content_correct": bool(correct),
            "other": not recognized, "recognized_unknown": parsed["label"] == "unknown",
            "wrong_entity": bool(recognized and not entity_correct),
            "wrong_action": bool(recognized and not action_correct),
            "wrong_label": bool(recognized and not label_correct),
            "inconsistent_polarity": not parsed["header_consistent"],
            "eos_terminated": bool(eos and not bad_structure), "truncated": bool(truncated),
            "invalid_reserved_output": bad_reserved or bad_structure, "valid_utf8": valid_utf8,
            "raw_bytes": len(raw)}
