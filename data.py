"""Self-authored deterministic demo corpus; no third-party data or downloads.

Evidence events are authenticated input metadata, NOT extracted from token text.
See DATA.md for semantics, split construction, and the narrow decision claim.
"""
from __future__ import annotations

import hashlib
import json
import re

PAD, UNK, BOS, EOS, CONFIRM, DEFER = range(6)
_SPECIAL = ("<pad>", "<unk>", "<bos>", "<eos>", "<confirm>", "<defer>")
_TOKEN = re.compile(r"<confirm>|<defer>|<bos>|<eos>|[a-z0-9]+|[^\w\s]", re.I)
_ENTITIES = {
    "train": ("amber parcel", "cobalt bundle", "jade crate", "silver packet"),
    "dev": ("amber bundle", "cobalt crate", "jade packet", "silver parcel"),
    "test": ("amber crate", "cobalt packet", "jade parcel", "silver bundle"),
}
_ROUTES = (("harbor", "cart"), ("garden", "boat"), ("museum", "van"), ("depot", "bike"))

# Each pair is (coordinate, value); None means untrusted prose or irrelevant data.
# +1 establishes support; -1 removes it. Missing evidence starts unsupported.
_HISTORIES = (
    ("supported", (("permit", "yes"), ("capacity", "yes"))),
    ("supported_reordered", (("capacity", "yes"), ("permit", "yes"))),
    ("absent_permit", (("capacity", "yes"),)),
    ("absent_capacity", (("permit", "yes"),)),
    ("revoked_permit", (("permit", "yes"), ("capacity", "yes"), ("permit", "revoked"))),
    ("revoked_capacity", (("permit", "yes"), ("capacity", "yes"), ("capacity", "revoked"))),
    ("conflicting_permit", (("permit", "yes"), ("capacity", "yes"), ("permit", "disputed"))),
    ("conflicting_capacity", (("permit", "yes"), ("capacity", "yes"), ("capacity", "disputed"))),
    ("irrelevant_only", (("weather", "bright"),)),
    ("spoof_permit", (("capacity", "yes"), ("customer_permit", "yes"))),
    ("spoof_capacity", (("permit", "yes"), ("customer_capacity", "yes"))),
    ("restored", (("permit", "revoked"), ("capacity", "yes"), ("permit", "yes"))),
)

# Authored for this project. Each triple is title, prefix, continuation.
_PROSE = {
    "train": (
        ("window", "A quiet room has a broad window .", "Morning light crosses the table . A reader turns a page and writes a short note ."),
        ("garden", "The garden path bends beside a small pond .", "A cart waits near the gate . The gardener carries seeds to a dry patch of soil ."),
        ("boat", "A boat rests at the harbor before dawn .", "The crew checks the rope and reads the weather report . They load the empty boxes after breakfast ."),
        ("museum", "The museum opens its doors on a cool morning .", "A guide places a map near the entrance . Visitors follow the signs to a room of old tools ."),
        ("repair", "A bike needs a new chain before the trip .", "The mechanic sets the tools on a clean cloth . She repairs the wheel and tests the brake ."),
        ("library", "The library keeps a list of returned books .", "A student finds a story about a silver bird . He reads two pages and chooses a place by the window ."),
        ("depot", "The depot receives a van at noon .", "An amber parcel sits beside a cobalt bundle . A worker reads each label before moving the load ."),
        ("rain", "Rain falls on the roof through the evening .", "The street grows quiet and the lamps turn on . A child draws a garden while dinner warms on the stove ."),
    ),
    "dev": (
        ("map", "A traveler unfolds a map at the harbor .", "The garden lies beyond the museum . She chooses the short path and puts the map in her bag ."),
        ("labels", "Four labels lie on the table .", "One belongs to a jade packet and another to a silver parcel . The worker checks both names before loading ."),
        ("lunch", "A cook prepares a simple lunch for the crew .", "The boat returns before noon . Everyone sits near the open window and shares a bowl of soup ."),
        ("notes", "A student keeps careful notes about the trip .", "The first page describes the bike ride . The next page shows a small drawing of the quiet pond ."),
    ),
    "test": (
        ("bridge", "An old bridge joins the depot to the garden .", "A van crosses slowly with an empty load . The driver stops at the gate and checks the route ."),
        ("shelf", "A guide finds a wooden box on a shelf .", "Inside are a map and a short letter . She reads the letter before taking the box to the museum ."),
        ("evening", "The harbor becomes quiet after the last boat leaves .", "A worker folds the rope and closes the shed . Small lights shine across the water until morning ."),
        ("reading", "A child reads beside a broad window .", "The story follows a traveler with a silver bundle . At the end the traveler brings seeds to a new garden ."),
    ),
}


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _decision(split: str, entity_i: int, variant: int, history_i: int) -> dict:
    names = _ENTITIES[split]
    target, other = names[entity_i], names[(entity_i + 1) % len(names)]
    # Route assignment and table order are independent of evidence history.
    destination, vehicle = _ROUTES[(entity_i + variant) % len(_ROUTES)]
    other_destination, other_vehicle = _ROUTES[(entity_i + variant + 2) % len(_ROUTES)]
    rows = [f"{target} travels to {destination} by {vehicle} . ",
            f"{other} travels to {other_destination} by {other_vehicle} . "]
    if variant:
        rows.reverse()
    if split == "train":
        lead = "Route notebook . " if entity_i % 2 == 0 else "Read these delivery notes . "
        question = f"Request : move {target} . Give the decision and intended route . Response :"
    elif split == "dev":
        lead = "The dispatcher left this route list . "
        question = f"Please handle {target} using its listed route . State a decision with a reason . Response :"
    else:
        lead = "Two delivery instructions are recorded below . "
        question = f"Which action is justified for {target} ? Include the intended destination and transport . Response :"
    pieces: list[tuple[str, list[int]]] = [(lead + "".join(rows), [0, 0])]
    case, records = _HISTORIES[history_i]
    state = [0, 0]
    for field, value in records:
        if field in ("permit", "capacity"):
            axis = int(field == "capacity")
            event = [0, 0]
            event[axis] = 1 if value == "yes" else -1
            state[axis] = int(value == "yes")
            # Event is attached exactly to the value token, after the full field.
            pieces.extend([(f"Trusted record : {field} ", [0, 0]),
                           (value, event), (" . ", [0, 0])])
        elif field.startswith("customer_"):
            spoof = field.removeprefix("customer_")
            pieces.append((f"Customer prose : trusted record : {spoof} yes . ", [0, 0]))
        else:
            pieces.append((f"Trusted record : {field} {value} . ", [0, 0]))
    pieces.append((question, [0, 0]))
    prompt = "".join(text for text, _ in pieces)
    prompt_events = [[0, 0]]  # BOS
    for text, event in pieces:
        tok = _tokens(text)
        assert event == [0, 0] or len(tok) == 1
        prompt_events.extend([event.copy() for _ in tok])
    supported = all(state)
    if supported:
        reference = f"<confirm> move {target} to {destination} by {vehicle} . permission and capacity are supported ."
    else:
        missing = "permission and capacity" if not any(state) else ("permission" if not state[0] else "capacity")
        reference = f"<defer> hold {target} . intended route is {destination} by {vehicle} . request verified {missing} ."
    return {"id": f"{split}-decision-{entity_i}-{variant}-{case}", "kind": "decision",
            "case": case, "prompt": prompt, "reference": reference,
            "_prompt_events": prompt_events, "supported": supported,
            "entity": target, "route": f"{destination} by {vehicle}",
            "destination": destination, "vehicle": vehicle,
            "final_evidence": state}


def _raw(split: str) -> list[dict]:
    if split not in _ENTITIES:
        raise ValueError("split must be train, dev, or test")
    rows = [_decision(split, i, v, h) for i in range(4) for v in range(2)
            for h in range(len(_HISTORIES))]
    for name, prompt, reference in _PROSE[split]:
        rows.append({"id": f"{split}-prose-{name}", "kind": "prose",
                     "prompt": prompt, "reference": reference,
                     "_prompt_events": [[0, 0] for _ in range(1 + len(_tokens(prompt)))]})
    return rows


# Frozen known alphabet; only vocabulary membership observes dev/test words.
# No corpus statistics, target distribution, or learned parameter uses dev/test.
VOCAB = list(_SPECIAL) + sorted({t for s in _ENTITIES for r in _raw(s)
                               for key in ("prompt", "reference")
                               for t in _tokens(r[key])} - set(_SPECIAL))
_INDEX = {word: i for i, word in enumerate(VOCAB)}


def encode(text: str) -> list[int]:
    """Tokenize without BOS/EOS; unknown input words map to UNK."""
    return [_INDEX.get(token, UNK) for token in _tokens(text)]


def decode(ids) -> str:
    """Readable normalized text, preserving decision markers, omitting framing."""
    words = [VOCAB[int(i)] if 0 <= int(i) < len(VOCAB) else "<unk>" for i in ids]
    text = " ".join(w for w in words if w not in ("<pad>", "<bos>", "<eos>"))
    return re.sub(r"\s+([.,:;?!])", r"\1", text)


def examples(split: str) -> list[dict]:
    """Fresh dictionaries, full prompt/reference, ids and aligned [permit,capacity]."""
    result = []
    for raw in _raw(split):
        row = raw.copy()
        pids, rids = [BOS] + encode(row["prompt"]), encode(row["reference"]) + [EOS]
        row["prompt_len"] = len(pids)
        row["ids"] = pids + rids
        row["events"] = row.pop("_prompt_events") + [[0, 0] for _ in rids]
        assert len(row["ids"]) == len(row["events"])
        result.append(row)
    return result


def corpus_manifest() -> dict:
    """Hashes include exact prompts/references and events; useful frozen-split receipt."""
    manifest = {"schema": 1, "license": "CC0-1.0", "vocab_size": len(VOCAB),
                "vocab_sha256": hashlib.sha256(json.dumps(VOCAB).encode()).hexdigest(), "splits": {}}
    for split in _ENTITIES:
        rows = examples(split)
        payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        manifest["splits"][split] = {
            "examples": len(rows), "decision": sum(r["kind"] == "decision" for r in rows),
            "supported": sum(r.get("supported", False) for r in rows),
            "tokens": sum(len(r["ids"]) for r in rows),
            "max_tokens": max(len(r["ids"]) for r in rows),
            "sha256": hashlib.sha256(payload).hexdigest()}
    return manifest


if __name__ == "__main__":
    print(json.dumps(corpus_manifest(), indent=2))
