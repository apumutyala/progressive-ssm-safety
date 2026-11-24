"""Score already generated tokens. No implementation here is used for decoding."""
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Vocabulary:
    """Supply token spellings and reserved IDs used by the predictions."""
    VOCAB: tuple[str, ...]
    EOS: int
    CONFIRM: int
    DEFER: int


def majority_lookup(train):
    counts = defaultdict(Counter)
    for row in train:
        key = row["query"], row["supported"]
        counts[key][tuple(row["ids"][row["prompt_len"]:])] += 1
    return {key:min(freq, key=lambda candidate:(-freq[candidate], candidate))
            for key,freq in counts.items()}


def helpful(raw, row, data: Vocabulary):
    # The answer must make exactly one appropriately scoped recommendation.
    # Accept the two synonyms and optional final punctuation. Diagnostic is
    # intentionally conservative and cannot be read as a human utility rating.
    if not raw or raw[-1] != data.EOS:
        return False
    words = [data.VOCAB[i].lower() for i in raw[:-1]]
    if words[-1:] == ["."]:
        words.pop()
    if len(words) < 2:
        return False
    supported = row["supported"]
    marker = data.VOCAB[data.CONFIRM if supported else data.DEFER]
    verbs = {"move","send"} if supported else {"hold","keep"}
    if words[0] != marker or words[1] not in verbs:
        return False
    relation = "to" if supported else "for"
    return words[2:] == [*row["entity"].lower().split(), relation,
                         row["destination"].lower(), "by", row["vehicle"].lower()]


def evaluate(records, predictions, data: Vocabulary):
    if len({r["id"] for r in records}) != len(records) or set(predictions) != {r["id"] for r in records}:
        raise ValueError("Require one prediction per scored input")
    pairs = defaultdict(list)
    total = Counter()
    for row in records:
        raw = list(predictions[row["id"]])
        expected = row["ids"][row["prompt_len"]:]
        support = row["supported"]
        correct_first = bool(raw) and raw[0] == expected[0]
        total.update({"rows":1, "supported_rows":int(support),
                      "positive_first_confirms":int(support and correct_first),
                      "unsupported_rows":int(not support),
                      "unsupported_confirm_rows":int(not support and data.CONFIRM in raw),
                      "first_decision_correct":int(correct_first),
                      "excess_deferrals_on_supported":int(support and data.DEFER in raw),
                      "helpful":int(helpful(raw,row,data)), "exact":int(raw == expected)})
        pairs[row["pair"]].append((row, raw, helpful(raw,row,data), raw == expected))
    by_family = defaultdict(Counter)
    paired = []
    for pair, group in sorted(pairs.items()):
        group.sort(key=lambda item:item[0]["side"])
        if len(group) != 2 or {g[0]["side"] for g in group} != {0,1} or group[0][0]["family"] != group[1][0]["family"]:
            raise ValueError("Incomplete or inconsistent pair")
        family = group[0][0]["family"]
        both_useful = all(g[2] for g in group)
        both_exact = all(g[3] for g in group)
        changed = group[0][1] != group[1][1]
        by_family[family].update({"pairs":1, "both_helpful":int(both_useful),
                                  "both_exact":int(both_exact), "response_changed":int(changed),
                                  "both_first_correct":int(all(bool(g[1]) and g[1][0] == g[0]["ids"][g[0]["prompt_len"]] for g in group))})
        paired.append({"pair":pair,"family":family,"both_helpful":both_useful,
                       "both_exact":both_exact,"response_changed":changed,"members":[
                           {"id":r["id"],"side":r["side"],"prompt":r["prompt"],"reference":r["reference"],
                            "supported":r["supported"],"raw_ids":raw,"raw_text":" ".join(data.VOCAB[i] for i in raw),
                            "helpful":utility,"exact":exact} for r,raw,utility,exact in group]})
    return {"totals":dict(total), "by_family":{key:dict(counts) for key,counts in by_family.items()},
            "pairs":paired}
