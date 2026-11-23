"""Paired public-fixture route swap on the frozen 1,024-update grown interval LM.

Use launch.py --mode route-probe on the Omen. Every held-out decision is paired
with a prompt in which the A/B destinations and vehicles exchange positions.
The query, receipts, and all other model input tokens remain identical. Both
versions are decoded from scratch, without a checker or teacher-forced output.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import time

START = time.monotonic()
ROOT = Path(__file__).resolve().parent
CHECKPOINT_SHA = "15bd11f0bcc6ad4059f48aff93de16f2774bd454d9b9af4f9abbfa189e348c6a"
TRAIN_SOURCES = {
    "experiment.py": "0cba7b98fda9504fd56939dd229f1da04002dc640c99b19e2632b6d28f179c78",
    "model.py": "61ca237e6296ddc79ce02fb457ae62087f9724517ffc1a8d4c8492df09229abf",
    "data.py": "9d301df4cdb8092520ec05b14b0c437f8afbfa8d5e57a9e39b27f9385a5cd0d3",
    "interval_model.py": "421f17b442105dea44fa44192f3c87be518a5f85d211b2edb8210c9296cc536a",
    "interval_data.py": "614b385c7ec9192c6595bc3c16e0d33e32cc89f11a49a5ed05bb641ca1560e33",
}
TRAIN_DATA_SHA = "3090fbd5ebabc8b480f2e5ca6e80985d2c39d9e28e749882731fa6cda0bfe115"
ROUTE = re.compile(r"\bPlan ([ab]) goes to ([a-z]+) by ([a-z]+) \.")
LIMIT = 24  # identical token cap to the historical held-out comparison


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def encoded(obj):
    return json.dumps(obj, separators=(",", ":"), allow_nan=False).encode()


def load_checkpoint(path, expected):
    if expected != CHECKPOINT_SHA or not 0 < path.stat().st_size <= 4 * 1024**2:
        raise ValueError("Only the prespecified digest and a checkpoint <=4 MiB are admitted")
    with path.open("rb") as stream:
        blob = stream.read(4 * 1024**2 + 1)
    if len(blob) > 4 * 1024**2 or digest(blob) != expected:
        raise ValueError("Checkpoint size or digest mismatch before deserialization")
    for name, wanted in TRAIN_SOURCES.items():
        if name != "experiment.py" and digest((ROOT / name).read_bytes()) != wanted:
            raise ValueError(f"Inference dependency differs from training source: {name}")
    import torch
    import interval_data as data
    from interval_model import IntervalModel

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    ck = torch.load(io.BytesIO(blob), weights_only=True, map_location="cpu")
    if (ck["format_version"] != 1 or ck["torch_version"] != str(torch.__version__)
            or ck["source_sha256"] != TRAIN_SOURCES or ck["dataset_sha256"] != TRAIN_DATA_SHA
            or ck["training_config"]["task"] != "interval"
            or (ck["step"], ck["blocks"], ck["width"]) != (1024, 2, 16)):
        raise ValueError("Historical training provenance or architecture mismatch")
    model = IntervalModel(len(data.VOCAB), 16, data.CONFIRM, data.DEFER, data.QUERY_A, data.QUERY_B)
    model.grow()
    wanted, actual = model.state_dict(), ck["model"]
    if set(wanted) != set(actual):
        raise ValueError("Checkpoint weight names mismatch")
    for name, target in wanted.items():
        loaded = actual[name]
        if (not isinstance(loaded, torch.Tensor) or loaded.shape != target.shape
                or loaded.dtype != torch.float32 or loaded.device.type != "cpu"
                or not torch.isfinite(loaded).all().item()):
            raise ValueError(f"Invalid stored model parameter {name}")
    model.load_state_dict(actual, strict=True)
    model.validate_parameters()
    return model.eval(), data


def routes(prompt):
    matches = list(ROUTE.finditer(prompt))
    if [match[1] for match in matches] != ["a", "b"]:
        raise ValueError("Expected exactly one A and one B route declaration")
    return matches


def swap(prompt):
    a, b = routes(prompt)
    replacements = [(own.span(i), other[i]) for own, other in [(a, b), (b, a)] for i in (2, 3)]
    for (start, end), word in sorted(replacements, reverse=True):
        prompt = prompt[:start] + word + prompt[end:]
    return prompt


def answer(prompt, ids, events, data):
    """Replay primitive receipts and derive route and reference from this exact prompt."""
    queries = [token for token in ids if token in (data.QUERY_A, data.QUERY_B)]
    if len(queries) != 1:
        raise ValueError("Expected one framed query per public prompt")
    plan = int(queries[0] == data.QUERY_B)
    fields = [data.CLEAR] * 3
    for receipt in events:
        for i, value in enumerate(receipt[plan]):
            if value != data.HOLD:
                fields[i] = value
    permit, lower, upper = fields
    supported = permit == 1 and lower >= 0 and upper >= 0 and lower >= upper
    label, destination, vehicle = routes(prompt)[plan].groups()
    route = f"{destination} by {vehicle}"
    if supported:
        prefix = f"<confirm> move plan {label} to {route} ."
        reason = "lower capacity covers upper demand and permission is current ."
    else:
        prefix = f"<defer> hold plan {label} . intended route is {route} ."
        reason = ("renew permission" if permit != 1 else "obtain missing bounds"
                  if min(lower, upper) < 0 else "resolve the capacity shortfall") + " ."
    return prefix, prefix + " " + reason, supported, plan


def make_pair(row, data):
    original = row["prompt"]
    swapped = swap(original)
    if swap(swapped) != original:
        raise ValueError("Swap did not invert byte-for-byte")
    events = row["events"][:row["prompt_len"]]
    pair = []
    for prompt in (original, swapped):
        ids = [data.BOS] + data.encode(prompt)
        if len(ids) != len(events) or len(ids) > 256 or data.UNK in ids:
            raise ValueError("Fixed-position input or vocabulary check failed")
        prefix, reference, supported, query = answer(prompt, ids, events, data)
        pair.append(dict(prompt=prompt, ids=ids, route_prefix=prefix, reference=reference,
                         supported=supported, query=query))
    left, right = pair
    changed = [i for i, (a, b) in enumerate(zip(left["ids"], right["ids"])) if a != b]
    if (left["ids"] != row["ids"][:row["prompt_len"]] or left["reference"] != row["reference"]
            or left["supported"] != row["supported"] or len(changed) != 4
            or left["query"] != right["query"] or left["supported"] != right["supported"]
            or left["route_prefix"] == right["route_prefix"]
            or any(v != data.HOLD for i in changed for p in events[i] for v in p)):
        raise ValueError("Counterfactual failed original-reference or matched-intervention checks")
    return pair, events, changed


def run_pair(model, pair, events, data, deadline):
    import torch

    ids = torch.tensor([side["ids"] for side in pair], dtype=torch.int64)
    receipts = torch.tensor([events, events], dtype=torch.int64)
    outputs = [[], []]
    with torch.inference_mode():
        all_logits, state = model.prefill(ids, receipts)
        if not torch.isfinite(all_logits).all().item():
            raise ValueError("Nonfinite prefill logits")
        logits = all_logits[:, -1]
        for _ in range(LIMIT):
            if time.monotonic() >= deadline - 1:
                raise TimeoutError("Route probe reached bounded worker reserve")
            if not torch.isfinite(logits).all().item():
                raise ValueError("Nonfinite decode logits")
            if state.query.tolist() != [side["query"] for side in pair]:
                raise ValueError("Protected query disagrees with framed input")
            if not torch.equal(state.records[0], state.records[1]):
                raise ValueError("Protected receipts differ between versions")
            for h in state.hidden:
                if not ((h >= -1) & (h <= 1)).all().item():
                    raise ValueError("Learned recurrence left the admitted finite interval")
            token = logits.argmax(-1)
            for i, t in enumerate(token.tolist()):
                if not outputs[i] or outputs[i][-1] != data.EOS:
                    outputs[i].append(t)
            if all(tokens[-1] == data.EOS for tokens in outputs):
                break
            logits, state = model.decode_step(token, state)
    for side, tokens in zip(pair, outputs):
        prefix = data.encode(side["route_prefix"])
        side.update(raw_ids=tokens, raw_text=data.decode(tokens), reached_eos=tokens[-1] == data.EOS,
                    first_decision_correct=tokens[0] == prefix[0],
                    route_in_slot=tokens[:len(prefix)] == prefix,
                    exact_reference=tokens == data.encode(side["reference"]) + [data.EOS],
                    unsupported_confirm=(not side["supported"] and data.CONFIRM in tokens))
        del side["ids"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, help="Optional fresh directory outside checkout for all pairs")
    parser.add_argument("--max-seconds", type=float, default=95)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 100:
        parser.error("Worker wall budget must be in (0,100] seconds")
    deadline = START + args.max_seconds
    dest = args.output.resolve() if args.output is not None else None
    if dest is not None and (dest.exists() or dest.is_relative_to(ROOT) or not dest.parent.is_dir()
                             or shutil.disk_usage(dest.parent).free < 256 * 1024**2):
        raise ValueError("Need a fresh external output and 256 MiB disk reserve")
    model, data = load_checkpoint(args.checkpoint, args.sha256)
    rows = [row for row in data.examples("test") if row["kind"] == "decision"]
    if len(rows) != 64 or len({row["id"] for row in rows}) != 64:
        raise ValueError("Unexpected frozen public test population")
    pairs = []
    for row in rows:
        pair, events, changed = make_pair(row, data)
        run_pair(model, pair, events, data, deadline)
        pairs.append(dict(id=row["id"], changed_input_positions=changed,
                          receipt_sha256=digest(encoded(events)), original=pair[0], swapped=pair[1]))
    metrics = ["first_decision_correct", "route_in_slot", "exact_reference", "unsupported_confirm"]
    counts = {side: {key: sum(int(p[side][key]) for p in pairs) for key in metrics}
              for side in ("original", "swapped")}
    paired = {key: sum(int(p["original"][key] and p["swapped"][key]) for p in pairs)
              for key in ("first_decision_correct", "route_in_slot", "exact_reference")}
    selected = [p for p in pairs if p["id"] in {"test-interval-0-slack-qa", "test-interval-0-slack-qb"}]
    examples = [{k: (v if k == "id" else {field: v[field] for field in ("route_prefix", "reference", "raw_ids", "raw_text")})
                 for k, v in p.items() if k in ("id", "original", "swapped")} for p in selected]
    result = dict(checkpoint_sha256=CHECKPOINT_SHA, probe_sha256=digest(Path(__file__).read_bytes()),
                  historical_source_sha256=TRAIN_SOURCES, test_pairs=len(pairs), raw_greedy_limit=LIMIT,
                  counts=counts, both_sides_correct=paired,
                  raw_response_changed=sum(p["original"]["raw_ids"] != p["swapped"]["raw_ids"] for p in pairs),
                  route_metric="Exact output prefix through the queried plan's route clause and punctuation",
                  examples=examples)
    if dest is not None:
        payload = encoded(pairs)
        if len(payload) > 1024**2 or shutil.disk_usage(dest.parent).free < 256 * 1024**2:
            raise RuntimeError("Output size or disk admission failed")
        dest.mkdir()
        (dest / "pairs.json").write_bytes(payload)
        result["pairs_sha256"] = digest(payload)
        (dest / "summary.json").write_bytes(encoded(result))
    print(encoded(result).decode())


if __name__ == "__main__":
    main()
