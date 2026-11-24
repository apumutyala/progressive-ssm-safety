"""Evidence-state interventions for the fitted, diagonal CorpusModel.

Pure helpers and fixtures import without Torch. The model adapter is optional
and must run in a budgeted worker. Its target is the FIRST DIFFERING ordinary
answer byte, not the correctness of a complete response or protected marker.
"""
from dataclasses import dataclass, replace
import math


VOCAB, EOS, OFFSET = 264, 3, 8


@dataclass(frozen=True)
class Contrast:
    name: str
    kind: str
    split: str
    prefixes: tuple  # two bytestrings ending immediately after selected evidence
    suffix: bytes    # identical ordinary bytes through the question
    answers: tuple   # two ordinary canonical response bytestrings, without EOS
    labels: tuple    # two distinct oracle labels in the same order

    def branch(self):
        if (self.kind not in {"evidence", "revoke", "long_distractor"}
                or self.split not in {"probe_train", "diagnostic"} or not self.name
                or len(self.prefixes) != 2 or len(self.answers) != 2 or len(self.labels) != 2
                or len(set(self.labels)) != 2 or self.prefixes[0] == self.prefixes[1]
                or not isinstance(self.suffix, bytes) or not 1 <= len(self.suffix) <= 384
                or any(not isinstance(p, bytes) or not 1 <= len(p) <= 384 for p in self.prefixes)
                or any(not isinstance(a, bytes) or not 1 <= len(a) <= 192 for a in self.answers)):
            raise ValueError("Invalid bounded evidence contrast")
        for i, (a, b) in enumerate(zip(*self.answers)):
            if a != b:
                return i, (a + OFFSET, b + OFFSET)
        raise ValueError("Need a divergence between two ordinary bytes before EOS")


def public_contrasts(split):
    """Secondary synthetic diagnostics. Supersession is an explicit extra convention."""
    names = {"probe_train": (("Mara", "Taro"), ("Gwen", "Ilan")),
             "diagnostic": (("Nila", "Orin"), ("Vera", "Pavo"))}[split]
    result = []
    for name, other in names:
        for kind in ("evidence", "revoke", "long_distractor"):
            lead = f"{name} will pack. {other} will sort. "
            if kind == "revoke":
                lead += f"The latest report replaces the earlier report. Earlier {name}: ready now. Now "
            prefixes = tuple((lead + name + ": " + status + ".").encode() for status in ("ready now", "not ready"))
            filler = (f" {other} is ready to sort. The crates are by the door." * 3
                      if kind == "long_distractor" else "")
            suffix = (filler + f" Is {name} ready for the assigned task? Answer: ").encode()
            answers = (f"Yes. {name} is ready to pack.".encode(), f"No. {name} is not ready to pack.".encode())
            result.append(Contrast(f"{kind}-{name}", kind, split, prefixes, suffix, answers,
                                   ("supported", "contradicted")))
    return result


def margin_scores(logits, expected, foil):
    if (len(logits) != VOCAB or not all(math.isfinite(v) for v in logits)
            or not OFFSET <= expected < VOCAB or not OFFSET <= foil < VOCAB or expected == foil):
        raise ValueError("Need two distinct ordinary IDs and a finite full vocabulary")
    rival = max((i for i in range(VOCAB) if i != expected), key=logits.__getitem__)
    chosen = max(range(VOCAB), key=logits.__getitem__)
    return {"greedy_id": chosen, "expected_first_index_argmax": chosen == expected,
            "strongest_rival_id": rival, "strict_full_margin": logits[expected] - logits[rival],
            "named_foil_margin": logits[expected] - logits[foil]}


def _probe_prediction(vector, centers):
    if not centers or not all(math.isfinite(x) for x in vector):
        raise ValueError("Probe needs finite features and at least one centroid")
    distances = {label: sum((x - c) ** 2 for x, c in zip(vector, center, strict=True))
                 for label, center in centers.items()}
    if not all(math.isfinite(v) for v in distances.values()):
        raise FloatingPointError("Nonfinite centroid distance")
    best = min(distances.values())
    winners = [k for k, v in distances.items() if v == best]
    return winners[0] if len(winners) == 1 else None


def _ids(model, raw):
    import torch
    return torch.tensor([[b + OFFSET for b in raw]], device=next(model.parameters()).device, dtype=torch.int64)


def _check_model(model):
    from corpus_model import CorpusBlock, CorpusModel
    import torch
    if (type(model) is not CorpusModel or not 1 <= len(model.blocks) <= 2 or model.width > 32
            or any(type(b) is not CorpusBlock for b in model.blocks) or model.training
            or not torch.are_deterministic_algorithms_enabled()
            or torch.get_float32_matmul_precision() != "highest"):
        raise ValueError("Expected frozen bounded eager CorpusModel in deterministic FP32 eval mode")
    model.validate_parameters()


def _vector(state):
    import torch
    return torch.cat(state.hidden, dim=-1)[0].detach().double().tolist()


def fit_probe(model, contrasts):
    """Nearest-centroid diagnostic at the forced branch, report ties as undecided."""
    import torch
    _check_model(model)
    samples, prompts = {}, set()
    with torch.no_grad():
        for pair in contrasts:
            branch, _ = pair.branch()
            if pair.split != "probe_train":
                raise ValueError("Only probe-training cases may fit centroid weights")
            for prefix, answer, label in zip(pair.prefixes, pair.answers, pair.labels):
                prompt = prefix + pair.suffix
                if prompt in prompts:
                    raise ValueError("Duplicate probe-training prompt")
                prompts.add(prompt)
                _, state = model.prefill(_ids(model, prefix))
                _, state = model._run(_ids(model, pair.suffix + answer[:branch]), state, events=None)
                samples.setdefault(label, []).append(_vector(state))
    if len(samples) < 2:
        raise ValueError("Probe needs at least two labeled training classes")
    centers = {k: tuple(sum(col) / len(v) for col in zip(*v, strict=True)) for k, v in samples.items()}
    # This diagnostic is in-memory, tied to one unmodified model instance.
    return centers, prompts, tuple((id(p), p._version) for p in model.parameters())


def _raw_branch(model, state, pair, common, expected, foil):
    """Free decode only through the specified slot. Prior errors never get repaired."""
    import torch
    logits, state = model._run(_ids(model, pair.suffix), state, events=None)
    scores, prior_tokens = logits[0, -1], []
    original_records, original_query = state.records, state.query
    for _ in common:
        token = int(scores.argmax())
        prior_tokens.append(token)
        if token == EOS:
            return {"reached_branch": False, "eos_before_branch": True, "consumed_prefix_ids": prior_tokens}
        new_ids = torch.tensor([token], dtype=torch.int64, device=scores.device)
        logits, state = model.decode_step(new_ids, state)
        scores = logits[0]
    return {"reached_branch": True, "consumed_prefix_ids": prior_tokens,
            "prefix_matches_reference": prior_tokens == [b + OFFSET for b in common],
            "protected_state_unchanged": torch.equal(original_records, state.records) and
                                         torch.equal(original_query, state.query),
            **margin_scores(scores.detach().double().tolist(), expected, foil)}


def _linearize(model, tokens, state, donor, expected, foil):
    """O[t]=C[t] diag(prod A[0:t+1]); local real surrogate at executed FP32 values."""
    import torch
    from control_probe import _spectrum
    observations = []

    def capture(hidden, proposed, retention, *unused):
        observations.append((hidden, proposed.detach(), retention.detach()))

    with torch.inference_mode(False), torch.enable_grad():
        tokens = tokens.detach().clone()
        state = replace(state, hidden=tuple(x.detach().clone().requires_grad_(True) for x in state.hidden),
                        records=state.records.detach().clone(), query=state.query.detach().clone())
        with model.capture_dynamics() as audit:
            audit.record = capture
            logits, _ = model._run(tokens, state, events=None)
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("Nonfinite evidence trajectory")
        h, proposed, keep = zip(*observations)
        a = torch.cat(keep, dim=-1)[0].detach().cpu().double()
        proposal = torch.cat(proposed, dim=-1)[0].detach().cpu().double()
        if model.arm == "certified":
            a *= (proposal.abs() <= 1)
        history_jacobian = torch.cumprod(a, dim=0)
        initial = torch.cat(state.hidden, dim=-1)[0].detach().cpu().double()
        donor_vector = torch.cat(donor.hidden, dim=-1)[0].detach().cpu().double()
        directions = {"zero": -initial, "swap": donor_vector - initial}
        excluded = logits.detach().clone()
        excluded[..., expected] = -torch.inf
        rivals = excluded.argmax(dim=-1, keepdim=True)
        traces = {"named_foil": logits[..., expected] - logits[..., foil],
                  "strongest_rival": logits[..., expected] - logits.gather(-1, rivals).squeeze(-1)}
        report = {}
        for name, contrast in traces.items():
            gradients = torch.autograd.grad(contrast.sum(), h, retain_graph=(name == "named_foil"))
            local = torch.cat(gradients, dim=-1)[0].detach().cpu().double() * history_jacobian
            if not torch.isfinite(local).all().item():
                raise FloatingPointError("Nonfinite evidence observability row")
            report[name] = {"observability": _spectrum(local),
                            "terminal_jvp": {key: torch.dot(local[-1], d).item() for key, d in directions.items()},
                            "trajectory_jvp_l2": {key: torch.linalg.vector_norm(local @ d).item()
                                                  for key, d in directions.items()}}
        report["horizon_bytes"] = tokens.size(1)
        report["terminal_rival_tie_count"] = int((excluded[0, -1] == excluded[0, -1].max()).sum())
        report["state_clamp_boundary_count"] = int((proposal.abs() == 1).sum()) if model.arm == "certified" else 0
        return report


def analyze_pair(model, pair, probe):
    """Run a single paired diagnostic. Worker owns deadlines between bounded calls."""
    import torch
    _check_model(model)
    index, alternatives = pair.branch()
    centers, training_prompts, version = probe
    if (pair.split != "diagnostic" or any(p + pair.suffix in training_prompts for p in pair.prefixes)
            or not set(pair.labels).issubset(centers)
            or version != tuple((id(p), p._version) for p in model.parameters())):
        raise ValueError("Require a disjoint diagnostic pair and matching frozen probe/model")
    with torch.no_grad():
        states = [model.prefill(_ids(model, p))[1] for p in pair.prefixes]
        if not (torch.equal(states[0].records, states[1].records) and torch.equal(states[0].query, states[1].query)):
            raise ValueError("Paired snapshots must share protected records and first query")
        rows, terminal_bits = [], {}
        for i, intact in enumerate(states):
            donor = states[1-i]
            common = pair.answers[i][:index]
            tokens = _ids(model, pair.suffix + common)
            expected, foil = alternatives[i], alternatives[1-i]
            linear = _linearize(model, tokens, intact, donor, expected, foil)
            row = {}
            for action, hidden in (("intact", intact.hidden), ("zero", tuple(torch.zeros_like(h) for h in intact.hidden)),
                                   ("swap", donor.hidden)):
                cut = replace(intact, hidden=hidden)
                logits, end = model._run(tokens, cut, events=None)
                if not torch.equal(cut.records, end.records) or not torch.equal(cut.query, end.query):
                    raise RuntimeError("Fixed ordinary teacher path changed protected fields")
                terminal_bits[i,action] = logits[0,-1].contiguous().view(torch.int32)
                row[action] = {"teacher": margin_scores(logits[0,-1].double().tolist(), expected, foil),
                               "raw": _raw_branch(model, cut, pair, common, expected, foil),
                               "probe_label": _probe_prediction(_vector(end), centers)}
            for action in ("zero", "swap"):
                for name, key in (("named_foil", "named_foil_margin"), ("strongest_rival", "strict_full_margin")):
                    actual = row[action]["teacher"][key] - row["intact"]["teacher"][key]
                    prediction = linear[name]["terminal_jvp"][action]
                    row[action].setdefault("finite_intervention", {})[name] = {
                        "actual_margin_change": actual, "local_jvp_prediction": prediction,
                        "actual_minus_local": actual - prediction}
            rows.append({"oracle_label": pair.labels[i], "donor_label": pair.labels[1-i],
                         "linearization": linear, "interventions": row})
        if any(not torch.equal(terminal_bits[i,"swap"], terminal_bits[1-i,"intact"]) for i in range(2)):
            raise RuntimeError("Transplant failed to reproduce donor full logits on the matched suffix")
        if not torch.equal(terminal_bits[0,"zero"], terminal_bits[1,"zero"]):
            raise RuntimeError("Cleared learned states retained an unexplained difference")
    return {"name": pair.name, "kind": pair.kind, "branch_zero_based": index,
            "same_suffix_bytes": len(pair.suffix), "swap_reproduced_donor_full_logits": True,
            "zeroed_pair_full_logits_equal": True, "rows": rows}
