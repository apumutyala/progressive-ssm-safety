"""Fixed eight-update query/byte fit on the public routed fixture.

The prose prefix encodes every event in arrival order. Only the query bytes
enter the selector. Expected routes and status sentences are supervised targets.
The typed observer, grammar witness and sink do not enter either neural forward.
"""
from collections import defaultdict
from hashlib import sha256
import json

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from corpus_model import BYTE_OFFSET, EOS, PAD, VOCAB_SIZE, ByteTokenizer
from qualification_model import ARMS, _linear
from routed_cases import ROUTES, request_for
from routed_neural import (CONTEXT, DECODE_LIMIT, DEPTH, LR, NONE_ID, REFERENCES,
                           SCORE_TAGS, SEED, WIDTH, RoutedByteLM, ids_for)

STEPS = 8
PROBE_STEPS = (0, 4, 8)
KINDS = ("ALLOW", "NO", "UNKNOWN")
OPCODES = (("grant", None), ("revoke", None), ("invalidate", None),
           ("observe", False), ("observe", True))


def compact_prefix(request, events):
    """Count, fixed-size event bytes, then unmodified request (including arbitrary binary)."""
    if type(request) is not bytes or len(request) > 512 or len(events) > 64:
        raise ValueError("Invalid compact prefix")
    out = bytearray([len(events)])
    for event in events:
        if type(event.revision) is not int or not 1 <= event.revision <= 3:
            raise ValueError("The frozen fixture codec supports revisions 1..3")
        if event.value is not None and type(event.value) is not bool:
            raise ValueError("Event value must be boolean or None")
        if type(event.route) is not tuple or type(event.route[-1]) is not int:
            raise ValueError("Invalid route")
        opcode = OPCODES.index((event.kind, event.value))
        out.append(16 * ROUTES.index(event.route) + 5 * (event.revision - 1) + opcode)
    return bytes(out) + request


def training_plan(cases):
    """Indices are fixed by tags. No sampling, adaptive weighting or score selection."""
    tags = {case.tag: i for i, case in enumerate(cases)}
    if len(tags) != 74 or len(tags) != len(cases):
        raise ValueError("Expected the unique 74-case public fixture")
    # One representative per request. Core ledger and expected status are never
    # passed to the selector; all sixteen spellings have identical exposure.
    core = {cases[tags[f"core:{slot}:ALLOW"]].request: tags[f"core:{slot}:ALLOW"]
            for slot in range(16)}
    supported = [core[request_for(route)] for route in ROUTES]
    steps = []
    for step in range(STEPS):
        selector = tuple(supported + [tags[f"unsupported:{step}"]])
        prose = tuple(tags[f"core:{slot}:{kind}"]
                     for slot in (step * 2, step * 2 + 1) for kind in KINDS)
        steps.append((selector, prose))
    if {(cases[i].expected_route, cases[i].expected_kind) for _, rows in steps for i in rows} != {
            (route, kind) for route in ROUTES for kind in KINDS}:
        raise ValueError("Incomplete route/status coverage")
    return tuple(steps)


def _digest(obj):
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _tensor(rows, device):
    return torch.tensor(rows, dtype=torch.long, device=device)


def _padded(rows, device):
    length = max(map(len, rows))
    if not 0 < length <= CONTEXT or any(not row for row in rows):
        raise ValueError("Invalid packed sequence length")
    return _tensor([row + [PAD] * (length - len(row)) for row in rows], device)


def _prose_arrays(rows):
    inputs, positions, targets = [], [], []
    for case in rows:
        prefix = ids_for(compact_prefix(case.request, case.events))
        response = [*ByteTokenizer.encode_bytes(REFERENCES[case.expected_kind]), EOS]
        inputs.append(prefix + response[:-1])
        positions.append(list(range(len(prefix) - 1, len(prefix) - 1 + len(response))))
        targets.append(response)
    return inputs, positions, targets


def _route_target(case):
    return NONE_ID if case.expected_route is None else ROUTES.index(case.expected_route)


def _states(model, ids):
    """Same full-sequence embedding, recurrent clamps and causal attention as Branch.forward."""
    branch = model.branch
    x = branch.embedding(ids).clamp(-4, 4) + .25 * branch.position[:ids.shape[1]]
    for layer in branch.layers:
        x, _ = layer(x, None, None)
    return F.layer_norm(x, (WIDTH,))


def _selector_logits(model, states, lengths, device):
    row_ind = torch.arange(len(lengths), device=device)
    return model.scope(states[row_ind, _tensor([length - 1 for length in lengths], device)])


def _finite(tensors, device, label):
    if not tensors or any(t is None or t.device != device or t.dtype != torch.float32 for t in tensors):
        raise ValueError(f"Missing or misplaced FP32 tensor in {label}")
    if not torch.stack([torch.isfinite(t).all() for t in tensors]).all().item():
        raise ValueError(f"Nonfinite {label}")


def _response_logits(model, states, positions, row_offset, device):
    batch = [row + row_offset for row, group in enumerate(positions) for _ in group]
    times = [t for group in positions for t in group]
    return _linear(states[_tensor(batch, device), _tensor(times, device)], model.branch.head)


def _train_step(model, optimizer, route_cases, prose_cases, device, check):
    route_ids = [ids_for(c.request) for c in route_cases]
    route_targets = [_route_target(c) for c in route_cases]
    prose_ids, positions, targets = _prose_arrays(prose_cases)
    # Right padding cannot affect earlier causal states. Both tasks use one
    # layer call. The full-vocabulary head is evaluated at all answer positions.
    ids = _padded(route_ids + prose_ids, device)
    states = _states(model, ids)
    selector = _selector_logits(model, states, list(map(len, route_ids)), device)
    logits = _response_logits(model, states, positions, len(route_cases), device)
    flat_targets = [t for group in targets for t in group]
    if selector.shape != (len(route_cases), NONE_ID + 1) or logits.shape != (len(flat_targets), VOCAB_SIZE):
        raise ValueError("Unexpected supervised output shape")
    _finite([selector, logits], device, "training logits")
    route_loss = F.cross_entropy(selector, _tensor(route_targets, device))
    prose_loss = F.cross_entropy(logits, _tensor(flat_targets, device))
    loss = route_loss + prose_loss
    _finite([loss], device, "objective")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    _finite([p.grad for p in model.parameters()], device, "gradient")
    check()
    optimizer.step()
    _finite(list(model.parameters()), device, "updated parameters")
    # SHA covers actual tokens, supervised positions, labels and row order.
    return dict(selector_loss=route_loss.item(), prose_loss=prose_loss.item(), loss=loss.item(),
                selector_targets=len(route_cases), prose_targets=len(flat_targets),
                eos_targets=flat_targets.count(EOS),
                exposure_sha256=_digest([route_ids, route_targets, prose_ids, positions, targets]))


@torch.no_grad()
def _probe(step, model, cases, device):
    requests = sorted({c.request for c in cases})
    rows = [ids_for(request) for request in requests]
    logits = _selector_logits(model, _states(model, _padded(rows, device)), list(map(len, rows)), device)
    if logits.shape != (len(requests), NONE_ID + 1):
        raise ValueError("Invalid selector probe")
    _finite([logits], device, "probe logits")
    guesses = {req: None if i == NONE_ID else ROUTES[i]
               for req, i in zip(requests, logits.argmax(-1).tolist(), strict=True)}
    return dict(step=step, predicted_routes=[guesses[c.request] for c in cases])


@torch.no_grad()
def _score_prose(model, cases, plan, device, check):
    by_tag = {c.tag: i for i, c in enumerate(cases)}
    indices = [by_tag[tag] for tag in SCORE_TAGS]
    score_cases = [cases[i] for i in indices]
    prefixes = [compact_prefix(c.request, c.events) for c in score_cases]
    prefix_ids = [ids_for(p) for p in prefixes]
    by_length = defaultdict(list)
    for i, row in enumerate(prefix_ids):
        by_length[len(row)].append(i)
    raw = [[] for _ in score_cases]
    # Equal-length batches preserve exact cache offsets, without padding or a
    # synthetic gap in the model's prefix. Finished rows never supply output.
    for length, group in by_length.items():
        if length + DECODE_LIMIT > CONTEXT:
            raise ValueError("Prefill exceeds context")
        check()
        logits, cache = model.branch(_tensor([prefix_ids[i] for i in group], device), 0, None, None)
        for offset in range(DECODE_LIMIT):
            check()
            last = logits[:, -1]
            _finite([last], device, "generation logits")
            tokens = last.argmax(-1)
            for i, token in zip(group, tokens.tolist(), strict=True):
                if not raw[i] or raw[i][-1] != EOS:
                    raw[i].append(token)
            if offset + 1 == DECODE_LIMIT or all(raw[i][-1] == EOS for i in group):
                break
            logits, cache = model.branch(tokens[:, None], length + offset, cache, None)

    # Separate reference-conditioned diagnostics. These never feed generation.
    prose_ids, positions, targets = _prose_arrays(score_cases)
    states = _states(model, _padded(prose_ids, device))
    logits = _response_logits(model, states, positions, 0, device)
    _finite([logits], device, "teacher-forced score logits")
    seen_prose = {i for _, rows in plan for i in rows}
    seen_queries = {cases[i].request for rows, _ in plan for i in rows}
    results, start = [], 0
    for index, case, prefix, tokens, gold in zip(indices, score_cases, prefixes, raw, targets, strict=True):
        eos = tokens[-1] == EOS
        content = tokens[:-1] if eos else tokens
        ordinary = all(BYTE_OFFSET <= t < VOCAB_SIZE for t in content)
        data = ByteTokenizer.decode_bytes(content) if ordinary else None
        reference = REFERENCES[case.expected_kind]
        pred = logits[start:start + len(gold)]
        labels = _tensor(gold, device)
        results.append(dict(case_index=index, tag=case.tag, prefix_hex=prefix.hex(), raw_ids=tokens,
                            raw_hex=data.hex() if ordinary else None, text=ByteTokenizer.render(content),
                            eos=eos, truncated=not eos, reference=reference.decode("ascii"),
                            exact_content=data == reference, exact=bool(eos and data == reference),
                            first_divergence=next((j for j, (a,b) in enumerate(zip(tokens,gold)) if a!=b),
                                                  None if tokens==gold else min(len(tokens),len(gold))),
                            teacher_forced_nll=F.cross_entropy(pred, labels).item(),
                            first_target_rank_min=1 + int((pred[0] > pred[0, gold[0]]).sum().item()),
                            seen_prose_training=index in seen_prose,
                            seen_selector_training=case.request in seen_queries))
        start += len(gold)
    return results


def run_arm(arm, cases, check, device):
    """One serial guarded worker. Initialize on CPU identically, then fit on the requested device."""
    check()
    if arm not in ARMS or str(device) not in ("cpu", "cuda", "cuda:0"):
        raise ValueError("Unexpected arm or device")
    if torch.get_num_threads() != 1 or not torch.are_deterministic_algorithms_enabled():
        raise ValueError("Single-thread deterministic guarded runtime required")
    target = torch.device("cpu" if str(device) == "cpu" else "cuda:0")
    plan = training_plan(cases)
    torch.random.default_generator.manual_seed(SEED)
    model = RoutedByteLM(arm).to(target)
    _finite(list(model.parameters()), target, "initial parameters")
    _finite(list(model.buffers()), target, "position/timescale buffers")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, foreach=False, fused=False)
    probes, updates = [], []
    # Fix the attention algorithm across the three arms. No mixed precision,
    # graph compilation, recurrent reassociation, dropout, or device fallback.
    with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type=target.type, enabled=False):
        for step in range(STEPS + 1):
            check()
            if step in PROBE_STEPS:
                model.eval()
                probes.append(_probe(step, model, cases, target))
            if step == STEPS:
                break
            model.train()
            selector, prose_indices = plan[step]
            update = _train_step(model, optimizer, [cases[i] for i in selector],
                                 [cases[i] for i in prose_indices], target, check)
            updates.append(dict(update=step + 1, selector_indices=selector, prose_indices=prose_indices,
                                **update))
        prose = _score_prose(model, cases, plan, target, check)
    config = dict(steps=STEPS, probe_steps=PROBE_STEPS, width=WIDTH, depth=DEPTH,
                  max_context=CONTEXT, decode_limit=DECODE_LIMIT, seed=SEED, learning_rate=LR,
                  attention="math", optimizer="Adam", betas=[0.9, 0.999], eps=1e-8,
                  weight_decay=0, foreach=False, fused=False, compact_codec="count-route-rev-op-v1",
                  objectives=["route_ce_mean", "answer_and_eos_byte_ce_mean"], prose_score_tags=SCORE_TAGS)
    return dict(arm=arm, predicted_routes=probes[-1]["predicted_routes"], probes=probes, prose=prose,
                width=WIDTH, depth=DEPTH, max_context=CONTEXT, decode_limit=DECODE_LIMIT,
                parameters=sum(p.numel() for p in model.parameters()), seed=SEED, tensor_device=str(target),
                config=config, config_sha256=_digest(config),
                training=dict(steps=STEPS, updates=updates, learning_rate=LR, optimizer="Adam",
                              selector_targets=sum(r["selector_targets"] for r in updates),
                              prose_targets=sum(r["prose_targets"] for r in updates),
                              eos_targets=sum(r["eos_targets"] for r in updates),
                              exposure_sha256=_digest([r["exposure_sha256"] for r in updates])))
