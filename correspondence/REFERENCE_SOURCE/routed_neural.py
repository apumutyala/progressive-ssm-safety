"""Tiny CPU learning-path qualification on public synthetic fixtures.

The selector consumes request bytes. The separate prose forward consumes the
entire chronological observation ledger and request. The only information
shared between these forwards is the learned byte model's weights. Neither
forward calls a parser, reads reference labels or selects evidence by oracle.
The fixed reference sentences below are targets for loss and scoring only.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from corpus_model import BOS, BYTE_OFFSET, EOS, PAD, ByteTokenizer
from qualification_model import ARMS, Branch
from routed_cases import ROUTES

WIDTH, DEPTH, CONTEXT = 16, 1, 4096
SEED, LR, DECODE_LIMIT = 29041, 0.003, 32
NONE_ID = len(ROUTES)
REFERENCES = {
    "ALLOW": b"Yes, it is authorized.",
    "NO": b"No, the record is negative.",
    "UNKNOWN": b"The evidence is missing.",
    "DENY": b"Permission is missing.",
    "UNSUPPORTED": b"Please rephrase the request.",
}
TRAIN_PANELS = (
    ("person_switch:ada", "person_switch:ben", "missing:all", "revoke:ada", "unsupported:0"),
    ("regrant:ada", "evidence_flip:ada", "invalidate:ada", "missing:authority", "unsupported:1"),
)
SCORE_TAGS = (
    "person_switch:ada", "person_switch:ben", "evidence_flip:ada", "nuisance:ben_flip",
    "invalidate:ada", "revoke:ada", "regrant:ada", "unsupported:2",
)


def prose_prefix(request, events):
    """Public rendering of actual *input* records only; no join, sort or reduction."""
    if type(request) is not bytes or len(request) > 512 or len(events) > 64:
        raise ValueError("Input exceeds the synthetic qualification limits")
    lines = [b"Records:\n"]
    for event in events:
        who, job, source, epoch = event.route
        value = {None: "-", True: "true", False: "false"}[event.value]
        lines.append(f"{event.kind} {who} {job} {source} {epoch} {event.revision} {value}\n".encode("ascii"))
    return b"".join(lines) + b"Request: " + request + b"\nAnswer: "


def packed(rows):
    """Pack short byte sequences. BOS makes an empty request a valid model input."""
    if not rows or any(len(row) > CONTEXT or not row for row in rows):
        raise ValueError("Empty batch/sequence or exceeded context")
    lengths = torch.tensor([len(row) for row in rows], dtype=torch.long)
    ids = torch.full((len(rows), int(lengths.max())), PAD, dtype=torch.long)
    for index, row in enumerate(rows):
        ids[index, :len(row)] = torch.tensor(row, dtype=torch.long)
    return ids, lengths


def ids_for(raw):
    return [BOS, *ByteTokenizer.encode_bytes(raw)]


class RoutedByteLM(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm}")
        self.branch = Branch(arm, WIDTH, DEPTH, CONTEXT, False)
        self.scope = nn.Linear(WIDTH, len(ROUTES) + 1)

    def selector(self, requests):
        """The causal hidden state at the last actual request byte predicts scope."""
        if any(type(r) is not bytes or len(r) > 512 for r in requests):
            raise ValueError("Selector requires plain request bytes")
        ids, lengths = packed([ids_for(r) for r in requests])
        x = self.branch.embedding(ids).clamp(-4, 4) + .25 * self.branch.position[:ids.size(1)]
        for layer in self.branch.layers:
            x, _ = layer(x, None, None)
        last = x[torch.arange(len(requests)), lengths - 1]
        return self.scope(F.layer_norm(last, (WIDTH,)))

    def validate(self):
        for p in self.parameters():
            if p.device.type != "cpu" or p.dtype != torch.float32 or not torch.isfinite(p).all().item():
                raise ValueError("Finite CPU float32 parameters required")


def prose_training_rows(cases):
    inputs, labels = [], []
    for case in cases:
        prefix = ids_for(prose_prefix(case.request, case.events))
        response = [*ByteTokenizer.encode_bytes(REFERENCES[case.expected_kind]), EOS]
        # At the final prompt position predict the first answer byte. The final
        # answer position predicts EOS. Ignore prompt and trailing batch padding.
        inputs.append(prefix + response[:-1])
        labels.append([-100] * (len(prefix) - 1) + response)
    ids, _ = packed(inputs)
    target = torch.full_like(ids, -100)
    for i, row in enumerate(labels):
        target[i, :len(row)] = torch.tensor(row, dtype=torch.long)
    return ids, target


@torch.no_grad()
def generate(model, request, events, check):
    raw_prefix = prose_prefix(request, events)
    prefix = ids_for(raw_prefix)
    if len(prefix) + DECODE_LIMIT > CONTEXT:
        raise ValueError("Generation exceeds qualification context")
    ids, _ = packed([prefix])
    logits, cache = model.branch(ids, 0, None, None)
    generated = []
    for step in range(DECODE_LIMIT):
        check()
        last = logits[0, -1]
        if not torch.isfinite(last).all().item():
            raise ValueError("Nonfinite autoregressive logits")
        token = int(last.argmax().item())
        generated.append(token)
        if token == EOS or step + 1 == DECODE_LIMIT:
            break
        logits, cache = model.branch(torch.tensor([[token]]), len(prefix) + step, cache, None)
    return raw_prefix, generated


def run_arm(arm, cases, check):
    """Exactly two CPU Adam steps. No file I/O, heldout, checkpoint, or GPU."""
    check()
    if torch.get_num_threads() != 1:
        raise ValueError("Worker must select one CPU thread")
    if not 1 <= len(cases) <= 2048:
        raise ValueError("Unexpected qualification case count")
    torch.random.default_generator.manual_seed(SEED)
    model = RoutedByteLM(arm)
    model.validate()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, foreach=False)
    by_tag = {c.tag: i for i, c in enumerate(cases)}
    if len(by_tag) != len(cases):
        raise ValueError("Qualification tags must be unique")
    # Route labels are used only for supervised targets. Each supported request
    # has an equal number of exposures irrespective of the number of ledgers.
    supported = {}
    for i, case in enumerate(cases):
        if case.expected_route is not None:
            supported.setdefault(case.request, i)
    if len(supported) != len(ROUTES):
        raise ValueError("Expected one supported spelling per route")
    train_steps, seen_prose, seen_requests = [], set(), set()
    for step, tags in enumerate(TRAIN_PANELS):
        check()
        model.train()
        selector_idxs = [supported[req] for req in sorted(supported)]
        selector_idxs += [by_tag[f"unsupported:{step}"]]
        route_cases = [cases[i] for i in selector_idxs]
        requests = [c.request for c in route_cases]
        seen_requests.update(requests)
        target = torch.tensor([NONE_ID if c.expected_route is None else ROUTES.index(c.expected_route)
                               for c in route_cases])
        route_loss = F.cross_entropy(model.selector(requests), target)
        prose_idxs = [by_tag[tag] for tag in tags]
        seen_prose.update(prose_idxs)
        ids, labels = prose_training_rows([cases[i] for i in prose_idxs])
        logits, _ = model.branch(ids, 0, None, None)
        prose_loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        loss = route_loss + prose_loss
        if not torch.isfinite(loss).item():
            raise ValueError("Nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in model.parameters()):
            raise ValueError("Missing or nonfinite gradient")
        check()
        optimizer.step()
        model.validate()
        train_steps.append(dict(update=step + 1, selector_indices=selector_idxs,
                                prose_indices=prose_idxs, selector_targets=len(selector_idxs),
                                prose_targets=int((labels != -100).sum()), eos_targets=int((labels == EOS).sum()),
                                selector_loss=route_loss.item(), prose_loss=prose_loss.item(), loss=loss.item()))

    model.eval()
    requests = sorted({c.request for c in cases})
    routes = {}
    with torch.no_grad():
        for start in range(0, len(requests), 16):
            check()
            batch = requests[start:start + 16]
            logits = model.selector(batch)
            if not torch.isfinite(logits).all().item():
                raise ValueError("Nonfinite selector logits")
            for request, idx in zip(batch, logits.argmax(-1).tolist(), strict=True):
                routes[request] = None if idx == NONE_ID else ROUTES[idx]

    prose = []
    for tag in SCORE_TAGS:
        check()
        i = by_tag[tag]
        case = cases[i]
        prefix, raw = generate(model, case.request, case.events, check)
        eos = raw[-1] == EOS
        content = raw[:-1] if eos else raw
        ordinary = all(t >= BYTE_OFFSET for t in content)
        content_bytes = ByteTokenizer.decode_bytes(content) if ordinary else None
        reference = REFERENCES[case.expected_kind]
        prose.append(dict(case_index=i, tag=tag, prefix_hex=prefix.hex(), raw_ids=raw,
                         raw_hex=content_bytes.hex() if ordinary else None,
                         text=ByteTokenizer.render(content), eos=eos, truncated=not eos,
                         reference=reference.decode("ascii"), exact_content=content_bytes == reference,
                         exact=bool(eos and content_bytes == reference),
                         seen_prose_training=i in seen_prose, seen_selector_training=case.request in seen_requests))

    return dict(arm=arm, predicted_routes=[routes[c.request] for c in cases], prose=prose,
                width=WIDTH, depth=DEPTH, max_context=CONTEXT, decode_limit=DECODE_LIMIT,
                parameters=sum(p.numel() for p in model.parameters()), seed=SEED,
                training=dict(steps=len(train_steps), updates=train_steps, learning_rate=LR,
                              optimizer="Adam", selector_targets=sum(r["selector_targets"] for r in train_steps),
                              prose_targets=sum(r["prose_targets"] for r in train_steps),
                              eos_targets=sum(r["eos_targets"] for r in train_steps)))
