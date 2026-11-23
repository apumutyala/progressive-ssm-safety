"""Bounded CPU greedy inference from a digest-pinned, self-produced checkpoint.

Ordinary mixed-domain inference uses the embedded frozen byte codec and does
not open the private training pack. Only self-authored dev/test fixtures can
supply primitive receipts. A literal control spelling in --prompt is bytes.
"""
from __future__ import annotations

import time
STARTED = time.monotonic()
import argparse
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import shutil

import torch

SOURCE = Path(__file__).resolve().parent
MAX_CHECKPOINT_BYTES = 4 * 1024**2 - 512 * 1024


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def args():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--sha256', required=True, help='Expected SHA256 of a trusted local self-produced checkpoint.')
    p.add_argument('--output', type=Path, required=True, help='New external output directory.')
    p.add_argument('--device', choices=['cpu'], default='cpu')
    p.add_argument('--task', choices=['planning', 'interval', 'mixed'], default='mixed')
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--prompt', help='Ordinary language only; mixed task only.')
    choice.add_argument('--case-id', help='Exact ID of a self-authored dev/test decision fixture.')
    p.add_argument('--max-input-tokens', type=int, default=256)
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--max-seconds', type=float, default=30)
    a = p.parse_args()
    if len(a.sha256) != 64 or any(c not in '0123456789abcdef' for c in a.sha256):
        p.error('Expected SHA256 must be 64 lowercase hexadecimal characters.')
    if not (1 <= a.max_input_tokens <= 512 and 1 <= a.max_new_tokens <= 64 and 0 < a.max_seconds <= 110):
        p.error('Input <=512 tokens, generation <=64 tokens, worker <=110 seconds.')
    if a.prompt is not None and (a.task != 'mixed' or not a.prompt or len(a.prompt.encode('utf-8')) > 4096):
        p.error('--prompt requires mixed task and 1..4096 UTF8 bytes.')
    return a


def finite(x, name):
    if not torch.isfinite(x).all().item():
        raise RuntimeError(f'Nonfinite {name}; generation rejected.')


def check_state(state):
    for h in state.hidden:
        finite(h, 'recurrent state')
        if (h.abs() > 1).any().item():
            raise RuntimeError('Recurrent state escaped the admitted invariant.')


def support(state, task):
    if task == 'planning':
        return bool(state.evidence.all().item()), None
    query = int(state.query.item())
    if query < 0:
        return False, query
    permit, capacity, demand = state.records[0, query].tolist()
    return permit == 1 and capacity >= 0 and demand >= 0 and capacity >= demand, query


def render_with_framing(ids, data):
    framing = {data.PAD, data.UNK, data.BOS, data.EOS, data.CONFIRM, data.DEFER}
    framing.update(getattr(data, k) for k in ('QUERY_A', 'QUERY_B') if hasattr(data, k))
    pieces, pending = [], []
    for token in ids:
        if token in framing:
            if pending:
                pieces.append(data.decode(pending)); pending.clear()
            pieces.append(data.VOCAB[token])
        else:
            pending.append(token)
    if pending:
        pieces.append(data.decode(pending))
    return ''.join(pieces)


def main():
    a = args()
    deadline = STARTED + a.max_seconds

    def budget():
        if time.monotonic() >= deadline - 1:
            raise RuntimeError('Bounded inference wall-time reserve reached.')

    output = a.output.resolve()
    if output.is_relative_to(SOURCE) or output.exists() or not output.parent.is_dir():
        raise RuntimeError('Choose a fresh output directory outside source, under an existing parent.')
    if shutil.disk_usage(output.parent).free < 256 * 1024**2:
        raise RuntimeError('Disk reserve below 256 MiB.')
    checkpoint_path = a.checkpoint.resolve()
    if not 0 < checkpoint_path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise RuntimeError('Checkpoint exceeds admitted size.')
    with checkpoint_path.open('rb') as checkpoint_file:
        raw = checkpoint_file.read(MAX_CHECKPOINT_BYTES + 1)
    if len(raw) > MAX_CHECKPOINT_BYTES:
        raise RuntimeError('Checkpoint grew beyond admitted size.')
    if sha(raw) != a.sha256:
        raise RuntimeError('Checkpoint expected digest mismatch; not deserialized.')
    # Only tensor and primitive objects are admitted; this tool never unpickles arbitrary code.
    checkpoint = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1 or checkpoint['training_config']['task'] != a.task:
        raise RuntimeError('Checkpoint format/task mismatch.')
    if checkpoint['torch_version'] != str(torch.__version__):
        raise RuntimeError('Checkpoint Torch runtime mismatch.')
    names = ['experiment.py', 'model.py', 'data.py']
    if a.task in ('interval', 'mixed'):
        names += ['interval_model.py', 'interval_data.py']
    if a.task == 'mixed':
        names += ['mixed_data.py']
    observed_sources = {n: sha((SOURCE/n).read_bytes()) for n in names}
    if checkpoint['source_sha256'] != observed_sources:
        raise RuntimeError('Checkpoint allowed model/data/experiment source pins mismatch.')
    data = importlib.import_module({'planning': 'data', 'interval': 'interval_data', 'mixed': 'mixed_data'}[a.task])
    vocab_sha = sha(json.dumps(data.VOCAB).encode())
    audit = checkpoint.get('data_audit', {})
    if audit.get('vocab_sha256') != vocab_sha:
        raise RuntimeError('Checkpoint exact vocabulary identity mismatch.')
    if a.task == 'mixed':
        for key, value in [('pack_manifest_sha256', data.PACK_MANIFEST_SHA256),
                           ('tokenizer_artifact_sha256', data.TOKENIZER_SHA),
                           ('tokenizer_digest', data.TOKENIZER_DIGEST)]:
            if audit.get(key) != value:
                raise RuntimeError(f'Checkpoint mixed codec provenance mismatch: {key}.')
    width, blocks = checkpoint['width'], checkpoint['blocks']
    if type(width) is not int or not 1 <= width <= 128 or type(blocks) is not int or not 1 <= blocks <= 2:
        raise RuntimeError('Checkpoint model dimensions exceed reviewed experiment family.')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    if a.task == 'planning':
        cls = importlib.import_module('model').Model
        extra = {}
    else:
        cls = importlib.import_module('interval_model').IntervalModel
        extra = {'query_a_id': data.QUERY_A, 'query_b_id': data.QUERY_B}
    model = cls(len(data.VOCAB), width, data.CONFIRM, data.DEFER, **extra)
    for _ in range(blocks-1):
        model.grow()
    model.load_state_dict(checkpoint['model'], strict=True)
    model.validate_parameters(); model.eval()
    fixture = None
    if a.prompt is not None:
        ids = [data.BOS] + data.encode(a.prompt)
        if data.decode(ids) != a.prompt:
            raise RuntimeError('Ordinary prompt failed exact byte reconstruction.')
        if {data.CONFIRM, data.DEFER, data.QUERY_A, data.QUERY_B}.intersection(ids):
            raise RuntimeError('Ordinary prompt unexpectedly encoded as protocol IDs.')
        events = [[[data.HOLD]*3 for _ in range(2)] for _ in ids]
        prompt = a.prompt
    else:
        # No call to examples(train), corpus_manifest, load_pack, or private source paths.
        candidates = [r for split in ('dev', 'test') for r in data.examples(split)
                      if r['kind'] == 'decision' and r['id'] == a.case_id]
        if len(candidates) != 1:
            raise RuntimeError('Unknown or ambiguous self-authored dev/test decision fixture ID.')
        fixture = candidates[0]
        ids = fixture['ids'][:fixture['prompt_len']]
        events = fixture['events'][:fixture['prompt_len']]
        prompt = fixture['prompt']
    if len(ids) > a.max_input_tokens:
        raise RuntimeError('Prompt exceeds max-input-tokens; no truncation permitted.')
    budget()
    x = torch.tensor([ids], dtype=torch.long)
    e = torch.tensor([events], dtype=torch.long)
    generated, steps = [], []
    with torch.no_grad():
        all_logits, state = model.prefill(x, events=e)
        finite(all_logits, 'prefill logits'); check_state(state)
        logits = all_logits[:, -1]
        initial_support, initial_query = support(state, a.task)
        if fixture is not None and initial_support != fixture['supported']:
            raise RuntimeError('Primitive receipt state disagrees with frozen fixture oracle.')
        for index in range(a.max_new_tokens):
            budget(); finite(logits, 'decode logits'); check_state(state)
            supported, query = support(state, a.task)
            token = int(logits.argmax(-1).item())
            probabilities = logits.softmax(-1)
            margin = float(logits[0, data.DEFER] - logits[0, data.CONFIRM])
            if not supported and not margin > 0:
                raise RuntimeError('Executed unsupported decision-margin assertion failed.')
            generated.append(token)
            steps.append({'index': index, 'selected_id': token, 'selected_symbol': data.VOCAB[token],
                'selected_probability': float(probabilities[0, token]),
                'confirm_probability': float(probabilities[0, data.CONFIRM]),
                'defer_probability': float(probabilities[0, data.DEFER]),
                'defer_minus_confirm_logit': margin, 'supported_at_emission': supported,
                'query_at_emission': query})
            if token == data.EOS:
                break
            hold = getattr(data, 'HOLD', 0)
            hold_events = torch.full(e.shape[2:], hold, dtype=torch.long).unsqueeze(0)
            logits, state = model.decode_step(torch.tensor([token]), state, events=hold_events)
    result = {'task': a.task, 'checkpoint_sha256': a.sha256, 'source_sha256': observed_sources,
        'vocab_sha256': vocab_sha, 'prompt': prompt, 'case_id': a.case_id,
        'input_ids': ids, 'input_events': events, 'input_with_framing': render_with_framing(ids, data),
        'raw_ids': generated, 'raw_symbols': [data.VOCAB[t] for t in generated],
        'raw_text_with_framing': render_with_framing(generated, data), 'raw_text': data.decode(generated),
        'steps': steps, 'reached_eos': generated[-1] == data.EOS,
        'length_capped': generated[-1] != data.EOS, 'worker_wall_seconds': time.monotonic()-STARTED,
        'initial_support': initial_support, 'initial_query': initial_query, 'greedy': True,
        'ordinary_inference_requires_private_pack': False,
        'scope': 'Reserved raw decision IDs only, conditional on trusted primitive receipts and pinned fp32 path. Ordinary generated prose has no semantic safety certificate.'}
    payload = json.dumps(result, separators=(',', ':'), allow_nan=False).encode()
    if len(payload) > 96 * 1024:
        raise RuntimeError('Inference output bound exceeded; no truncated release.')
    if shutil.disk_usage(output.parent).free < 256 * 1024**2:
        raise RuntimeError('Disk reserve below 256 MiB before write.')
    output.mkdir()
    (output/'generation.json').write_bytes(payload)
    (output/'manifest.json').write_text(json.dumps({'generation.json': {'sha256': sha(payload), 'bytes': len(payload)}}), encoding='utf-8')
    print(payload.decode())


if __name__ == '__main__':
    main()
