"""TRAIN-only overfit slices for LATER supervised use; import is stdlib-only.

No CLI, file writer, launcher, ledger or evaluator. Production calls perform
exactly8 updates, at most128 cumulative, using three rows (one per status).
Only ordinary input IDs reach the model; masks implement shifted complete-reply
next-token loss, with one unit weight per answer byte/EOS target. No forced
predictions, answer slot labels, development examples or evaluation gradients.

Snapshots are bounded4MiB in-memory full-state images, not durable checkpoints.
The parent owns admission, persistence, readback and independent evaluation.
"""
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import random
import struct
import sys
import time

ROOT = Path(__file__).resolve().parent
ARMS = ('certified_ssm', 'unconstrained_ssm', 'transformer')
STATUSES = ('affirmative', 'negative', 'abstain')
SEED, UPDATES_PER_CALL, MAX_UPDATES, BATCH_ROWS = 12, 8, 128, 3
MAX_STATE_BYTES = 4*1024**2
IGNORE = -100
RECIPE = {'seed': 12, 'width': 32, 'depth': 2, 'max_context': 1024, 'batch_rows': 3,
          'updates_per_call': 8, 'max_updates': 128, 'lr': .003, 'clip': 1.,
          'loss': 'unit_weight_masked_next_token_mean_including_answer_eos',
          'schedule': 'one_per_status_rotating_six_row_epoch_seed12'}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source_manifest():
    names = ('qualification_overfit.py', 'qualification_data.py', 'qualification_model.py',
             'corpus_model.py', 'interval_model.py', 'model.py')
    pins = []
    for name in names:
        path = ROOT/name
        size = path.stat().st_size
        need(0 < size <= 128*1024, 'Source file cap.')
        with path.open('rb') as stream:
            raw = stream.read(128*1024+1)
        need(len(raw) == size, 'Source changed during read.')
        pins.append((name, sha(raw)))
    return sha(canonical(pins))


def gate_description():
    """Delegates semantic judgement to the independently owned evaluator."""
    return {'scope': 'TRAIN-only overfit qualification, not heldout competence or alignment',
            'cases': 18, 'complete_grounded_raw_greedy_correct_required': 18,
            'required_correct_by_status': {s: 6 for s in STATUSES},
            'complete_reply_eos_required': 18, 'truncated_allowed': 0,
            'wrong_entity_or_action_allowed': 0, 'ordinary_reply_protocol_tokens_allowed': 0,
            'decode': 'raw full vocabulary; no forced prefix, filter, checker repair or supplied copy labels',
            'stop': 'stop candidate at128 updates; failure to fit is reported, never extended automatically',
            'evaluation': 'parent evaluator owns raw generations/scoring; this module never calls it'}


def validate_rows(rows, max_context=1024):
    need(type(rows) in (tuple, list) and len(rows) == 18, 'Exactly18 overfit TRAIN rows required.')
    need(Counter(r['status'] for r in rows) == Counter({s: 6 for s in STATUSES}), 'Balanced6/status overfit group required.')
    need(len({r['id'] for r in rows}) == 18, 'Distinct row IDs required.')
    result = []
    for row in rows:
        ids, mask, p = tuple(row['ids']), tuple(row['loss_mask']), row['prompt_len']
        need(row.get('kind') == 'synthetic' and ids[0] == 2 and ids[-1] == 3
             and 2 < len(ids) <= max_context+1
             and all(type(t) is int and 8 <= t < 264 for t in ids[1:-1]), 'Ordinary-byte complete reply framing required.')
        need(type(p) is int and 1 < p < len(ids)-1 and len(mask) == len(ids)-1
             and all(type(m) is bool for m in mask)
             and mask == tuple(i >= p-1 for i in range(len(ids)-1)), 'Prediction-aligned reply-only loss mask required.')
        result.append({'id': row['id'], 'kind': 'synthetic', 'status': row['status'],
                       'ids': ids, 'loss_mask': mask, 'prompt_len': p})
    return tuple(result)


def data_manifest():
    """Public TRAIN metadata only. No development/final reader is called."""
    import qualification_data as data
    rows = validate_rows(data.overfit_group(seed=SEED))
    return {'split': 'train', 'group': data.overfit_manifest(),
            'train_catalog_sha256': data.catalog_manifest('train')['sha256'],
            'model_record_sha256': sha(canonical(rows))}


def batch_rows(rows, step, seed=SEED):
    """Each six-update cycle visits every row once; no global RNG consumption."""
    need(type(step) is int and 0 <= step < MAX_UPDATES and seed == SEED, 'Frozen step/seed required.')
    epoch, offset = divmod(step, 6)
    chosen = []
    for status in STATUSES:
        pool = sorted((r for r in rows if r['status'] == status), key=lambda r: r['id'])
        need(len(pool) == 6, 'Need six rows per status.')
        random.Random(int(sha(canonical(['overfit', seed, epoch, status])), 16)).shuffle(pool)
        chosen.append(pool[offset])
    random.Random(int(sha(canonical(['batch-order', seed, step])), 16)).shuffle(chosen)
    return tuple(chosen)


def collate(rows):
    """Pure Python shifted masks. No oracle or row label enters model inputs."""
    width = max(len(r['ids'])-1 for r in rows)
    x, y = [], []
    for row in rows:
        pad = width-len(row['ids'])+1
        x.append(tuple(row['ids'][:-1])+(3,)*pad)
        y.append(tuple(t if keep else IGNORE for t, keep in zip(row['ids'][1:], row['loss_mask']))+(IGNORE,)*pad)
    counts = {'answer_byte_targets': sum(t >= 8 for values in y for t in values),
              'answer_eos_targets': sum(t == 3 for values in y for t in values),
              'masked_prompt_targets': sum(r['prompt_len']-1 for r in rows),
              'padding_positions': sum(width-len(r['ids'])+1 for r in rows)}
    counts['total_targets'] = counts['answer_byte_targets']+counts['answer_eos_targets']
    need(counts['total_targets'] == sum(sum(r['loss_mask']) for r in rows) > 0, 'Actual target count mismatch.')
    return tuple(x), tuple(y), counts


def _runtime(torch, device, production):
    on_cuda = torch.device(device).type == 'cuda'
    value = {'python': sys.version.split()[0], 'torch': str(torch.__version__), 'device': str(torch.device(device)),
             'deterministic': torch.are_deterministic_algorithms_enabled(), 'threads': torch.get_num_threads(),
             'interop_threads': torch.get_num_interop_threads(), 'tf32': torch.backends.cuda.matmul.allow_tf32 if on_cuda else None,
             'cudnn_tf32': torch.backends.cudnn.allow_tf32 if on_cuda else None,
             'matmul_precision': torch.get_float32_matmul_precision() if on_cuda else 'cpu_fp32'}
    if production:
        need(value['deterministic'] and not value['tf32'] and not value['cudnn_tf32']
               and value['threads'] == value['interop_threads'] == 1 and value['matmul_precision'] == ('highest' if on_cuda else 'cpu_fp32'),
             'Parent must admit deterministic single-thread FP32 settings.')
    return value


def _cpu(value, torch):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(v, torch) for key, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v, torch) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v, torch) for v in value)
    return value


def _state_digest(value, torch):
    h = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            t = item.detach().cpu().contiguous()
            h.update(canonical(['tensor', str(t.dtype), list(t.shape)]))
            h.update(t.numpy().tobytes())  # dtype/shape and exact bits, incl signed zero
        elif isinstance(item, dict):
            h.update(b'dict')
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode()); h.update(canonical(len(item)))
            for element in item: visit(element)
        else:
            h.update(canonical([type(item).__name__, item]))
    visit(value)
    return h.hexdigest()


def _rng(torch, device):
    return {'python': random.getstate(), 'cpu': torch.get_rng_state().clone(),
            'cuda': torch.cuda.get_rng_state(torch.device(device)).clone() if torch.device(device).type == 'cuda' else None}


def _restore_rng(value, torch, device):
    need(type(value) is dict and set(value) == {'python', 'cpu', 'cuda'}, 'Unknown RNG schema.')
    cpu = value['cpu']
    need(torch.is_tensor(cpu) and cpu.dtype == torch.uint8 and cpu.ndim == 1
         and cpu.shape == torch.get_rng_state().shape, 'CPU RNG shape/type mismatch.')
    check = random.Random(0)
    check.setstate(value['python'])
    if torch.device(device).type == 'cuda':
        cuda = value['cuda']
        need(torch.is_tensor(cuda) and cuda.dtype == torch.uint8
             and cuda.shape == torch.cuda.get_rng_state(torch.device(device)).shape, 'CUDA RNG shape/type mismatch.')
        torch.cuda.set_rng_state(cuda, device=torch.device(device))
    else:
        need(value['cuda'] is None, 'Foreign CUDA state in CPU checkpoint.')
    torch.set_rng_state(cpu)
    random.setstate(value['python'])


def _components(payload, torch):
    keys = ('model', 'optimizer', 'rng', 'step', 'counts', 'sampling_digest', 'identity')
    return {key: _state_digest(payload[key], torch) for key in keys}


def _decode(checkpoint, expected, torch):
    need(type(checkpoint) is bytes and 0 < len(checkpoint) <= MAX_STATE_BYTES
         and type(expected) is str and len(expected) == 64 and sha(checkpoint) == expected,
         'Expected exact bounded4MiB in-memory checkpoint pin.')
    payload = torch.load(io.BytesIO(checkpoint), map_location='cpu', weights_only=True)
    need(type(payload) is dict and set(payload) == {'schema', 'kind', 'model', 'optimizer', 'rng', 'step',
         'counts', 'sampling_digest', 'identity', 'component_hashes'} and payload['schema'] == 1,
         'Unexpected full-state schema.')
    need(payload['component_hashes'] == _components(payload, torch), 'Full-state component fingerprint mismatch.')
    return payload


def _validate_optimizer(model, optimizer, step, torch):
    named = tuple(model.parameters())
    need(len(optimizer.param_groups) == 1 and list(optimizer.param_groups[0]['params']) == list(named), 'Optimizer parameter order changed.')
    need(optimizer.param_groups[0]['lr'] == .003 and optimizer.param_groups[0]['betas'] == (.9, .999)
         and optimizer.param_groups[0]['eps'] == 1e-8 and optimizer.param_groups[0]['weight_decay'] == 0
         and optimizer.param_groups[0]['amsgrad'] is False, 'Frozen Adam settings changed.')
    if step:
        need(set(optimizer.state) == set(named), 'Missing/extra Adam state.')
        for parameter in named:
            slot = optimizer.state[parameter]
            need(set(slot) == {'step', 'exp_avg', 'exp_avg_sq'} and float(slot['step']) == step, 'Adam cursor/schema changed.')
            for key in ('exp_avg', 'exp_avg_sq'):
                v = slot[key]
                need(v.dtype == parameter.dtype == torch.float32 and v.shape == parameter.shape
                     and torch.isfinite(v).all().item(), 'Invalid Adam moment.')
            need(not (slot['exp_avg_sq'] < 0).any().item(), 'Negative Adam second moment.')


def _run_slice(arm, rows, factory, identity, *, updates, checkpoint=None, checkpoint_sha256=None,
               device='cpu', check, production):
    """Private fixture seam; production always enters run_slice below."""
    started = time.monotonic()
    need(callable(check) and type(updates) is int and 1 <= updates <= UPDATES_PER_CALL, 'Bounded update count and supervisor guard required.')
    import torch
    def guard(training=False):
        need(check() is not False and time.monotonic()-started < (80 if training else 100), 'Supervised deadline/admission rejected.')
    guard()
    device = torch.device(device)
    need(device.type in ('cpu', 'cuda') and device.index in (None, 0), 'Only CPU or admitted CUDA0.')
    runtime = _runtime(torch, device, production)
    bound = dict(identity, runtime=runtime)
    random.seed(SEED); torch.manual_seed(SEED)
    model = factory(arm).to(device)
    need(sum(p.numel() for p in model.parameters()) <= 250000, 'Qualification parameter bound exceeded.')
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    step, counts, sampling = 0, {'answer_byte_targets': 0, 'answer_eos_targets': 0, 'total_targets': 0}, '0'*64
    if checkpoint is not None:
        payload = _decode(checkpoint, checkpoint_sha256, torch)
        need(payload['kind'] == ('qualification_overfit_full' if production else 'qualification_overfit_fixture')
             and payload['identity'] == bound and type(payload['step']) is int
             and 0 <= payload['step'] <= MAX_UPDATES, 'Exact source/data/runtime/arm/recipe/cursor restore mismatch.')
        model.load_state_dict(payload['model'], strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        step, counts, sampling = payload['step'], dict(payload['counts']), payload['sampling_digest']
        _validate_optimizer(model, optimizer, step, torch)
        # Restore generators AFTER model construction and optimizer loading.
        _restore_rng(payload['rng'], torch, device)
        actual = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': _rng(torch, device),
                  'step': step, 'counts': counts, 'sampling_digest': sampling, 'identity': bound}
        need(_components(actual, torch) == payload['component_hashes'], 'Independent live post-restore component bits differ.')
    else:
        need(checkpoint_sha256 is None, 'Do not provide a pin without a state.')
    need(step+updates <= MAX_UPDATES and (not production or step % 8 == 0), 'Candidate stop128 or fixed8-step ring violated.')
    model.validate_parameters()
    model.train()
    logs = []
    for cursor in range(step, step+updates):
        guard(True)
        selected = batch_rows(rows, cursor)
        xx, yy, measured = collate(selected)
        x = torch.tensor(xx, device=device, dtype=torch.long)
        y = torch.tensor(yy, device=device, dtype=torch.long)
        model.validate_parameters(); optimizer.zero_grad(set_to_none=True)
        logits = model(x)  # Raw ordinary IDs ONLY. No oracle/metadata/events.
        need(logits.shape == (*x.shape, 264) and logits.dtype == torch.float32
             and torch.isfinite(logits).all().item(), 'Nonfinite or incomplete raw vocabulary.')
        loss_sum = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(),
            reduction='sum', ignore_index=IGNORE)
        loss = loss_sum/measured['total_targets']
        need(torch.isfinite(loss).item(), 'Nonfinite next-token loss.')
        loss.backward()
        need(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()), 'Missing/nonfinite gradient.')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); model.validate_parameters()
        _validate_optimizer(model, optimizer, cursor+1, torch)
        guard(True)
        descriptor = [(r['id'], r['ids'], r['loss_mask']) for r in selected]
        sampling = sha(sampling.encode()+canonical({'step': cursor, 'rows': descriptor}))
        for key in counts: counts[key] += measured[key]
        logs.append({'step': cursor+1, 'row_ids': [r['id'] for r in selected],
                     'status_counts': dict(Counter(r['status'] for r in selected)),
                     **measured, 'loss': float(loss.detach()), 'sampling_digest': sampling})
    guard()
    endpoint = step+updates
    payload = {'schema': 1, 'kind': 'qualification_overfit_full' if production else 'qualification_overfit_fixture',
        'model': _cpu(model.state_dict(), torch), 'optimizer': _cpu(optimizer.state_dict(), torch),
        'rng': _cpu(_rng(torch, device), torch), 'step': endpoint, 'counts': counts,
        'sampling_digest': sampling, 'identity': bound}
    payload['component_hashes'] = _components(payload, torch)
    buffer = io.BytesIO(); torch.save(payload, buffer); state = buffer.getvalue()
    need(len(state) <= MAX_STATE_BYTES, 'Serialized state exceeds4MiB; no partial state returned.')
    decoded = _decode(state, sha(state), torch)
    need(decoded['component_hashes'] == payload['component_hashes'], 'In-memory state readback differs.')
    guard()
    return {'state': state, 'state_sha256': sha(state), 'state_bytes': len(state), 'step': endpoint,
            'training_rows': logs, 'counts': dict(counts), 'component_hashes': payload['component_hashes'],
            'data_manifest': identity['data'], 'source_sha256': identity['source_sha256'],
            'candidate_stop_reached': endpoint == MAX_UPDATES, 'worker_seconds': time.monotonic()-started,
            'evaluations_performed': 0, 'persisted': False}


def run_slice(arm, *, updates=8, seed=12, checkpoint=None, checkpoint_sha256=None,
              source_sha256, device='cpu', check):
    """Parent-supervised production entry; EXACT8 updates, no persistence/eval."""
    need(arm in ARMS and updates == 8 and seed == SEED, 'Frozen arm/8-update seed12 qualification ring required.')
    need(source_sha256 == source_manifest(), 'Reviewed executable source fingerprint changed.')
    import qualification_data as data
    from qualification_model import make_model
    rows = validate_rows(data.overfit_group(seed=SEED))
    manifest = data_manifest()
    identity = {'arm': arm, 'source_sha256': source_sha256, 'data': manifest, 'recipe': dict(RECIPE)}
    return _run_slice(arm, rows, lambda a: make_model(a, width=32, depth=2, max_context=1024), identity,
        updates=updates, checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha256,
        device=device, check=check, production=True)


def load_model_from_checkpoint(arm, checkpoint, checkpoint_sha256, source_sha256, device='cpu'):
    """Strict read-only-data score loader; preserves caller RNG, returns eval model.

    Does not evaluate, persist, repair predictions, load development or train.
    The evaluator owns no_grad and raw decoding. Same runtime/device is required.
    """
    need(arm in ARMS and source_sha256 == source_manifest(), 'Unknown arm or changed executable source.')
    import torch
    from qualification_model import make_model
    before = _rng(torch, device)
    try:
        payload = _decode(checkpoint, checkpoint_sha256, torch)
        expected = {'arm': arm, 'source_sha256': source_sha256, 'data': data_manifest(),
                    'recipe': dict(RECIPE), 'runtime': _runtime(torch, device, True)}
        need(payload['kind'] == 'qualification_overfit_full' and payload['identity'] == expected
             and type(payload['step']) is int and 0 < payload['step'] <= MAX_UPDATES and payload['step'] % 8 == 0,
             'Read-only score identity/cursor mismatch.')
        model = make_model(arm, width=32, depth=2, max_context=1024).to(device)
        model.load_state_dict(payload['model'], strict=True); model.validate_parameters(); model.eval()
        need(_state_digest(model.state_dict(), torch) == payload['component_hashes']['model'], 'Score model raw bits differ.')
        return model
    finally:
        _restore_rng(before, torch, device)
