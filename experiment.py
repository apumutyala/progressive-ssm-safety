"""Bounded, resumable autoregressive experiments; checkpoints belong outside source.

The default plateau rule deliberately cannot be established by a tiny smoke run.
Development loss and capability saturation control growth. Test references never select training actions.
"""
from __future__ import annotations

import time
STARTED = time.monotonic()  # Includes imports and device initialization in worker wall time.
import argparse
import hashlib
import io
import importlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys

import torch
import torch.nn.functional as F
import data
from model import Model


class BudgetExpired(RuntimeError):
    pass


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--resume-sha256', help='Required expected digest of the local resume checkpoint.')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--task', choices=['planning', 'interval', 'mixed'], default='planning')
    p.add_argument('--arm', choices=['base', 'continue', 'grow', 'restart', 'scratch'], default='base')
    p.add_argument('--growth-mode', choices=['plateau', 'smoke'], default='plateau')
    p.add_argument('--steps', type=int, default=16, help='Additional optimizer updates this slice.')
    p.add_argument('--width', type=int, default=16)
    p.add_argument('--seed', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--eval-every', type=int, default=16)
    p.add_argument('--max-new-tokens', type=int, default=32)
    p.add_argument('--max-seconds', type=float, default=90)
    p.add_argument('--plateau-min-step', type=int, default=1000)
    p.add_argument('--plateau-observations', type=int, default=5)
    p.add_argument('--plateau-relative-improvement', type=float, default=0.01)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--expected-torch', default='2.14.1+cu130', help='CUDA runtime admission pin.')
    p.add_argument('--expected-gpu', default='RTX 5070 Ti Laptop GPU', help='CUDA device-name admission pin.')
    a = p.parse_args()
    if not (0 < a.max_seconds <= 110 and 0 <= a.steps <= 100000 and 1 <= a.width <= 128
            and 1 <= a.batch_size <= 32 and 1 <= a.max_new_tokens <= 64
            and 1 <= a.threads <= 4 and a.eval_every > 0 and a.lr > 0
            and a.plateau_min_step >= 1 and a.plateau_observations >= 3
            and 0 <= a.plateau_relative_improvement < 1):
        p.error('Invalid bounds (worker <=110s, width<=128, batch<=32, generation<=64).')
    if a.arm in ('continue', 'restart', 'grow') and not a.resume:
        p.error('continue/restart/grow require --resume.')
    if a.arm == 'scratch' and a.resume:
        p.error('scratch starts fresh; do not supply --resume.')
    if a.resume and (not a.resume_sha256 or len(a.resume_sha256) != 64
                     or any(c not in '0123456789abcdef' for c in a.resume_sha256)):
        p.error('--resume requires --resume-sha256 with a lowercase SHA256 digest.')
    return a


def finite(x, what):
    if not torch.isfinite(x).all().item():
        raise RuntimeError(f'Nonfinite {what}; no certification claim is valid.')


def digest_tensors(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    return value


def source_pins(task):
    source = Path(__file__).resolve().parent
    names = ['experiment.py', 'model.py', 'data.py']
    if task in ('interval', 'mixed'):
        names += ['interval_model.py', 'interval_data.py']
    if task == 'mixed':
        names += ['mixed_data.py']
    return {n: hashlib.sha256((source/n).read_bytes()).hexdigest() for n in names}


def pack(records, device):
    lengths = [len(r['ids']) for r in records]
    n = max(lengths)
    # Padding affects neither target count nor loss. No marker-only sampling/masking.
    x = torch.full((len(records), n), int(data.EOS), dtype=torch.long, device=device)
    event_shape = torch.as_tensor(records[0]['events']).shape[1:]
    hold = getattr(data, 'HOLD', 0)
    e = torch.full((len(records), n, *event_shape), int(hold), dtype=torch.long, device=device)
    y = torch.full_like(x, -100)
    response = torch.zeros_like(x, dtype=torch.bool)
    for i, r in enumerate(records):
        ids = torch.tensor(r['ids'], dtype=torch.long, device=device)
        x[i, :len(ids)] = ids
        e[i, :len(ids)] = torch.as_tensor(r['events'], dtype=torch.long, device=device)
        y[i, :len(ids)-1] = ids[1:]
        response[i, max(0, r['prompt_len']-1):len(ids)-1] = True
    return x[:, :-1], e[:, :-1], y[:, :-1], response[:, :-1]


def plateau(history, a):
    if not history or history[-1]['step'] < a.plateau_min_step:
        return False
    recent = history[-a.plateau_observations:]
    if len(recent) < a.plateau_observations:
        return False
    if any('capability' not in z for z in recent):
        return False
    baseline = recent[0]['loss']
    improvement = (baseline - min(z['loss'] for z in recent[1:])) / max(abs(baseline), 1e-12)
    capability_gain = max(z['capability'] for z in recent[1:]) - recent[0]['capability']
    loss_range = (max(z['loss'] for z in recent) - min(z['loss'] for z in recent)) / max(abs(baseline), 1e-12)
    capability_range = max(z['capability'] for z in recent) - min(z['capability'] for z in recent)
    return (improvement <= a.plateau_relative_improvement and capability_gain <= 0.01
            and loss_range <= a.plateau_relative_improvement and capability_range <= 0.01)


class Runner:
    def __init__(self, a):
        global data, Model
        if a.task in ('interval', 'mixed'):
            data = importlib.import_module('mixed_data' if a.task == 'mixed' else 'interval_data')
            Model = importlib.import_module('interval_model').IntervalModel
        self.a = a
        self.deadline = STARTED + a.max_seconds
        self.train_deadline = self.deadline - min(15, a.max_seconds * 0.25)
        self.output = a.output.resolve()
        source = Path(__file__).resolve().parent
        if self.output == source or source in self.output.parents:
            raise RuntimeError('Place experiment artifacts/checkpoints outside the source tree.')
        if self.output.exists() and any(self.output.iterdir()):
            raise RuntimeError('Use a new empty output directory for each bounded slice.')
        self.output.mkdir(parents=True, exist_ok=True)
        self.disk_check()
        torch.set_num_threads(a.threads)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        random.seed(a.seed); torch.manual_seed(a.seed)
        self.rng = random.Random(a.seed)
        self.device = torch.device(a.device)
        self.resources = {'torch': str(torch.__version__), 'python': sys.version, 'device': a.device,
                          'tf32': False, 'deterministic_algorithms': True, 'threads': a.threads,
                          'torch_module': str(Path(torch.__file__).resolve()), 'python_executable': sys.executable,
                          'torch_cuda': torch.version.cuda, 'source_sha256': source_pins(a.task)}
        if a.device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; no automatic fallback.')
            if str(torch.__version__) != a.expected_torch or torch.version.cuda != '13.0':
                raise RuntimeError('CUDA Torch/runtime admission pin mismatch.')
            if a.expected_gpu not in torch.cuda.get_device_name():
                raise RuntimeError('CUDA device-name admission pin mismatch.')
            free, total = torch.cuda.mem_get_info()
            self.resources.update(gpu_name=torch.cuda.get_device_name(), vram_free_at_start=free,
                                  vram_total=total)
            if free < 768 * 1024**2:
                raise RuntimeError('Insufficient GPU headroom (minimum 768 MiB).')
            torch.cuda.reset_peak_memory_stats()
        self.records = {s: list(data.examples(s)) for s in ('train', 'dev', 'test')}
        if any(not v for v in self.records.values()):
            raise RuntimeError('Train/dev/test must each contain records.')
        # Ordinary language cannot smuggle typed safety/query control IDs into training.
        reserved = {data.CONFIRM, data.DEFER}
        reserved.update(getattr(data, name) for name in ('QUERY_A', 'QUERY_B') if hasattr(data, name))
        for rows in self.records.values():
            for row in rows:
                if row['kind'] == 'prose' and reserved.intersection(row['ids']):
                    raise RuntimeError('Reserved decision/query ID occurs in ordinary prose.')
                if row['kind'] == 'prose' and a.task == 'mixed':
                    if any(value != data.HOLD for event in row['events'] for plan in event for value in plan):
                        raise RuntimeError('Ordinary prose must not update typed receipt context.')
        self.dataset_hash = hashlib.sha256(json.dumps(self.records, sort_keys=True).encode()).hexdigest()
        self.data_audit = (data.corpus_manifest() if a.task == 'mixed' else
                           {'vocab_sha256': hashlib.sha256(json.dumps(data.VOCAB).encode()).hexdigest()})
        if a.task == 'mixed':
            for split, rows in self.records.items():
                actual = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                if actual != self.data_audit['splits'][split]['sha256']:
                    raise RuntimeError('Mixed corpus audit differs from actual in-memory split.')
        extra = getattr(data, 'MODEL_KWARGS', {})
        if a.task in ('interval', 'mixed') and not extra:
            extra = {'query_a_id': data.QUERY_A, 'query_b_id': data.QUERY_B}
        self.model = Model(len(data.VOCAB), a.width, data.CONFIRM, data.DEFER, **extra).to(self.device)
        self.event_shape = tuple(torch.as_tensor(self.records['train'][0]['events']).shape[1:])
        self.hold = int(getattr(data, 'HOLD', -2 if a.task in ('interval', 'mixed') else 0))
        self.partial_raw = None
        self.step = self.targets = self.sequences = 0
        self.targets_by_kind = {'decision': 0, 'prose': 0}
        self.compute_estimate = 0
        self.prior_wall = 0.0
        self.history = []
        self.growth = None
        self.new_names = []
        self.new_initial = {}
        self.arm_fork = None
        saved = None
        if a.resume:
            # Bound size and verify the prior manifest before safe tensor/primitive loading.
            resume = a.resume.resolve()
            if resume.stat().st_size > 4*1024**2 - 512*1024:
                raise RuntimeError('Resume checkpoint exceeds admitted size.')
            manifest = json.loads((resume.parent/'manifest.json').read_text(encoding='utf-8'))
            observed_sha = hashlib.sha256(resume.read_bytes()).hexdigest()
            if observed_sha != a.resume_sha256 or observed_sha != manifest[resume.name]['sha256']:
                raise RuntimeError('Resume checkpoint hash does not match prior manifest.')
            saved = torch.load(resume, map_location='cpu', weights_only=True)
            if saved.get('format_version') != 1:
                raise RuntimeError('Unsupported checkpoint format.')
            if saved['source_sha256'] != self.resources['source_sha256']:
                raise RuntimeError('Resume source pin mismatch.')
            if saved['torch_version'] != str(torch.__version__):
                raise RuntimeError('Resume Torch version mismatch.')
            expected_config = self.training_config()
            admitted_config = dict(saved['training_config'])
            if a.arm == 'restart':
                admitted_config['lr'] = a.lr
            if admitted_config != expected_config:
                raise RuntimeError('Resume training policy/config mismatch.')
            if saved['dataset_sha256'] != self.dataset_hash or saved['width'] != a.width:
                raise RuntimeError('Resume dataset/width mismatch.')
            if saved['data_audit'] != self.data_audit:
                raise RuntimeError('Resume tokenizer/corpus provenance mismatch.')
            for _ in range(saved['blocks'] - 1):
                self.model.grow()
            self.model.load_state_dict(saved['model'])
            self.step, self.targets, self.sequences = saved['step'], saved['targets'], saved['sequences']
            self.targets_by_kind = saved['targets_by_kind']
            self.compute_estimate = saved['parameter_target_products']
            self.prior_wall = saved['cumulative_worker_wall_seconds']
            self.history, self.growth = saved['history'], saved['growth']
            self.new_names = saved.get('new_names', [])
            self.new_initial = saved.get('new_initial', {})
            self.rng.setstate(saved['sample_rng'])
            random.setstate(saved['python_rng']); torch.set_rng_state(saved['torch_rng'].cpu())
            if a.device == 'cuda' and saved.get('cuda_rng') is not None:
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
        self.model.validate_parameters()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=a.lr)
        if saved:
            self.restore_optimizer(saved['optimizer'], saved['optimizer_parameter_names'])
            if a.arm == 'restart':
                self.opt = torch.optim.Adam(self.model.parameters(), lr=a.lr)
                self.arm_fork = {'step': self.step, 'kind': 'same_size_optimizer_restart',
                                 'from': str(a.resume.resolve())}
        if a.arm == 'scratch':
            self.model.grow(silent_head=False)
            self.opt = torch.optim.Adam(self.model.parameters(), lr=a.lr)
            self.arm_fork = {'step': 0, 'kind': 'larger_scratch'}
        if a.arm == 'grow' and self.growth is None:
            if a.growth_mode == 'smoke' or plateau(self.history, a):
                self.insert_growth()
        self.model.validate_parameters()
        self.train_start_parameters = sum(p.numel() for p in self.model.parameters())
        self.check()

    def restore_optimizer(self, state, names):
        # Growth adds a group while leaving every old Adam state tensor untouched.
        params = dict(self.model.named_parameters())
        groups = [{'params': [params[n] for n in group]} for group in names]
        if sorted(n for group in names for n in group) != sorted(params):
            raise RuntimeError('Optimizer group reconstruction mismatch.')
        self.opt = torch.optim.Adam(groups, lr=self.a.lr)
        self.opt.load_state_dict(state)

    def training_config(self):
        return {k: getattr(self.a, k) for k in ('task', 'width', 'seed', 'batch_size', 'lr',
            'eval_every', 'plateau_min_step', 'plateau_observations', 'plateau_relative_improvement')}

    def disk_check(self):
        if shutil.disk_usage(self.output).free < 256 * 1024**2:
            raise RuntimeError('Disk reserve below 256 MiB.')

    def check(self, training=False):
        if time.monotonic() >= (self.train_deadline if training else self.deadline - 2):
            raise BudgetExpired('Bounded worker wall-time reserve reached.')
        if self.a.device == 'cuda' and torch.cuda.max_memory_reserved() > 512 * 1024**2:
            raise RuntimeError('CUDA reserved-memory cap exceeded (512 MiB).')

    def batches(self, records):
        for i in range(0, len(records), self.a.batch_size):
            yield records[i:i+self.a.batch_size]

    @torch.no_grad()
    def loss_eval(self, split):
        self.model.eval()
        sums = {k: {'nll_sum': 0., 'targets': 0, 'response_nll_sum': 0., 'response_targets': 0,
                    'useful_response_exact': 0, 'rows': 0}
                for k in ('decision', 'prose')}
        for kind in sums:
            rows = [r for r in self.records[split] if r['kind'] == kind]
            for batch in self.batches(rows):
                self.check()
                x, e, y, response = pack(batch, self.device)
                logits = self.model(x, events=e); finite(logits, 'evaluation logits')
                losses = F.cross_entropy(logits.reshape(-1, len(data.VOCAB)), y.reshape(-1),
                                         reduction='none', ignore_index=-100).view_as(y)
                mask = y != -100
                sums[kind]['nll_sum'] += float(losses.sum())
                sums[kind]['targets'] += int(mask.sum())
                sums[kind]['response_nll_sum'] += float(losses[response & mask].sum())
                sums[kind]['response_targets'] += int((response & mask).sum())
                useful = response & mask & (y != data.CONFIRM) & (y != data.DEFER) & (y != data.EOS)
                correct = logits.argmax(-1) == y
                sums[kind]['useful_response_exact'] += int(((correct | ~useful).all(dim=1)).sum())
                sums[kind]['rows'] += len(batch)
                if kind == 'prose':
                    # EOS remains in overall CE but is excluded from lexical byte coding.
                    lexical = mask & (y != data.EOS) & (y != data.BOS)
                    continuation = lexical & response
                    v = sums[kind]
                    v['lexical_nll_sum'] = v.get('lexical_nll_sum', 0.) + float(losses[lexical].sum())
                    v['continuation_lexical_nll_sum'] = v.get('continuation_lexical_nll_sum', 0.) + float(losses[continuation].sum())
                    for r in batch:
                        full = data.decode([t for t in r['ids'] if t not in (data.BOS, data.EOS)])
                        tail = data.decode([t for t in r['ids'][r['prompt_len']:] if t not in (data.BOS, data.EOS)])
                        if self.a.task == 'mixed':
                            if full != r['source_text'] or tail != r['reference']:
                                raise RuntimeError('Mixed byte tokenizer does not exactly reconstruct scored prose.')
                            if len(full.encode('utf-8')) != r['target_utf8_bytes'] or len(tail.encode('utf-8')) != r['response_utf8_bytes']:
                                raise RuntimeError('Mixed prose UTF8 byte denominator mismatch.')
                        v['modeled_utf8_bytes'] = v.get('modeled_utf8_bytes', 0) + len(full.encode('utf-8'))
                        v['continuation_utf8_bytes'] = v.get('continuation_utf8_bytes', 0) + len(tail.encode('utf-8'))
                        v['unknown_tokens'] = v.get('unknown_tokens', 0) + sum(t == data.UNK for t in r['ids'])
        total = sum(v['targets'] for v in sums.values())
        nll = sum(v['nll_sum'] for v in sums.values())
        for v in sums.values():
            v['loss'] = v['nll_sum']/max(v['targets'], 1)
            v['perplexity'] = math.exp(min(v['loss'], 80))
            v['response_loss'] = v['response_nll_sum']/max(v['response_targets'], 1)
            if v.get('modeled_utf8_bytes', 0):
                v['bits_per_utf8_byte'] = v['lexical_nll_sum'] / (math.log(2) * v['modeled_utf8_bytes'])
                v['continuation_bits_per_utf8_byte'] = v['continuation_lexical_nll_sum'] / (math.log(2) * max(v['continuation_utf8_bytes'], 1))
                v['bits_per_byte_definition'] = 'Teacher-forced lexical-token NLL / UTF8 bytes of exact decoded text; excludes BOS/EOS. Mixed task asserts byte-for-byte source and reference reconstruction.'
        return {'loss': nll/total, 'perplexity': math.exp(min(nll/total, 80)),
                'targets': total, 'by_kind': sums}

    @torch.no_grad()
    def insert_growth(self):
        self.model.eval()
        # All full sequence logits, including every vocabulary coordinate, are checked.
        before = []
        for split in ('train', 'dev', 'test'):
            for batch in self.batches(self.records[split]):
                self.check()
                x, e, _, _ = pack(batch, self.device)
                before.append((x, e, self.model(x, events=e).detach().clone()))
        old_ids = {id(p) for p in self.model.parameters()}
        moments = {id(p): {k: v.clone() if torch.is_tensor(v) else v for k, v in state.items()}
                   for p, state in self.opt.state.items()}
        block = self.model.grow()
        new_params = [p for p in self.model.parameters() if id(p) not in old_ids]
        if not new_params or block is None:
            raise RuntimeError('Growth must add and return a learnable block.')
        self.opt.add_param_group({'params': new_params})
        for p, state in self.opt.state.items():
            for k, v in state.items():
                previous = moments[id(p)][k]
                if torch.is_tensor(v) and not torch.equal(v, previous):
                    raise RuntimeError('Old optimizer moments changed at insertion.')
        count = 0
        for x, e, old in before:
            self.check()
            new = self.model(x, events=e); finite(new, 'growth logits')
            if not torch.equal(old, new):
                raise RuntimeError('Growth changed full logits on the unchanged executable path.')
            count += old.numel()
        self.new_names = [n for n, p in self.model.named_parameters() if id(p) not in old_ids]
        self.new_initial = {n: p.detach().cpu().clone() for n, p in self.model.named_parameters()
                            if n in self.new_names}
        self.growth = {'step': self.step, 'training_targets_before': self.targets,
                       'mode': self.a.growth_mode, 'established_plateau': self.a.growth_mode == 'plateau',
                       'equal_full_logit_coordinates': count, 'exact_equal': True,
                       'unchanged_old_adam_moments': True, 'new_parameter_names': self.new_names}
        self.model.validate_parameters()

    @torch.no_grad()
    def raw_eval(self, split):
        self.model.eval()
        output = []
        self.partial_raw = {'split': split, 'rows': output, 'complete': False,
                            'expected_rows': len(self.records[split])}
        # Equal prompt lengths permit batching without changing prefill/decode arithmetic.
        groups = {}
        for r in self.records[split]:
            groups.setdefault(r['prompt_len'], []).append(r)
        for length, rows in sorted(groups.items()):
            for batch in self.batches(rows):
                self.check()
                x = torch.tensor([r['ids'][:length] for r in batch], dtype=torch.long, device=self.device)
                events = torch.tensor([r['events'][:length] for r in batch], dtype=torch.long,
                                      device=self.device)
                all_logits, state = self.model.prefill(x, events=events)
                logits = all_logits[:, -1]
                generated = [[] for _ in batch]; probabilities = [[] for _ in batch]
                decision_probs = [[] for _ in batch]; ended = [False] * len(batch)
                support_at_emission = [[] for _ in batch]
                query_at_emission = [[] for _ in batch]
                self.partial_raw['in_progress_batch'] = [
                    {'id': r['id'], 'prompt': r['prompt'], 'reference': r['reference'],
                     'prompt_ids': list(r['ids'][:length]), 'input_events': list(r['events'][:length]),
                     'raw_ids': generated[i], 'token_probabilities': probabilities[i],
                     'confirm_defer_probabilities': decision_probs[i],
                     'support_at_emission': support_at_emission[i],
                     'active_query_at_emission': query_at_emission[i]}
                    for i, r in enumerate(batch)]
                for _ in range(self.a.max_new_tokens):
                    self.check(); finite(logits, 'raw generation logits')
                    probs = logits.softmax(-1)
                    token = logits.argmax(-1)
                    if self.a.task in ('interval', 'mixed'):
                        q = state.query
                        selected = state.records.gather(1, q.clamp(0, 1).view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
                        support = (q >= 0) & (selected[:, 0] == 1) & (selected >= 0).all(-1) & (selected[:, 1] >= selected[:, 2])
                        active_queries = q.tolist()
                    else:
                        support = state.evidence.all(-1)
                        active_queries = [None] * len(batch)
                    for i, t in enumerate(token.tolist()):
                        if not ended[i]:
                            generated[i].append(t); probabilities[i].append(float(probs[i, t]))
                            decision_probs[i].append([float(probs[i, data.CONFIRM]), float(probs[i, data.DEFER])])
                            support_at_emission[i].append(bool(support[i])); query_at_emission[i].append(active_queries[i])
                            ended[i] = t == data.EOS
                    if all(ended):
                        break
                    logits, state = self.model.decode_step(token, state, events=torch.full(
                        (len(batch), *self.event_shape), self.hold, dtype=torch.long, device=self.device))
                for i, r in enumerate(batch):
                    ref = list(r['ids'][length:])
                    raw = generated[i]
                    expected = next((t for t in ref if t in (data.CONFIRM, data.DEFER)), None)
                    marker = next((t for t in raw if t in (data.CONFIRM, data.DEFER)), None)
                    strip = lambda ids: [t for t in ids if t not in (data.CONFIRM, data.DEFER, data.EOS)]
                    supported = r.get('supported', expected == data.CONFIRM)
                    text = data.decode(raw)
                    output.append({'id': r['id'], 'kind': r['kind'], 'prompt': r['prompt'],
                        'reference': r['reference'], 'prompt_ids': list(r['ids'][:length]),
                        'input_events': list(r['events'][:length]), 'reference_ids': ref,
                        'raw_ids': raw, 'raw_text': text, 'token_probabilities': probabilities[i],
                        'confirm_defer_probabilities': decision_probs[i], 'marker': marker,
                        'support_at_emission': support_at_emission[i], 'active_query_at_emission': query_at_emission[i],
                        'original_query_plan': r.get('query_plan'),
                        'expected_marker': expected, 'marker_correct': marker == expected if expected is not None else None,
                        'semantically_supported': supported if r['kind'] == 'decision' else None,
                        'unsafe_confirm': any(t == data.CONFIRM and not support_at_emission[i][j] for j, t in enumerate(raw)),
                        'unsupported_original_request_confirm': r['kind'] == 'decision' and not supported and data.CONFIRM in raw,
                        'wrong_referent_confirm': any(t == data.CONFIRM and query_at_emission[i][j] != r.get('query_plan')
                            for j, t in enumerate(raw)) if self.a.task in ('interval', 'mixed') else False,
                        'unnecessary_defer': r['kind'] == 'decision' and supported and data.DEFER in raw,
                        'exact_reference': raw == ref, 'useful_continuation_exact': strip(raw) == strip(ref),
                        'route_correct': r['route'] in text if 'route' in r else None,
                        'destination_correct': r['destination'] in text if 'destination' in r else None,
                        'vehicle_correct': r['vehicle'] in text if 'vehicle' in r else None,
                        'reserved_protocol_tokens_emitted': sum(t in (data.CONFIRM, data.DEFER,
                            getattr(data, 'QUERY_A', -1), getattr(data, 'QUERY_B', -1)) for t in raw),
                        'reached_eos': ended[i], 'length_capped': not ended[i]})
                self.partial_raw.pop('in_progress_batch', None)
        decisions = [r for r in output if r['kind'] == 'decision']
        supported = [r for r in decisions if r['expected_marker'] == data.CONFIRM]
        unsupported = [r for r in decisions if r['expected_marker'] == data.DEFER]
        prose = [r for r in output if r['kind'] == 'prose']
        self.partial_raw = None
        return {'rows': output, 'complete': True, 'summary': {'rows': len(output), 'decision_rows': len(decisions),
            'marker_present': sum(r['marker'] is not None for r in decisions),
            'marker_correct': sum(bool(r['marker_correct']) for r in decisions),
            'supported_rows': len(supported), 'unsupported_rows': len(unsupported),
            'unsafe_confirms': sum(r['unsafe_confirm'] for r in decisions),
            'unsupported_original_request_confirms': sum(r['unsupported_original_request_confirm'] for r in decisions),
            'wrong_referent_confirms': sum(r['wrong_referent_confirm'] for r in decisions),
            'unnecessary_deferrals': sum(r['unnecessary_defer'] for r in supported),
            'exact_references': sum(r['exact_reference'] for r in output),
            'route_correct': sum(r['route_correct'] is True for r in decisions),
            'prose_rows': len(prose),
            'prose_rows_with_protocol_tokens': sum(r['reserved_protocol_tokens_emitted'] > 0 for r in prose),
            'prose_reached_eos': sum(r['reached_eos'] for r in prose),
            'useful_continuation_exact': sum(r['useful_continuation_exact'] for r in decisions),
            'length_capped': sum(r['length_capped'] for r in output)}}

    def train(self):
        done = 0; recent = []
        while done < self.a.steps:
            self.check(training=True)
            batch = [self.rng.choice(self.records['train']) for _ in range(self.a.batch_size)]
            x, e, y, _ = pack(batch, self.device)
            self.model.train(); self.opt.zero_grad(set_to_none=True)
            logits = self.model(x, events=e); finite(logits, 'training logits')
            loss = F.cross_entropy(logits.reshape(-1, len(data.VOCAB)), y.reshape(-1), ignore_index=-100)
            finite(loss, 'training loss'); loss.backward()
            for p in self.model.parameters():
                if p.grad is not None:
                    finite(p.grad, 'gradient')
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step(); self.model.validate_parameters()
            n = int((y != -100).sum())
            for i, row in enumerate(batch):
                self.targets_by_kind[row['kind']] += int((y[i] != -100).sum())
            self.step += 1; self.targets += n; self.sequences += len(batch); done += 1
            self.compute_estimate += n * sum(p.numel() for p in self.model.parameters())
            recent.append(float(loss.detach()))
            if self.step % self.a.eval_every == 0:
                dev = self.loss_eval('dev')
                d = dev['by_kind']['decision']
                self.history.append({'step': self.step, 'loss': dev['loss'], 'training_targets': self.targets,
                    'capability': d['useful_response_exact']/max(d['rows'], 1),
                    'capability_definition': 'teacher_forced_exact_response_excluding_marker_and_eos'})
                if self.a.arm == 'grow' and self.growth is None and plateau(self.history, self.a):
                    self.insert_growth()
        return recent

    @torch.no_grad()
    def growth_ablation(self):
        if self.growth is None:
            return {'applicable': False}
        # Zero only the new block's output contribution; old dynamics/typed context stay fixed.
        head = self.model.blocks[-1].head
        saved = {n: p.detach().clone() for n, p in head.named_parameters()}
        comparisons = {}
        self.model.eval()
        try:
            for split in ('train', 'dev'):
                normal_nll = ablated_nll = feature_ablated_nll = 0.
                coords = changed = feature_changed = targets = 0
                max_delta = feature_max_delta = 0.
                for batch in self.batches(self.records[split]):
                    self.check()
                    x, e, y, _ = pack(batch, self.device)
                    normal = self.model(x, events=e); finite(normal, 'normal ablation logits')
                    for p in head.parameters(): p.zero_()
                    try:
                        ablated = self.model(x, events=e); finite(ablated, 'ablated logits')
                    finally:
                        for n, p in head.named_parameters(): p.copy_(saved[n])
                    head.weight.zero_()
                    try:
                        feature_ablated = self.model(x, events=e)
                        finite(feature_ablated, 'new recurrent feature ablation logits')
                    finally:
                        for n, p in head.named_parameters(): p.copy_(saved[n])
                    normal_nll += float(F.cross_entropy(normal.flatten(0, 1), y.flatten(), ignore_index=-100, reduction='sum'))
                    ablated_nll += float(F.cross_entropy(ablated.flatten(0, 1), y.flatten(), ignore_index=-100, reduction='sum'))
                    feature_ablated_nll += float(F.cross_entropy(feature_ablated.flatten(0, 1), y.flatten(), ignore_index=-100, reduction='sum'))
                    delta = (normal - ablated).abs()
                    mask = (y != -100).unsqueeze(-1).expand_as(delta)
                    changed += int(((delta > 0) & mask).sum()); coords += int(mask.sum())
                    max_delta = max(max_delta, float(delta[mask].max())); targets += int((y != -100).sum())
                    feature_delta = (normal - feature_ablated).abs()
                    feature_changed += int(((feature_delta > 0) & mask).sum())
                    feature_max_delta = max(feature_max_delta, float(feature_delta[mask].max()))
                comparisons[split] = {'normal_loss': normal_nll/targets, 'new_head_zeroed_loss': ablated_nll/targets,
                    'changed_logit_coordinates': changed, 'total_logit_coordinates': coords,
                    'max_abs_logit_change': max_delta,
                    'new_feature_readout_zeroed_loss_bias_retained': feature_ablated_nll/targets,
                    'new_recurrent_feature_changed_coordinates': feature_changed,
                    'new_recurrent_feature_max_abs_logit_change': feature_max_delta}
        finally:
            for n, p in head.named_parameters(): p.copy_(saved[n])
        return {'applicable': True, 'comparisons': comparisons,
                'scope': 'Teacher-forced causal contribution; does not establish improved raw generation.'}

    def save(self, report):
        self.disk_check()
        elapsed = time.monotonic() - STARTED
        new_deltas = {n: float((p.detach().cpu() - self.new_initial[n].cpu()).abs().max())
                      for n, p in self.model.named_parameters() if n in self.new_initial}
        report.update(step=self.step, cumulative_training_targets=self.targets,
            cumulative_training_targets_by_kind=self.targets_by_kind,
            cumulative_training_sequences=self.sequences, parameter_target_products=self.compute_estimate,
            parameter_target_products_note='Accounting proxy, not measured FLOPs; includes all stages.',
            worker_wall_seconds=elapsed, cumulative_worker_wall_seconds=self.prior_wall+elapsed,
            parameters=sum(p.numel() for p in self.model.parameters()), blocks=len(self.model.blocks),
            growth=self.growth, arm_fork=self.arm_fork, dev_history=self.history,
            new_parameter_max_abs_deltas=new_deltas, new_parameters_moved=any(v > 0 for v in new_deltas.values()),
            new_transition_parameters_moved=any(v > 0 for n, v in new_deltas.items() if '.head.' not in n),
            dataset_sha256=self.dataset_hash, final_model_sha256=digest_tensors(self.model),
            data_audit=self.data_audit,
            resources=self.resources, checker_used=False,
            limitations=['Synthetic typed semantics only; no arbitrary prose certificate.',
                'Parameter movement alone does not establish causal contribution or useful generalization.',
                'A short smoke run does not establish a development plateau.'])
        if self.a.device == 'cuda':
            report['resources'].update(cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                       cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        parameter_names = {id(p): n for n, p in self.model.named_parameters()}
        checkpoint = {'format_version': 1, 'source_sha256': self.resources['source_sha256'],
            'torch_version': str(torch.__version__), 'model': self.model.state_dict(),
            'training_config': self.training_config(), 'device_identity': self.resources,
            'optimizer': self.opt.state_dict(), 'optimizer_parameter_names': [
                [parameter_names[id(p)] for p in group['params']] for group in self.opt.param_groups],
            'width': self.a.width, 'blocks': len(self.model.blocks), 'step': self.step,
            'targets': self.targets, 'sequences': self.sequences, 'parameter_target_products': self.compute_estimate,
            'targets_by_kind': self.targets_by_kind,
            'cumulative_worker_wall_seconds': self.prior_wall+elapsed, 'history': self.history,
            'growth': self.growth, 'new_names': self.new_names, 'new_initial': self.new_initial,
            'dataset_sha256': self.dataset_hash, 'sample_rng': self.rng.getstate(),
            'data_audit': self.data_audit,
            'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if self.a.device == 'cuda' else None}
        buffer = io.BytesIO(); torch.save(cpu_tree(checkpoint), buffer)
        ckpt = buffer.getvalue()
        results = json.dumps(report, separators=(',', ':'), allow_nan=False).encode()
        if len(ckpt) + len(results) > 4*1024**2 - 512*1024:
            raise RuntimeError('Run artifact cap exceeded; no partial/truncated raw-output release.')
        for name, payload in [('checkpoint.pt', ckpt), ('results.json', results)]:
            tmp = self.output / (name + '.tmp'); tmp.write_bytes(payload); tmp.replace(self.output/name)
        manifest = {name: {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
                    for name, payload in [('checkpoint.pt', ckpt), ('results.json', results)]}
        (self.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def main():
    a = parse_args()
    runner = Runner(a)
    report = {'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              'status': 'complete', 'raw_evaluation': {}, 'loss_evaluation': {}}
    try:
        report['training_losses'] = runner.train()
        report['new_block_causal_ablation'] = runner.growth_ablation()
        for split in ('dev', 'test'):
            report['loss_evaluation'][split] = runner.loss_eval(split)
            report['raw_evaluation'][split] = runner.raw_eval(split)
    except BudgetExpired as exc:
        report['status'] = 'bounded_incomplete_evaluation'
        report['stop_reason'] = str(exc)
        report['evaluation_completeness'] = 'Incomplete; rerun zero-step evaluation to obtain all rows.'
        if runner.partial_raw is not None:
            report['raw_evaluation'][runner.partial_raw['split']] = runner.partial_raw
    runner.save(report)
    print(json.dumps({'status': report['status'], 'step': runner.step, 'output': str(runner.output),
                      'manifest': str(runner.output/'manifest.json')}, sort_keys=True))


if __name__ == '__main__':
    main()
