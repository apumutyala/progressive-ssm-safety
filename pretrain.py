"""Ordinary-corpus-only byte next-token pretraining with on-path dynamics audits.

No corpus text, token windows, prompts, or corpus generations are written to
results. Checkpoints are private resumable artifacts, not publication evidence.
"""
from __future__ import annotations
import time
STARTED = time.monotonic()
import argparse
import copy
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import shutil

import torch
import torch.nn.functional as F
from corpus_io import Corpus
from corpus_model import make_model, BOS, EOS, CONFIRM, DEFER, VOCAB_SIZE

ROOT = Path(__file__).resolve().parent
POLICY = {'seed': 12, 'width': 16, 'batch_size': 8, 'content_context': 128,
          'endpoint': 128, 'lr': .003, 'clip': 1., 'eval_windows': 16,
          'synthetic_content_fraction': 0., 'sampler': 'stateless_corpus_io_step_seed',
          'loss': 'mean_nonpadding_next_token_ce_including_explicit_eos'}
LOAD_CAP = 512*1024


class BudgetExpired(RuntimeError): pass
def sha(x): return hashlib.sha256(x).hexdigest()
def canonical(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def cpu(x):
    if torch.is_tensor(x): return x.detach().cpu()
    if isinstance(x, dict): return {k: cpu(v) for k, v in x.items()}
    if isinstance(x, tuple): return tuple(cpu(v) for v in x)
    if isinstance(x, list): return [cpu(v) for v in x]
    return x


def state_digest(value):
    """Hash tensor bytes, shapes, types and primitive structure, independent of pickle."""
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            v = x.detach().cpu().contiguous()
            h.update(canonical(['tensor', str(v.dtype), list(v.shape)])); h.update(v.numpy().tobytes())
        elif isinstance(x, dict):
            h.update(b'dict')
            for k in sorted(x, key=lambda y: (type(y).__name__, str(y))): visit(k); visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode()); h.update(str(len(x)).encode())
            for item in x: visit(item)
        else: h.update(canonical([type(x).__name__, x]))
    visit(value)
    return h.hexdigest()


def exact_equal(left, right):
    """Independent structural equality for actual checkpoint tensors, not file digests."""
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.cpu(), right.cpu()))
    if type(left) is not type(right): return False
    if isinstance(left, dict): return left.keys() == right.keys() and all(exact_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)): return len(left) == len(right) and all(exact_equal(a, b) for a, b in zip(left, right))
    return left == right


def compare_checkpoint_states(left, right):
    """Compare already safely loaded full/resumed payloads; wall time is intentionally excluded."""
    keys = ('model', 'optimizer', 'torch_rng', 'cuda_rng', 'python_rng', 'sampling_digest',
            'step', 'byte_targets', 'framing_targets', 'sources', 'recipe', 'corpus_receipt', 'window_spans')
    fields = {k: exact_equal(left[k], right[k]) for k in keys}
    return {'exact_equal': all(fields.values()), 'fields': fields}


def finite(x, label):
    if not torch.isfinite(x).all().item(): raise RuntimeError(f'Nonfinite {label}.')


def audit_ordinary_output(logits, targets, stats, certified):
    """Measure the unsupported reserved-decision invariant on actual corpus logits."""
    values = logits.detach()
    mask = targets != -100
    margins = values[..., DEFER]-values[..., CONFIRM]
    finite(margins, 'ordinary unsupported decision margins')
    minimum = float(margins[mask].min())
    unsafe = int(((values.argmax(-1) == CONFIRM) & mask).sum())
    stats.update(ordinary_defer_minus_confirm_min=minimum,
                 ordinary_unsupported_confirm_argmax=unsafe,
                 ordinary_output_vectors=int(mask.sum()))
    if certified and (minimum <= 0 or unsafe):
        raise RuntimeError('Certified ordinary-text unsupported output invariant failed.')
    return stats


def pack(rows, device):
    length = max(len(r['ids']) for r in rows)-1
    ids = torch.full((len(rows), length), EOS, dtype=torch.long, device=device)
    targets = torch.full_like(ids, -100)
    for i, row in enumerate(rows):
        values = row['ids']
        if values[0] != BOS or any(not 0 <= token < VOCAB_SIZE for token in values):
            raise RuntimeError('Invalid byte corpus framing/token domain.')
        if any(token < 8 and token != EOS for token in values[1:]):
            raise RuntimeError('Protocol or synthetic content token in ordinary corpus.')
        n = len(values)-1
        ids[i, :n] = torch.tensor(values[:-1], dtype=torch.long, device=device)
        targets[i, :n] = torch.tensor(values[1:], dtype=torch.long, device=device)
    return ids, targets


class AuditAggregate:
    def __init__(self):
        self.calls = 0; self.scalars = {}; self.lo = []; self.hi = []; self.outside = []
        self.margin_min = None; self.output_vectors = self.unsafe_confirms = 0
    def add(self, stats, certified):
        if not stats['parameters_finite']: raise RuntimeError('Nonfinite admitted parameters.')
        if certified and (stats['outside_box'] or stats['retention_above_one'] or stats['max_abs_drive'] > 1
                or stats['retention_min'] < 1/256 or stats['retention_max'] > 255/256):
            raise RuntimeError('Certified effective-weight/state-family invariant failed on executed path.')
        self.calls += 1
        if 'ordinary_defer_minus_confirm_min' in stats:
            margin = stats['ordinary_defer_minus_confirm_min']
            self.margin_min = margin if self.margin_min is None else min(self.margin_min, margin)
            self.output_vectors += stats['ordinary_output_vectors']
            self.unsafe_confirms += stats['ordinary_unsupported_confirm_argmax']
        for key, value in stats.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                old = self.scalars.setdefault(key, {'min': value, 'max': value})
                old['min'] = min(old['min'], value); old['max'] = max(old['max'], value)
        for i, (low, high, count) in enumerate(zip(stats['hidden_min_by_time'], stats['hidden_max_by_time'], stats['outside_box_by_time'])):
            if i == len(self.lo): self.lo.append(low); self.hi.append(high); self.outside.append(count)
            else:
                self.lo[i] = min(self.lo[i], low); self.hi[i] = max(self.hi[i], high); self.outside[i] += count
    def report(self):
        return {'calls': self.calls, 'scalar_min_max_over_calls': self.scalars,
                'min_ordinary_defer_minus_confirm': self.margin_min,
                'ordinary_output_vectors': self.output_vectors,
                'ordinary_unsupported_confirm_argmax_count': self.unsafe_confirms,
                'hidden_min_by_time_over_calls': self.lo, 'hidden_max_by_time_over_calls': self.hi,
                'outside_box_by_time_total': self.outside,
                'scope': 'All executed states in every audited batch, including any computed trailing padding. No audit clamp is applied.'}


class Worker:
    def __init__(self, a):
        self.a = a; self.deadline = STARTED+a.max_seconds; self.train_deadline = self.deadline-22
        self.output = a.output.resolve()
        self.limit = min(350*1024, int(os.environ.get('CORPUS_MAX_ARTIFACT_BYTES', 350*1024)))
        if self.limit <= 0 or self.output.exists() or self.output.is_relative_to(ROOT) or not self.output.parent.is_dir():
            raise RuntimeError('Fresh external output and positive corpus artifact allowance required.')
        self.disk()
        names = ('pretrain.py', 'corpus_model.py', 'corpus_io.py', 'SMOKE_PROTOCOL.md',
                 'corpus_sensitivity.py', 'control_probe.py', 'semantic_eval.py', 'model.py', 'interval_model.py')
        self.sources = {n: sha((ROOT/n).read_bytes()) for n in names}
        self.corpus = Corpus.load(a.manifest, a.manifest_sha256, corpus_root=a.corpus_root)
        self.receipt = self.corpus.receipt()
        self.recipe = dict(POLICY)
        self.step = self.byte_targets = self.framing_targets = 0
        self.sampling_digest = '0'*64; self.window_spans = []; self.prior_seconds = 0.
        self.initial_dev = None; self.initial_sha = None; self.losses = []
        self.private_semantic = None
        self.pre_audit = AuditAggregate(); self.post_audit = AuditAggregate()
        self.device = torch.device(a.device)
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        if str(torch.__version__) != '2.14.1+cu130' or torch.version.cuda != '13.0': raise RuntimeError('Reviewed Torch runtime mismatch.')
        random.seed(12); torch.manual_seed(12)
        self.resources = {'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda, 'python': sys.version,
            'torch_module': str(Path(torch.__file__).resolve()), 'device': a.device, 'tf32': False, 'deterministic': True}
        if a.device == 'cuda':
            if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'NVIDIA GeForce RTX 5070 Ti Laptop GPU': raise RuntimeError('Reviewed GPU identity mismatch.')
            free, total = torch.cuda.mem_get_info()
            if free < 768*1024**2: raise RuntimeError('CUDA headroom below768MiB.')
            self.resources.update(gpu=torch.cuda.get_device_name(0), free_vram_at_start=free, total_vram=total)
            torch.cuda.reset_peak_memory_stats()
        saved = None
        if a.resume:
            saved = self.load_checkpoint(a.resume, a.resume_sha256)
            if (saved.get('schema') != 1 or saved['arm'] != a.arm or saved['sources'] != self.sources
                    or saved['recipe'] != self.recipe or saved['corpus_receipt'] != self.receipt
                    or saved['torch'] != str(torch.__version__) or saved['device'] != a.device):
                raise RuntimeError('Exact-resume source/recipe/corpus/runtime/arm mismatch.')
            self.step = saved['step']; self.byte_targets = saved['byte_targets']; self.framing_targets = saved['framing_targets']
            self.sampling_digest = saved['sampling_digest']; self.window_spans = saved['window_spans']
            self.initial_dev = saved['initial_dev']; self.prior_seconds = saved['cumulative_worker_seconds']
        if self.step+a.steps > 128: raise RuntimeError('Frozen smoke endpoint is128 updates.')
        self.model = make_model(a.arm, width=16).to(self.device)
        self.initial_sha = state_digest(self.model.state_dict())
        self.opt = torch.optim.Adam(self.model.parameters(), lr=.003)
        if sum(p.numel() for p in self.model.parameters()) != 17432 or len(self.model.blocks) != 1:
            raise RuntimeError('Frozen one-block architecture count changed.')
        if saved:
            if saved['initial_tensor_sha256'] != self.initial_sha: raise RuntimeError('Seeded random initialization changed.')
            self.model.load_state_dict(saved['model'], strict=True); self.opt.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['torch_rng']); random.setstate(saved['python_rng'])
            if a.device == 'cuda' and saved['cuda_rng'] is not None: torch.cuda.set_rng_state_all(saved['cuda_rng'])
        self.model.validate_parameters(); self.check()

    @staticmethod
    def load_checkpoint(path, expected):
        path = path.resolve()
        if not 0 < path.stat().st_size <= LOAD_CAP: raise RuntimeError('Resume artifact size exceeds cap.')
        with path.open('rb') as f: raw = f.read(LOAD_CAP+1)
        if len(raw) > LOAD_CAP or sha(raw) != expected: raise RuntimeError('Expected checkpoint digest mismatch; not loaded.')
        return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)

    def disk(self):
        if shutil.disk_usage(self.output.parent).free < 256*1024**2: raise RuntimeError('Disk reserve below256MiB.')

    def check(self, training=False):
        if time.monotonic() >= (self.train_deadline if training else self.deadline-3): raise BudgetExpired('Bounded corpus worker reserve reached.')
        if self.a.device == 'cuda' and torch.cuda.max_memory_reserved() > 512*1024**2: raise RuntimeError('CUDA reserved-memory cap exceeded512MiB.')

    def train(self):
        if self.initial_dev is None: self.initial_dev = self.evaluate('dev')
        self.model.train()
        for _ in range(self.a.steps):
            self.check(training=True)
            rows = self.corpus.sample_batch(step=self.step, seed=12, batch_size=8, seq_len=128)
            if len(rows) != 8 or any(r['content_targets'] != 128 for r in rows): raise RuntimeError('Frozen byte batch contract changed.')
            x, y = pack(rows, self.device)
            self.model.validate_parameters(); self.opt.zero_grad(set_to_none=True)
            logits, before = self.model.audited(x); finite(logits, 'training logits')
            audit_ordinary_output(logits, y, before, self.a.arm == 'certified')
            self.pre_audit.add(before, self.a.arm == 'certified')
            losses = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='none', ignore_index=-100).view_as(y)
            mask = y != -100
            loss = losses[mask].mean(); finite(loss, 'training loss'); loss.backward()
            for parameter in self.model.parameters():
                if parameter.grad is not None: finite(parameter.grad, 'gradient')
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            self.opt.step(); self.model.validate_parameters()
            with torch.no_grad():
                after_logits, after = self.model.audited(x); finite(after_logits, 'post-update logits')
                audit_ordinary_output(after_logits, y, after, self.a.arm == 'certified')
            self.post_audit.add(after, self.a.arm == 'certified')
            byte_count = int((y >= 8).sum()); frame_count = int(((y >= 0) & (y < 8)).sum())
            self.step += 1; self.byte_targets += byte_count; self.framing_targets += frame_count
            descriptor = [(r['doc_id'], r['start'], r['end'], sha(canonical(r['ids']))) for r in rows]
            self.sampling_digest = sha(self.sampling_digest.encode()+canonical(descriptor))
            self.window_spans.extend((r['doc_id'], r['start'], r['end']) for r in rows)
            self.losses.append({'step': self.step, 'loss': float(loss.detach()), 'byte_targets': byte_count,
                'framing_targets': frame_count, 'pre_hidden_min': before['hidden_min'], 'pre_hidden_max': before['hidden_max'],
                'post_hidden_min': after['hidden_min'], 'post_hidden_max': after['hidden_max'],
                'pre_min_unsupported_margin': before['ordinary_defer_minus_confirm_min'],
                'post_min_unsupported_margin': after['ordinary_defer_minus_confirm_min'],
                'pre_unsupported_confirm_argmax': before['ordinary_unsupported_confirm_argmax'],
                'post_unsupported_confirm_argmax': after['ordinary_unsupported_confirm_argmax'],
                'pre_outside_box': before['outside_box'], 'post_outside_box': after['outside_box']})

    @torch.no_grad()
    def evaluate(self, split):
        if split == 'test' and self.step != 128: raise RuntimeError('Terminal corpus test is sealed until128 updates.')
        self.model.eval()
        rows = list(self.corpus.evaluation_windows(split, seq_len=128, max_windows=16))
        if not rows: raise RuntimeError('Empty source-disjoint holdout.')
        nll = byte_nll = 0.; targets = byte_targets = framing = 0
        audit = AuditAggregate()
        for start in range(0, len(rows), 8):
            self.check(); x, y = pack(rows[start:start+8], self.device)
            logits, stats = self.model.audited(x); finite(logits, 'holdout logits')
            audit_ordinary_output(logits, y, stats, self.a.arm == 'certified')
            audit.add(stats, self.a.arm == 'certified')
            losses = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='none', ignore_index=-100).view_as(y)
            mask = y != -100; byte_mask = y >= 8
            nll += float(losses[mask].sum()); byte_nll += float(losses[byte_mask].sum())
            targets += int(mask.sum()); byte_targets += int(byte_mask.sum()); framing += int((mask & (y < 8)).sum())
        return {'split': split, 'windows': len(rows), 'window_identity_sha256': sha(canonical([
            (r['doc_id'], r['start'], r['end'], sha(canonical(r['ids']))) for r in rows])),
            'targets': targets, 'ordinary_byte_targets': byte_targets, 'framing_targets': framing,
            'loss': nll/targets, 'byte_loss': byte_nll/byte_targets, 'bits_per_byte': byte_nll/(math.log(2)*byte_targets),
            'perplexity_per_byte': math.exp(min(byte_nll/byte_targets, 80)), 'audit': audit.report()}

    @torch.no_grad()
    def growth_audit(self):
        self.check()
        rows = self.corpus.sample_batch(step=0, seed=12, batch_size=8, seq_len=128)
        x, y = pack(rows, self.device)
        devices = [torch.cuda.current_device()] if self.a.device == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            disposable = copy.deepcopy(self.model).eval()
            old_logits, old_audit = disposable.audited(x); old_prefill = disposable.prefill(x)[0]
            old_count = sum(p.numel() for p in disposable.parameters())
            disposable.grow(); disposable.validate_parameters()
            new_logits, new_audit = disposable.audited(x); new_prefill = disposable.prefill(x)[0]
            finite(new_logits, 'grown logits'); finite(new_prefill, 'grown prefill logits')
            audit_ordinary_output(old_logits, y, old_audit, self.a.arm == 'certified')
            audit_ordinary_output(new_logits, y, new_audit, self.a.arm == 'certified')
            old_prefill_output = audit_ordinary_output(old_prefill, y, {}, self.a.arm == 'certified')
            new_prefill_output = audit_ordinary_output(new_prefill, y, {}, self.a.arm == 'certified')
            audit = AuditAggregate(); audit.add(old_audit, self.a.arm == 'certified'); audit.add(new_audit, self.a.arm == 'certified')
            equal = torch.equal(old_logits, new_logits) and torch.equal(old_prefill, new_prefill)
            if not equal: raise RuntimeError('Actual fitted checkpoint growth changed full unchanged-path logits.')
            return {'exact_equal': equal, 'forward_logit_coordinates': old_logits.numel(),
                'prefill_logit_coordinates': old_prefill.numel(), 'max_abs_delta': float((old_logits-new_logits).abs().max()),
                'old_parameters': old_count, 'new_parameters': sum(p.numel() for p in disposable.parameters()),
                'before_forward_output': {k: v for k, v in old_audit.items() if k.startswith('ordinary_')},
                'after_forward_output': {k: v for k, v in new_audit.items() if k.startswith('ordinary_')},
                'before_prefill_output': old_prefill_output, 'after_prefill_output': new_prefill_output,
                'audit': audit.report(), 'scope': 'Disposable fitted checkpoint copy on actual ordinary training bytes; no growth optimization or capability claim.'}

    @torch.no_grad()
    def suffix_audit(self):
        from corpus_sensitivity import verify_suffix
        self.check()
        rows = list(self.corpus.evaluation_windows('dev', seq_len=128, max_windows=8))
        content = [[t for t in r['ids'] if t >= 8] for r in rows]
        content = [r for r in content if len(r) >= 128][:8]
        if not content: return {'status': 'unavailable_no_complete_byte_window'}
        tokens = torch.tensor([r[:128] for r in content], dtype=torch.long, device=self.device)
        results = {}
        for t in (16, 64, 128):
            self.check(); results[str(t)] = verify_suffix(self.model, tokens[:, -t:])
        return {'status': 'complete', 'rows': len(content), 'ordinary_byte_tokens_only': True, 'by_length': results}

    def control_audit(self):
        from control_probe import analyze
        self.check()
        rows = list(self.corpus.evaluation_windows('dev', seq_len=128, max_windows=2))
        content = [[t for t in r['ids'] if t >= 8] for r in rows]
        content = [r for r in content if len(r) >= 16][:2]
        if not content: return {'status': 'unavailable_no_complete_byte_window'}
        tokens = torch.tensor([r[:16] for r in content], dtype=torch.long, device=self.device)
        return {'status': 'complete', 'ordinary_byte_rows': len(content), 'token_horizon': 16,
                'summary': analyze(self.model, tokens)}

    def semantic_audit(self):
        if self.a.semantic_fixture is None: return {'status': 'not_requested'}
        from semantic_eval import evaluate
        self.check()
        path = self.a.semantic_fixture.resolve()
        if not 0 < path.stat().st_size <= 65536: raise RuntimeError('Semantic fixture exceeds local-only size bound.')
        with path.open('rb') as f: raw = f.read(65537)
        if len(raw) > 65536 or sha(raw) != self.a.semantic_sha256: raise RuntimeError('Expected semantic fixture hash mismatch.')
        result = evaluate(self.model, path, device=self.device, max_new_tokens=48, check=self.check,
                          expected_sha256=self.a.semantic_sha256)
        self.private_semantic = result['private_audit']
        return result['aggregate']

    def save(self, report):
        self.disk(); seconds = time.monotonic()-STARTED
        cpu_rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state_all() if self.a.device == 'cuda' else None
        model_state = self.model.state_dict(); optimizer_state = self.opt.state_dict(); python_rng = random.getstate()
        hashes = {'model': state_digest(model_state), 'optimizer': state_digest(optimizer_state),
                  'torch_rng': state_digest(cpu_rng), 'cuda_rng': state_digest(cuda_rng), 'python_rng': state_digest(python_rng)}
        exposure = {}
        for doc, start, end in self.window_spans:
            exposure.setdefault(doc, []).append((start, end))
        coverage = {}
        for doc, intervals in exposure.items():
            union = []; current = None
            for start, end in sorted(set(intervals)):
                if current is None: current = [start, end]
                elif start <= current[1]: current[1] = max(current[1], end)
                else: union.append(current); current = [start, end]
            if current is not None: union.append(current)
            coverage[doc] = {'sampled_windows': len(intervals), 'distinct_windows': len(set(intervals)),
                'ordinary_target_exposures': sum(end-start for start, end in intervals),
                'unique_source_bytes': sum(end-start for start, end in union)}
        if sum(v['ordinary_target_exposures'] for v in coverage.values()) != self.byte_targets:
            raise RuntimeError('Actual source-byte exposure accounting disagrees with supervised targets.')
        self.resources['peak_reserved_cuda_bytes'] = torch.cuda.max_memory_reserved() if self.a.device == 'cuda' else 0
        report.update(arm=self.a.arm, step=self.step, initial_tensor_sha256=self.initial_sha,
            state_hashes=hashes, sources=self.sources, recipe=self.recipe, corpus_receipt=self.receipt,
            ordinary_corpus_targets=self.byte_targets, framing_targets=self.framing_targets,
            synthetic_content_targets=0, content_mix_fractions={'ordinary': 1., 'synthetic': 0.},
            sampling_digest=self.sampling_digest, sampled_windows=len(self.window_spans),
            distinct_sampled_windows=len(set(self.window_spans)), worker_seconds=seconds,
            source_byte_coverage=coverage,
            cumulative_worker_seconds=self.prior_seconds+seconds, resources=self.resources,
            before_update_dynamics=self.pre_audit.report(), after_update_dynamics=self.post_audit.report(),
            parameter_target_products=17432*(self.byte_targets+self.framing_targets),
            invariant_scope='Certified exact-real convex closure plus explicit fp32 state projection; audit records executed preprojection/projection counts. Comparator relaxes drive/state and decision bounds.',
            framing_scope='BOS resets each sampled window, including mid-document. EOS targets only at true source-fragment boundaries, counted separately from ordinary bytes and synthetic semantic content.',
            publication_scope='Aggregate corpus evidence only. No raw text/windows/prompts/corpus generations. Checkpoints remain private.')
        checkpoint = {'schema': 1, 'arm': self.a.arm, 'step': self.step, 'model': model_state, 'optimizer': optimizer_state,
            'recipe': self.recipe, 'sources': self.sources, 'corpus_receipt': self.receipt,
            'byte_targets': self.byte_targets, 'framing_targets': self.framing_targets,
            'sampling_digest': self.sampling_digest, 'window_spans': self.window_spans,
            'initial_dev': self.initial_dev, 'initial_tensor_sha256': self.initial_sha,
            'torch': str(torch.__version__), 'device': self.a.device,
            'torch_rng': cpu_rng, 'cuda_rng': cuda_rng, 'python_rng': python_rng,
            'cumulative_worker_seconds': self.prior_seconds+seconds}
        buffer = io.BytesIO(); torch.save(cpu(checkpoint), buffer)
        files = {'checkpoint.pt': buffer.getvalue(), 'results.json.gz': gzip.compress(canonical(report), compresslevel=6, mtime=0)}
        if self.private_semantic is not None:
            files['private_semantic.json.gz'] = gzip.compress(canonical(self.private_semantic), compresslevel=6, mtime=0)
        manifest = canonical({k: {'sha256': sha(v), 'bytes': len(v)} for k, v in files.items()})
        if sum(len(v) for v in files.values())+len(manifest) > self.limit: raise RuntimeError('Corpus run artifact allowance exceeded; no truncation/write.')
        self.disk(); self.output.mkdir()
        for name, content in files.items(): (self.output/name).write_bytes(content)
        (self.output/'manifest.json').write_bytes(manifest)
        print(json.dumps({'status': report['status'], 'arm': self.a.arm, 'step': self.step,
            'output': str(self.output), 'checkpoint_sha256': sha(files['checkpoint.pt']), 'artifact_bytes': sum(len(v) for v in files.values())+len(manifest)}, separators=(',', ':')))


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--arm', choices=['certified', 'unconstrained'], required=True); p.add_argument('--steps', type=int, default=64)
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--corpus-root', type=Path, required=True)
    p.add_argument('--resume', type=Path); p.add_argument('--resume-sha256')
    p.add_argument('--semantic-fixture', type=Path); p.add_argument('--semantic-sha256')
    p.add_argument('--max-seconds', type=float, default=100)
    a = p.parse_args()
    if not 0 <= a.steps <= 128 or not 25 <= a.max_seconds <= 110: p.error('Frozen updates0..128 and worker25..110 seconds.')
    for value in (a.manifest_sha256, a.resume_sha256 if a.resume else a.manifest_sha256):
        if not value or len(value) != 64 or any(c not in '0123456789abcdef' for c in value): p.error('Explicit lowercase SHA256 pins required.')
    if (a.semantic_fixture is None) != (a.semantic_sha256 is None): p.error('Semantic fixture requires its explicit SHA256.')
    if a.semantic_sha256 and (len(a.semantic_sha256) != 64 or any(c not in '0123456789abcdef' for c in a.semantic_sha256)):
        p.error('Semantic fixture requires a lowercase SHA256.')
    worker = Worker(a)
    report = {'status': 'complete', 'training_losses': worker.losses}
    try:
        worker.train(); report['initial_dev'] = worker.initial_dev; report['checkpoint_dev'] = worker.evaluate('dev')
        if worker.step == 128:
            report['terminal_test'] = worker.evaluate('test')
            report['growth_audit'] = worker.growth_audit()
            if a.arm == 'certified': report['common_suffix_audit'] = worker.suffix_audit()
            report['control_probe'] = worker.control_audit()
            report['semantic'] = worker.semantic_audit()
    except BudgetExpired as exc:
        report.update(status='bounded_incomplete', stop_reason=str(exc), initial_dev=worker.initial_dev)
    worker.save(report)


if __name__ == '__main__': main()
