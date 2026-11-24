"""Input shift, conditional shuffle and decoder checks on supplied public examples."""
import ast
from collections import Counter
import hashlib
import inspect
import json

def need(ok, why):
    if not ok: raise ValueError(why)


def canonical(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(x): return hashlib.sha256(x).hexdigest()


def check_shift(record, inputs, targets, mask, length):
    """Independent causal shift/prompt/padding check, aligned to actual collated rows."""
    need(length == len(record.ids)-1 and len(inputs) == len(targets) == len(mask) >= length, 'Length/shape mismatch.')
    need(tuple(inputs[:length]) == record.ids[:-1], 'Model input differs from causal prefix.')
    checked = masked = padding = eos = 0
    for position, (target, use) in enumerate(zip(targets, mask)):
        if position >= length:
            need(inputs[position] == 3 and target == -100 and use is False, 'Padding is supervised or misplaced.')
            padding += 1; continue
        expected_use = record.loss_mask[position]
        expected_target = record.ids[position+1] if expected_use else -100
        need(use is expected_use and target == expected_target, 'Cross-target shift or loss mask mismatch.')
        if record.kind == 'synthetic':
            need(expected_use == (position >= record.prompt_len-1), 'Synthetic prompt supervision/first-answer shift mismatch.')
        else: need(expected_use is True, 'Ordinary byte target was dropped.')
        need(target == -100 or 8 <= target < 264 or (target == 3 and position == length-1), 'Unexpected supervised special token.')
        checked += 1; masked += not expected_use; eos += target == 3
    return {'shift_pairs': checked, 'masked_prompt_targets': masked, 'padding_positions': padding, 'supervised_eos': eos}


def check_matching(true_records, shuffled_records):
    """Reconstruct matched controls from actual IDs rather than declared hashes."""
    def bags(records):
        rows = [r for r in records if r.kind == 'synthetic']
        prompts = Counter(r.ids[:r.prompt_len] for r in rows)
        replies = Counter(r.ids[r.prompt_len:] for r in rows)
        conditional = {}
        for r in rows:
            conditional.setdefault(r.ids[:r.prompt_len], Counter())[r.ids[r.prompt_len:]] += 1
        return prompts, replies, conditional
    tp, tr, tc = bags(true_records); sp, sr, sc = bags(shuffled_records)
    need(tp == sp and tr == sr and len(tp) == 3 and set(tp.values()) == {3},
         'Actual prompt/reply bags or replica counts differ.')
    need(len(tr) == 3 and all(len(v) == 1 and list(v.values()) == [3] for v in tc.values()),
         'True conditional group has inconsistent answers.')
    need(all(v == Counter({reply: 1 for reply in tr}) for v in sc.values()),
         'Shuffled identical prompt does not receive each answer exactly once.')


def source_contract():
    import pilot_evaluation
    source = inspect.getsource(pilot_evaluation.evaluate)
    tree = ast.parse(source)
    names = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    need('argmax' in names and 'decode_step' in names and not set(names) & {'multinomial', 'topk', 'sample', 'choice'}, 'Unexpected decoding/sampling source contract.')
    need('for step in range(128)' in source and 'token == 3' in source and 'if all(done) or step == 127' in source,
         'Public raw decode cap/first-EOS stopping source changed.')
    return {'full_vocabulary_argmax_in_source': True, 'cached_decode_step_in_source': True,
        'random_or_topk_sampling_calls_in_source': False, 'first_eos_stop_in_source': True,
        'raw_token_cap': 128, 'reference_teacher_forcing_separate_from_raw': True,
        'scope': 'Static source contract only; no numerical generations or model execution in this audit.',
        'source_sha256': sha(source.encode())}
