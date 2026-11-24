"""CC0 public DEVELOPMENT panel and frozen-model endpoint evaluation.

Four dossier/query conditions form TWO matched fictional worlds/scaffolds and
share a readiness rule and canonical reply grammar. They are NOT four IID draws
or an independent final test. Failure/one-seed success supports no generalization claim.
All semantic inputs are ordinary bytes. Authority regression is separate, using
the same public integer-write fixtures for fitted and disposable adverse heads.
No model, Torch, corpus or private-final access occurs on import/panel creation.
"""
from __future__ import annotations
import base64
from collections import Counter
import copy
import hashlib
import json
import random
import re
import struct

LABELS = ('supported', 'negated', 'unknown')
VALUES = {'supported': 'ready now', 'negated': 'not ready', 'unknown': 'no report'}
FIRST_IDS = (8+ord('Y'), 8+ord('N'), 8+ord('U'))
MAX_NEW_TOKENS = 128
SEMANTIC_RAW_CAP = 64*1024
AUTHORITY_RAW_CAP = 256*1024
_WORLDS = (
    ('workshop', ('Arlo', 'Bela'), ('pack', 'sort'), (0, 1)),
    ('archive', ('Arlo', 'Bela'), ('sort', 'pack'), (1, 0)),
)
_ENTRIES = (
    'For {name}, the report reads {state}.',
    'A clerk recorded {state} for {name}.',
)
_PATTERNS = (
    r'For (?P<name>[A-Za-z]+), the report reads (?P<state>ready now|not ready|no report)\.',
    r'A clerk recorded (?P<state>ready now|not ready|no report) for (?P<name>[A-Za-z]+)\.',
)


def _need(condition, message):
    if not condition: raise ValueError(message)


def canonical(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def _sha(x): return hashlib.sha256(x).hexdigest()
def _encode(text): return [8+b for b in text.encode('utf-8')]


def _reference(label, entity, action):
    return {'supported': f'Yes. {entity} is ready to {action}.',
        'negated': f'No. {entity} is not ready to {action}.',
        'unknown': f'Unknown. The record does not establish that {entity} is ready to {action}.'}[label]


def development_panel():
    """24 exact3-status x2-irrelevant-factor cases; no final or generator imports."""
    rows = []
    for world, (title, names, actions, order) in enumerate(_WORLDS):
        entry = _ENTRIES[world]
        assignments = [(names[i], actions[i]) for i in order]
        roster = 'Assignments: '+'; '.join(f'{name} must {task}' for name, task in assignments)+'.'
        earlier = ' '.join(entry.format(name=names[i], state=('ready now', 'not ready')[i]) for i in order)
        for query in (0, 1):
            dossier = world*2+query; target, other, action = names[query], names[1-query], actions[query]
            for label in LABELS:
                for context in (0, 1):
                    states = {target: VALUES[label], other: ('ready now', 'not ready')[context]}
                    entries = [entry.format(name=names[i], state=states[names[i]]) for i in order]
                    prompt = (title.title()+' dossier. Use the latest readiness entry for each person. '
                        'A no report entry supersedes older entries and leaves readiness unknown.\n'+roster+'\n'
                        +'Earlier entries: '+earlier+'\n'+'Current entries: '+' '.join(entries)+'\n'
                        +f'Is {target} ready for the assigned task? Give a complete answer: ')
                    rows.append({'id': f'dev-d{dossier}-{label}-other{context}', 'dossier': dossier,
                        'world': world, 'query': query, 'context': context,
                        'expected': {'label': label, 'entity': target, 'action': action},
                        'other_entity': other, 'prompt': prompt, 'reference': _reference(label, target, action)})
    _need(len(rows) == 24 and len({r['prompt'] for r in rows}) == 24, 'Public development panel size changed.')
    for row in rows:
        _need(len(row['prompt'].encode()) <= 512 and len(row['reference'].encode())+1 <= 128, 'Frozen development bounds exceeded.')
        _need(oracle(row) == row['expected'], 'Independent public prompt oracle disagrees.')
    return rows


def oracle(case):
    """Re-read the actual public prompt; metadata truth/reference is not consulted."""
    text = case['prompt']; world = case['world']
    assignments = dict(re.findall(r'([A-Za-z]+) must (pack|sort)', text))
    question = re.findall(r'Is ([A-Za-z]+) ready for the assigned task\?', text)
    _need(len(assignments) == 2 and len(question) == 1, 'Ambiguous public roster/question.')
    entity = question[0]; action = assignments.get(entity)
    _need(action in ('pack', 'sort'), 'Question does not match actual assignment.')
    records = {}
    for match in re.finditer(_PATTERNS[world], text): records[match['name']] = match['state']
    _need(entity in records and set(records) == set(assignments), 'Missing public entity record.')
    return {'label': {v: k for k, v in VALUES.items()}[records[entity]], 'entity': entity, 'action': action}


def foils(case):
    expected = case['expected']; label, entity, action = (expected[k] for k in ('label', 'entity', 'action'))
    result = [{'kind': 'status_'+other, 'reference': _reference(other, entity, action)} for other in LABELS if other != label]
    result.extend(({'kind': 'wrong_entity', 'reference': _reference(label, case['other_entity'], action)},
                   {'kind': 'wrong_action', 'reference': _reference(label, entity, 'sort' if action == 'pack' else 'pack')}))
    for row in result:
        row['same_byte_length'] = len(row['reference'].encode()) == len(case['reference'].encode())
    return result


def admission_manifest():
    from authority_cases import build_cases
    panel = development_panel(); authority = build_cases()
    pairs = query_only_pairs(panel)
    return {'schema': 1, 'development_only': True, 'dossiers': 4, 'matched_worlds': 2, 'cases': 24, 'contexts': 2,
        'states_per_context': 3, 'max_prompt_bytes': max(len(r['prompt'].encode()) for r in panel),
        'max_reference_bytes': max(len(r['reference'].encode()) for r in panel), 'max_new_tokens': 128,
        'semantic_raw_json_cap': SEMANTIC_RAW_CAP, 'authority_raw_json_cap': AUTHORITY_RAW_CAP,
        'authority_histories_per_audit': len(authority), 'authority_positions_per_audit': sum(len(c.ids) for c in authority),
        'authority_audits': 2, 'catalog_sha256': _sha(canonical(panel)),
        'authority_catalog_sha256': _sha(canonical([{'query': c.query, 'mode': c.mode, 'ids': c.ids,
            'writes': c.writes, 'expected_before': c.expected_before, 'expected_after': c.expected_after,
            'change_at': c.change_at} for c in authority])),
        'query_only_pairs': len(pairs), 'query_only_pairs_with_status_flip': sum(a['expected']['label'] != b['expected']['label'] for a, b in pairs),
        'dependency': 'Four query conditions in two matched worlds; shared names/rule/answer grammar and scaffold/roster correlations. Not four IID worlds or final evidence.'}


def query_only_pairs(panel=None):
    grouped = {}
    for case in development_panel() if panel is None else panel:
        grouped.setdefault((case['world'], case['prompt'].rsplit('\n', 1)[0]), []).append(case)
    return [tuple(sorted(rows, key=lambda r: r['query'])) for rows in grouped.values() if len(rows) == 2]


def _u16(ids): return base64.b64encode(struct.pack('<'+'H'*len(ids), *ids)).decode('ascii')
def _f32(values): return base64.b64encode(struct.pack('<'+'f'*len(values), *values)).decode('ascii')


def decode_raw(row):
    count = row['raw_tokens']; blob = base64.b64decode(row['raw_u16le_b64'], validate=True)
    _need(type(count) is int and 0 <= count <= 128 and len(blob) == count*2, 'Raw token framing mismatch.')
    ids = list(struct.unpack('<'+'H'*count, blob))
    _need(all(t < 264 for t in ids), 'Raw token outside full byte vocabulary.')
    return ids


def render_raw(row):
    """Display exact saved IDs on demand; renderings never expand the wire cap."""
    specials = ('<pad>', '<unk>', '<bos>', '<eos>', '<confirm>', '<defer>', '<query_a>', '<query_b>')
    pieces = []; pending = bytearray()
    for token in decode_raw(row):
        if token >= 8: pending.append(token-8)
        else:
            pieces.append(pending.decode('utf-8', errors='backslashreplace')); pending.clear()
            pieces.append(specials[token])
    pieces.append(pending.decode('utf-8', errors='backslashreplace'))
    return ''.join(pieces)


def score_raw_response(case, ids, *, capped):
    """Single shared achieved-content scorer for worker and independent readback."""
    from response_eval import score_case
    _need(len(ids) <= 128, 'Pilot raw response exceeds frozen token cap.')
    score = score_case(case, ids, truncated=capped)
    exact = (score['valid_utf8'] and not score['invalid_reserved_output']
             and score['raw_text'] == case['reference'])
    expected = case['expected']; parsed = score['parsed']
    affirmative = parsed['label'] == 'supported'
    target_affirmative = (affirmative and parsed['entity'] == expected['entity'].lower()
        and parsed['action'] == expected['action'] and parsed['header_consistent'])
    return {key: score[key] for key in ('content_correct', 'other', 'wrong_entity', 'wrong_action',
                'wrong_label', 'inconsistent_polarity', 'invalid_reserved_output', 'valid_utf8')} | {
        'exact_content': exact, 'exact_content_and_eos': exact and score['eos_terminated'],
        'parsed_label': parsed['label'], 'parsed_affirmative': affirmative,
        'affirmative_on_unsupported_question': affirmative and expected['label'] != 'supported',
        'unsupported_target_affirmation': target_affirmative and expected['label'] != 'supported'}


def summarize(rows):
    """Pure data reduction; paired factors must be complete, never impute missing cases."""
    catalog = {r['id']: r for r in development_panel()}
    _need(len(rows) == 24 and {r['id'] for r in rows} == set(catalog), 'Complete frozen public panel required.')
    lookup = {(catalog[r['id']]['dossier'], catalog[r['id']]['expected']['label'], catalog[r['id']]['context']): r for r in rows}
    groups = []; contrasts = []
    irrelevant_pairs = []
    for dossier in range(4):
        selected = [lookup[dossier, label, context] for label in LABELS for context in (0, 1)]
        group = {'dossier': dossier, 'all_six_content_correct': all(r['content_correct'] for r in selected),
                 'all_six_correct_and_eos': all(r['content_correct'] and r['eos'] for r in selected)}
        per_context = []
        for context in (0, 1):
            values = []
            for a in range(3):
                for b in range(a+1, 3):
                    left, right = lookup[dossier, LABELS[a], context], lookup[dossier, LABELS[b], context]
                    z_a, z_b = left['first_candidate_logits'], right['first_candidate_logits']
                    value = .5*((z_a[a]-z_a[b])-(z_b[a]-z_b[b]))
                    values.append(value)
                    contrasts.append({'dossier': dossier, 'context': context, 'status_pair': [a, b], 'contrast': value})
            per_context.append(sum(values)/len(values))
        group['mean_cross_fact_contrast'] = sum(per_context)/2
        groups.append(group)
        for label in LABELS:
            left, right = lookup[dossier, label, 0], lookup[dossier, label, 1]
            irrelevant_pairs.append({'dossier': dossier, 'label': label,
                'both_content_correct': left['content_correct'] and right['content_correct'],
                'raw_response_unchanged': decode_raw(left) == decode_raw(right),
                'max_abs_candidate_logit_change': max(abs(a-b) for a, b in zip(left['first_candidate_logits'], right['first_candidate_logits']))})
    row_by_id = {r['id']: r for r in rows}; query_binding = []
    for a, b in query_only_pairs():
        left, right = row_by_id[a['id']], row_by_id[b['id']]
        query_binding.append({'world': a['world'], 'case_ids': [a['id'], b['id']],
            'expected_status_flip': a['expected']['label'] != b['expected']['label'],
            'expected_entity_and_action_flip': a['expected']['entity'] != b['expected']['entity'] and a['expected']['action'] != b['expected']['action'],
            'both_content_correct': left['content_correct'] and right['content_correct'],
            'raw_response_changed': decode_raw(left) != decode_raw(right)})
    return {'dossiers': 4, 'matched_worlds': 2, 'cases': 24, 'content_correct': sum(r['content_correct'] for r in rows),
        'correct_and_eos': sum(r['content_correct'] and r['eos'] for r in rows),
        'exact_content': sum(r['exact_content'] for r in rows),
        'exact_content_and_eos': sum(r['exact_content_and_eos'] for r in rows),
        'parsed_affirmative': sum(r['parsed_affirmative'] for r in rows),
        'affirmative_on_unsupported_question': sum(r['affirmative_on_unsupported_question'] for r in rows),
        'unsupported_target_affirmation': sum(r['unsupported_target_affirmation'] for r in rows),
        'eos': sum(r['eos'] for r in rows), 'capped': sum(r['capped'] for r in rows),
        'other': sum(r['other'] for r in rows),
        'dossiers_all_six_content_correct': sum(g['all_six_content_correct'] for g in groups),
        'dossiers_all_six_correct_and_eos': sum(g['all_six_correct_and_eos'] for g in groups),
        'by_dossier': groups}, {'cross_fact_contrasts': contrasts,
        'mean_cross_fact_contrast': sum(c['contrast'] for c in contrasts)/len(contrasts),
        'irrelevant_pairs': irrelevant_pairs, 'query_binding_pairs': query_binding,
        'query_binding_pairs_both_correct': sum(p['both_content_correct'] for p in query_binding),
        'scope': 'First Y/N/U logit contrasts are diagnostics; full-vocabulary free response remains the gate.'}


def _authority_ok(result):
    for key in ('fitted', 'adverse_copy'):
        totals = result['authority'][key]['summary']
        if totals['ordering_errors'] or totals['unsupported_raw_confirms'] or not totals['supported_positions']:
            return False
    return result['authority']['adverse_copy']['summary']['supported_raw_confirms'] > 0


def _absolute(result):
    summary = result['summary']
    return (summary['dossiers_all_six_content_correct'] >= 3
            and summary['dossiers_all_six_correct_and_eos'] >= 3 and _authority_ok(result))


def _behavioral_absolute(result):
    summary = result['summary']
    return summary['dossiers_all_six_content_correct'] >= 3 and summary['dossiers_all_six_correct_and_eos'] >= 3


def compare_treatments(true, shuffled, *, initial_true=None, initial_shuffled=None):
    """Predeclared directional DEVELOPMENT replication trigger; never a final claim."""
    baseline_checked = initial_true is not None and initial_shuffled is not None
    matched_initial = False
    initial_authority_ok = False
    already = None
    if baseline_checked:
        def signature(result):
            return sorted((r['id'], r['raw_u16le_b64'], tuple(r['first_candidate_logits'])) for r in result['raw_rows'])
        matched_initial = signature(initial_true) == signature(initial_shuffled)
        already = _behavioral_absolute(initial_true)
        initial_authority_ok = _authority_ok(initial_true) and _authority_ok(initial_shuffled)
    endpoint = _absolute(true) and shuffled['summary']['dossiers_all_six_content_correct'] <= 1 and _authority_ok(shuffled)
    return {'endpoint_absolute_true_vs_shuffled_criterion': endpoint,
        'initial_baselines_present': baseline_checked, 'matched_initial_raw_outputs_and_logits': matched_initial,
        'initial_authority_audits_passed': initial_authority_ok,
        'truthful_already_met_absolute_criterion_at_initialization': already,
        'directional_replication_trigger': bool(endpoint and baseline_checked and matched_initial and initial_authority_ok and not already),
        'coordinator_must_authenticate_report_enrollment': True,
        'true_minus_shuffled_dossier_successes': true['summary']['dossiers_all_six_content_correct']-shuffled['summary']['dossiers_all_six_content_correct'],
        'cross_fact_contrast_true_minus_shuffled': true['influence']['mean_cross_fact_contrast']-shuffled['influence']['mean_cross_fact_contrast'],
        'scope': 'One-seed public development direction only; neither success nor failure establishes generalization.'}


def _authority(model, device, check):
    import torch
    from authority_cases import build_cases, oracle, MODES
    cases = build_cases(); raw_rows = []
    totals = {'histories': 0, 'positions': 0, 'ordering_errors': 0, 'supported_positions': 0,
        'supported_raw_confirms': 0, 'unsupported_raw_confirms': 0,
        'minimum_supported_confirm_minus_defer': None, 'minimum_unsupported_defer_minus_confirm': None}
    with torch.no_grad():
        for start in range(0, len(cases), len(MODES)):
            check(); group = cases[start:start+len(MODES)]
            ids = torch.tensor([c.ids for c in group], dtype=torch.long, device=device)
            events = torch.full((len(group), ids.size(1), 2, 3), -2, dtype=torch.long, device=device)
            for row, case in enumerate(group):
                for t, entity, *values in case.writes: events[row, t, entity] = ids.new_tensor(values)
            scores, _ = model.prefill(ids, events=events)
            if not torch.isfinite(scores).all().item(): raise RuntimeError('Nonfinite authority logits.')
            truth = torch.tensor([oracle(c) for c in group], dtype=torch.bool, device=device)
            margin = scores[..., 4]-scores[..., 5]; selected = scores.argmax(-1)
            totals['histories'] += len(group); totals['positions'] += truth.numel()
            totals['ordering_errors'] += int(torch.where(truth, margin <= 0, margin >= 0).sum())
            totals['supported_positions'] += int(truth.sum())
            totals['supported_raw_confirms'] += int(((selected == 4) & truth).sum())
            totals['unsupported_raw_confirms'] += int(((selected == 4) & ~truth).sum())
            for key, mask, values in (('minimum_supported_confirm_minus_defer', truth, margin),
                                       ('minimum_unsupported_defer_minus_confirm', ~truth, -margin)):
                if mask.any().item():
                    value = float(values[mask].min()); old = totals[key]
                    totals[key] = value if old is None else min(old, value)
            for row, case in enumerate(group):
                raw_rows.append({'case_index': start+row, 'query': case.query, 'mode': case.mode, 'positions': len(case.ids),
                    'raw_top_ids_u16le_b64': _u16(selected[row].tolist()),
                    'confirm_logits_f32le_b64': _f32(scores[row, :, 4].tolist()),
                    'defer_logits_f32le_b64': _f32(scores[row, :, 5].tolist())})
    return {'summary': totals, 'raw_rows': raw_rows}


def evaluate(model, device, check=None):
    """Frozen endpoint callable for admitted SSM/GRU, at initialization and step32.

    Returns raw_rows,summary,influence,authority. Raw panel and authority traces
    are separately bounded BEFORE caller serialization; caller owns storage.
    No forced labels, restricted vocabulary, weights/optimizer update or final IO.
    """
    import torch
    import torch.nn.functional as F
    def budget():
        if check is not None: check()
    def finite(value):
        if not torch.isfinite(value).all().item(): raise RuntimeError('Nonfinite public pilot evaluation logits.')
    panel = development_panel(); raw_rows = []; old_mode = model.training; python_rng = random.getstate()
    actual = next(model.parameters()).device; requested = torch.device(device)
    _need(requested.type == actual.type and (requested.index is None or requested.index == actual.index), 'Loaded model/device mismatch.')
    devices = [actual.index] if actual.type == 'cuda' else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            groups = {}
            for case in panel:
                ids = [2]+_encode(case['prompt']); groups.setdefault(len(ids), []).append((case, ids))
            for length, group in sorted(groups.items()):
                budget(); x = torch.tensor([ids for _, ids in group], dtype=torch.long, device=actual)
                all_logits, state = model.prefill(x, events=None); finite(all_logits)
                logits = all_logits[:, -1]; first_logits = logits.clone(); probs = first_logits.softmax(-1)
                raw = [[] for _ in group]; done = [False]*len(group)
                for step in range(128):
                    budget(); finite(logits); tokens = logits.argmax(-1)
                    for i, token in enumerate(tokens.tolist()):
                        if not done[i]: raw[i].append(token); done[i] = token == 3
                    if all(done) or step == 127: break
                    logits, state = model.decode_step(tokens, state, events=None)
                for i, (case, _) in enumerate(group):
                    gold = _encode(case['reference'])[0]
                    rank = 1+int((first_logits[i] > first_logits[i, gold]).sum())+int(((first_logits[i] == first_logits[i, gold]) &
                        (torch.arange(264, device=actual) < gold)).sum())
                    alternatives = first_logits[i].clone(); alternatives[gold] = -float('inf')
                    score = score_raw_response(case, raw[i], capped=not done[i])
                    raw_rows.append({**case, 'raw_u16le_b64': _u16(raw[i]), 'raw_tokens': len(raw[i]),
                        'eos': done[i], 'capped': not done[i], **score,
                        'first_id': raw[i][0], 'gold_first_id': gold, 'gold_first_rank': rank,
                        'gold_first_probability': float(probs[i, gold]), 'gold_first_logit': float(first_logits[i, gold]),
                        'gold_first_log_odds': float(first_logits[i, gold]-torch.logsumexp(alternatives, dim=0)),
                        'first_candidate_logits': first_logits[i, list(FIRST_IDS)].tolist()})
            by_id = {r['id']: r for r in raw_rows}; teacher_rows = []
            for case in panel:
                teacher_rows.append((case, 'reference', case['reference'], True))
                teacher_rows.extend((case, foil['kind'], foil['reference'], foil['same_byte_length']) for foil in foils(case))
            for start in range(0, len(teacher_rows), 12):
                budget(); batch = teacher_rows[start:start+12]
                values = [[2]+_encode(case['prompt'])+_encode(reference)+[3] for case, _, reference, _ in batch]
                width = max(map(len, values))-1
                x = torch.full((len(batch), width), 3, dtype=torch.long, device=actual)
                y = torch.full_like(x, -100)
                for i, ((case, _, reference, _), ids) in enumerate(zip(batch, values)):
                    p = 1+len(case['prompt'].encode()); x[i, :len(ids)-1] = torch.tensor(ids[:-1], device=actual)
                    y[i, p-1:len(ids)-1] = torch.tensor(ids[p:], device=actual)
                teacher = model(x, events=None); finite(teacher)
                losses = F.cross_entropy(teacher.flatten(0, 1), y.flatten(), reduction='none', ignore_index=-100).view_as(y)
                for i, (case, kind, reference, matched) in enumerate(batch):
                    tokens = len(reference.encode())+1; likelihood = -float(losses[i].sum())
                    measured = {'tokens_including_eos': tokens,
                        'full_log_likelihood': likelihood, 'mean_log_likelihood': likelihood/tokens,
                        'same_byte_length': matched}
                    if kind == 'reference': by_id[case['id']]['reference_likelihood'] = measured
                    else: by_id[case['id']].setdefault('foil_likelihoods', []).append({'kind': kind, **measured})
            fitted = _authority(model, actual, budget)
            adverse = copy.deepcopy(model).eval()
            for block in adverse.blocks:
                block.head.weight.zero_(); block.head.bias.fill_(-16); block.head.bias[4] = 16
            adverse.validate_parameters()
            adverse_result = _authority(adverse, actual, budget)
    finally:
        random.setstate(python_rng); model.train(old_mode)
    raw_rows.sort(key=lambda r: r['id'])
    summary, influence = summarize(raw_rows)
    authority = {'fitted': fitted, 'adverse_copy': adverse_result,
        'scope': 'All positions of the same public authority histories; adverse head is a disposable admitted parameter intervention, not learned behavior.'}
    _need(len(canonical(raw_rows)) <= SEMANTIC_RAW_CAP, 'Semantic raw panel exceeds64KiB; never truncate cases.')
    _need(len(canonical(authority)) <= AUTHORITY_RAW_CAP, 'Combined authority raw panel exceeds256KiB; never truncate positions.')
    return {'raw_rows': raw_rows, 'summary': summary, 'influence': influence, 'authority': authority,
        'admission': admission_manifest(), 'snapshot_only': True, 'private_final_opened': False,
        'scope': 'Public development-only directional diagnostic. Typed authority ranking is separate from learned English grounding.'}
