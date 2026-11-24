"""CC0 public TRAIN/DEV readiness-and-action curriculum; no final holdout.

Raw ordinary byte IDs are the only semantic model input. Standard BOS/EOS frame
independent LM rows; no query/type IDs, evidence events, status labels, nuisance
metadata or hidden replica index enter the prompt. Reset state at every row.

Each nuisance group contains evidence status3 x hidden replica3 =9 rows. True
targets agree with the record. The shuffled arm uses a frozen conditional Latin
permutation: for EACH identical prompt all three possible replies occur once.
Thus its label is exactly independent of evidence/full input within a group;
it is not a deterministic flip/inverse. Reply strings, lengths, lexical counts,
entity/action frequencies and input rows match exactly between arms per group.

Semantics are narrow: readiness applies to the named entity's assigned pack/sort
task. 'not ready' contradicts readiness; 'no report' establishes neither ready
nor not-ready. This is learned data semantics, not an architectural certificate.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import itertools
import json
import random
import re

PAD, UNK, BOS, EOS, CONFIRM, DEFER, QUERY_A, QUERY_B = range(8)
BYTE_OFFSET, VOCAB_SIZE = 8, 264
STATUSES = ('supported', 'contradicted', 'unknown')
VALUES = ('ready now', 'not ready', 'no report')
TASKS = ('pack', 'sort')
NAMES = {'train': ('Mira', 'Noel'), 'dev': ('Luma', 'Venn')}
DATA_SEEDS = {'train': 95173, 'dev': 95179}
ARMS = ('true', 'shuffled')
HISTORIES = ('single', 'reversal', 'retained', 'irrelevant_update')


def encode(text):
    return [BYTE_OFFSET+b for b in text.encode('utf-8')]


def decode(ids):
    if any(type(t) is not int or not BYTE_OFFSET <= t < VOCAB_SIZE for t in ids):
        raise ValueError('ordinary byte IDs required')
    return bytes(t-BYTE_OFFSET for t in ids).decode('utf-8')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def reply(status, name, task):
    if status == 'supported':
        return f'Yes. {name} is ready to {task}.'
    if status == 'contradicted':
        return f'No. {name} is not ready to {task}.'
    if status == 'unknown':
        return f'Unknown. The record does not establish that {name} is ready to {task}.'
    raise ValueError('unknown reply status')


def _prompt(split, form, names, tasks, facts, query, assignment_order, fact_order, history='single'):
    ao = (1, 0) if assignment_order else (0, 1)
    fo = (1, 0) if fact_order else (0, 1)
    name = names[query]
    records = [(i, facts[i]) for i in fo]
    if history in ('reversal', 'retained'):
        old = facts[query] if history == 'retained' else ('not ready' if facts[query] == 'ready now' else 'ready now')
        records = [(query, old), (1-query, facts[1-query])] if not fact_order else [(1-query, facts[1-query]), (query, old)]
        records.append((query, facts[query]))
    elif history == 'irrelevant_update':
        old = 'not ready' if facts[1-query] == 'ready now' else 'ready now'
        records = [(query, facts[query]), (1-query, old)] if not fact_order else [(1-query, old), (query, facts[query])]
        records.append((1-query, facts[1-query]))
    elif history != 'single':
        raise ValueError('unknown public history')
    convention = ('Use the last readiness entry for each name. ' if split == 'train' else
                  'For each person, the newest readiness note replaces the earlier note. ')
    if split == 'train' and form == 0:
        return (convention+'Tasks: '+''.join(f'{names[i]} {tasks[i]}. ' for i in ao)+
                'Record: '+''.join(f'{names[i]}: {value}. ' for i, value in records)+
                f'Question: Can {name} do the assigned task? Answer: ')
    if split == 'train' and form == 1:
        return (convention+'Job list: '+'; '.join(f'{tasks[i]} for {names[i]}' for i in ao)+'. '+
                'State list: '+'; '.join(f'{value} for {names[i]}' for i, value in records)+'. '+
                f'Question: For the assigned task, can {name} proceed? Reply: ')
    if split == 'dev' and form == 0:
        return (convention+'Assignments are '+' and '.join(f'{names[i]}={tasks[i]}' for i in ao)+'. '+
                'Reports are '+'; '.join(f'{names[i]}: {value}' for i, value in records)+'. '+
                f'For {name}, does its assigned task have support? Response: ')
    if split == 'dev' and form == 1:
        return (convention+'; '.join(f'{names[i]} has task {tasks[i]}' for i in ao)+'. '+
                'Latest notes: '+'; '.join(f'{names[i]} [{value}]' for i, value in records)+'. '+
                f'Is the task for {name} supported by these notes? Response: ')
    raise ValueError('only public train/dev forms are available')


def _nuisances():
    # No evidence status or hidden replicate is a nuisance/model-visible label.
    return itertools.product(range(2), range(2), range(2), range(2), range(2), range(3), HISTORIES)


def _group(split, nuisance, arm):
    form, query, assignment, assignment_order, fact_order, other_status, history = nuisance
    names = NAMES[split]
    tasks = TASKS if assignment == 0 else tuple(reversed(TASKS))
    group_id = f'{split}-'+''.join(str(i) for i in nuisance)
    rng = random.Random(int(_digest([DATA_SEEDS[split], list(nuisance)]), 16))
    evidence_order, replica_order, label_order = list(range(3)), list(range(3)), list(range(3))
    rng.shuffle(evidence_order)
    rng.shuffle(replica_order)
    rng.shuffle(label_order)
    rows = []
    for evidence in range(3):
        facts = [None, None]
        facts[query], facts[1-query] = VALUES[evidence], VALUES[other_status]
        prompt = _prompt(split, form, names, tasks, facts, query, assignment_order, fact_order, history)
        for replica in range(3):
            label = evidence if arm == 'true' else label_order[(evidence_order.index(evidence)+replica_order[replica]) % 3]
            reference = reply(STATUSES[label], names[query], tasks[query])
            pids, rids = [BOS]+encode(prompt), encode(reference)+[EOS]
            rows.append({'id': f'{group_id}-e{evidence}-r{replica}', 'group': group_id,
                'split': split, 'arm': arm, 'replica': replica,
                'prompt': prompt, 'reference': reference, 'prompt_ids': encode(prompt),
                'response_ids': encode(reference), 'ids': pids+rids, 'prompt_len': len(pids),
                'response_targets': len(rids), 'ordinary_response_targets': len(rids)-1,
                'evidence_status': STATUSES[evidence], 'label_status': STATUSES[label],
                'entity': names[query], 'action': tasks[query], 'query': query,
                'model_input_fields': ['prompt_ids'],
                'nuisance': {'form': form, 'query': query, 'assignment': assignment,
                    'assignment_order': assignment_order, 'fact_order': fact_order,
                    'other_status': other_status, 'history': history}})
    return {'id': group_id, 'split': split, 'arm': arm, 'rows': rows,
            'target_histogram': {s: 3 for s in STATUSES},
            'response_targets': sum(r['response_targets'] for r in rows)}


def iter_groups(split, seed=0, arm='true'):
    """Yield384 complete nine-row groups; same ordering/input rows in both arms.

    The seed shuffles groups/rows, not their frozen label assignment. Sampling a
    COMPLETE group preserves exact arm matching even in a single training batch.
    Treat groups as dependent paired experiments, not3456 independent samples.
    """
    if split not in NAMES or arm not in ARMS or type(seed) is not int:
        raise ValueError('public train/dev split, true/shuffled arm, integer order seed required')
    nuisances = list(_nuisances())
    rng = random.Random(int(_digest(['order', split, seed]), 16))
    rng.shuffle(nuisances)
    for nuisance in nuisances:
        group = _group(split, nuisance, arm)
        rng.shuffle(group['rows'])
        yield group


def rows(split, arm='true', seed=0):
    return [row for group in iter_groups(split, seed, arm) for row in group['rows']]


def evaluation_rows(split='dev'):
    """1152 UNIQUE public dev prompts, true gold only; no final holdout API."""
    if split != 'dev':
        raise ValueError('evaluation convenience is public dev only')
    return [r for r in rows(split, arm='true') if r['replica'] == 0]


def parser_oracle(prompt):
    """Independent public train/dev text parser; no generator latent metadata."""
    name_pattern = r'(Mira|Noel|Luma|Venn)'
    tasks = {}
    assignment_patterns = (name_pattern+r' (pack|sort)(?=[.;])',
        r'(pack|sort) for '+name_pattern, name_pattern+r'=(pack|sort)',
        name_pattern+r' has task (pack|sort)')
    for index, pattern in enumerate(assignment_patterns):
        for m in re.finditer(pattern, prompt):
            name, task = (m[2], m[1]) if index == 1 else (m[1], m[2])
            if name in tasks:
                raise ValueError('duplicate assignment')
            tasks[name] = task
    facts = {}
    chronological_facts = []
    value_pattern = r'(ready now|not ready|no report)'
    fact_patterns = (name_pattern+r': '+value_pattern,
                     value_pattern+r' for '+name_pattern,
                     name_pattern+r' \['+value_pattern+r'\]')
    for index, pattern in enumerate(fact_patterns):
        for m in re.finditer(pattern, prompt):
            name, value = (m[2], m[1]) if index == 1 else (m[1], m[2])
            chronological_facts.append((m.start(), name, value))
    if len(chronological_facts) > 2 and not prompt.startswith((
            'Use the last readiness entry for each name. ',
            'For each person, the newest readiness note replaces the earlier note. ')):
        raise ValueError('repeated facts need the explicit last-write convention')
    for _, name, value in sorted(chronological_facts):
        facts[name] = value
    queries = []
    for pattern in (r'Question: Can '+name_pattern+r' do the assigned task\? Answer: $',
                    r'Question: For the assigned task, can '+name_pattern+r' proceed\? Reply: $',
                    r'For '+name_pattern+r', does its assigned task have support\? Response: $',
                    r'Is the task for '+name_pattern+r' supported by these notes\? Response: $'):
        queries.extend(m[1] for m in re.finditer(pattern, prompt))
    if len(tasks) != 2 or len(facts) != 2 or set(tasks) != set(facts) or sorted(tasks.values()) != ['pack', 'sort'] or len(queries) != 1:
        raise ValueError('complete two-entity bijective task/fact/query frame required')
    name = queries[0]
    if name not in tasks:
        raise ValueError('query outside record')
    status = STATUSES[VALUES.index(facts[name])]
    # Separate declaration from reply() to catch response-source alignment bugs.
    if status == 'supported':
        reference = f'Yes. {name} is ready to {tasks[name]}.'
    elif status == 'contradicted':
        reference = f'No. {name} is not ready to {tasks[name]}.'
    else:
        reference = f'Unknown. The record does not establish that {name} is ready to {tasks[name]}.'
    return {'entity': name, 'action': tasks[name], 'evidence_status': status, 'reference': reference}


def matching_audit(split):
    """Exact public data checks, no tensor/model inference or final data access."""
    groups = [list(iter_groups(split, 0, arm)) for arm in ARMS]
    joint = {arm: Counter() for arm in ARMS}
    for true, shuffled in zip(*groups):
        if true['id'] != shuffled['id'] or len(true['rows']) != 9 or len(shuffled['rows']) != 9:
            raise ValueError('complete matched nuisance groups required')
        for key in ('reference', 'response_targets'):
            if Counter(r[key] for r in true['rows']) != Counter(r[key] for r in shuffled['rows']):
                raise ValueError('answer or length marginal changed')
        for a, b in zip(true['rows'], shuffled['rows']):
            if a['id'] != b['id'] or a['ids'][:a['prompt_len']] != b['ids'][:b['prompt_len']]:
                raise ValueError('arm input/context alignment changed')
            oracle = parser_oracle(a['prompt'])
            if any(a[key] != oracle[key] for key in oracle):
                raise ValueError('true reference disagrees with independent parser')
        for group in (true, shuffled):
            table = Counter((r['evidence_status'], r['label_status']) for r in group['rows'])
            expected = Counter({(e, l): (3 if e == l else 0) for e in STATUSES for l in STATUSES}) if group['arm'] == 'true' else Counter({(e, l): 1 for e in STATUSES for l in STATUSES})
            if table != expected:
                raise ValueError('conditional association/permutation contract failed')
            joint[group['arm']].update(table)
    return {'split': split, 'groups': len(groups[0]), 'rows_per_arm': 9*len(groups[0]),
            'unique_prompts': len({r['prompt'] for g in groups[0] for r in g['rows']}),
            'exact_context_and_answer_histograms_match': True, 'conditional_shuffled_joint_counts_equal': True,
            'joint_counts': {arm: {e+'__'+l: count for (e, l), count in joint[arm].items()} for arm in ARMS}}


def manifest(split):
    selected = rows(split)
    return {**matching_audit(split), 'data_seed': DATA_SEEDS[split], 'vocab_size': VOCAB_SIZE,
            'prompt_byte_lengths': sorted({len(r['prompt_ids']) for r in selected}),
            'response_target_lengths': sorted({r['response_targets'] for r in selected}),
            'true_sha256': _digest(rows(split, 'true')), 'shuffled_sha256': _digest(rows(split, 'shuffled')),
            'model_semantic_inputs': 'raw ordinary UTF8 bytes only; BOS/EOS row framing',
            'final_holdout_available': False}


if __name__ == '__main__':
    # Only the public TRAIN/DEV curriculum exists here.
    print(json.dumps({split: manifest(split) for split in NAMES}, indent=2))
