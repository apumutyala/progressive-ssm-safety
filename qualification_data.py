"""CC0 synthetic TRAIN and explicitly public development qualification catalogs.

All prose/fictional records authored for this fixture are dedicated to CC0-1.0.
Development shares the grammar: it is a diagnostic, never a sealed holdout.
Only ids, loss_mask and prompt_len belong in a training adapter; oracle labels,
pair metadata and source provenance are bookkeeping, never auxiliary inputs.
"""
from collections import Counter
import hashlib
import json
import re

PAD, UNK, BOS, EOS, CONFIRM, DEFER, QUERY_A, QUERY_B = range(8)
BYTE_OFFSET, VOCAB_SIZE = 8, 264
STATUSES = ('affirmative', 'negative', 'abstain')
FAMILIES = ('name_only', 'source_only', 'epoch_only', 'relevant_edit',
            'nuisance_edit', 'consistent_update', 'correction')
LICENSE = 'CC0-1.0'
CATALOG_SHA256 = {'train':'9da5d35660f2c1baa0870b23c6477258a87e01ed0a3b9eef6f88350a19f2bc36',
                  'development':'bb8992ba71b20c3cf479684b4e5374d93c707ba64c44d3e5f58a4bccf65feaf6'}
OVERFIT_SHA256 = 'b0d6b8b34f550d8cebedac83ae7d24e5b1525fd42d613bd2f258f6a8f402f891'


def _require(ok, message):
    if not ok: raise ValueError(message)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _sha(value): return hashlib.sha256(value).hexdigest()
def encode(text): return tuple(BYTE_OFFSET+b for b in text.encode('utf-8'))
def decode(ids):
    _require(all(type(i) is int and 8 <= i < 264 for i in ids), 'Not an ordinary byte sequence.')
    return bytes(i-8 for i in ids).decode('utf-8')


def canonical_response(status, entity, verb):
    _require(status in STATUSES, 'Unknown readiness status.')
    if status == 'affirmative': return f'Yes. {entity} is ready to {verb}.'
    if status == 'negative': return f'No. {entity} is not ready to {verb}.'
    return f'Unknown. The record does not establish that {entity} is ready to {verb}.'


def oracle(prompt):
    """Independent text parser: exact source/epoch/name, last matching entry wins.

    The fictional records are stipulated truthful within their named scope.
    An explicit ``no report`` clears an earlier entry; absence is also unknown.
    This is a finite fixture semantics, not verification of external truth.
    """
    _require(isinstance(prompt, str) and len(prompt.encode()) <= 2048, 'Prompt outside parser bound.')
    assignment = re.findall(r'^([A-Z][a-z]+) is assigned to (pack|sort)\.$', prompt, re.M)
    _require(len(assignment) == 2 and len(dict(assignment)) == 2, 'Expected two unique assignments.')
    requests = re.findall(r'^At epoch ([12]), using source (Cedar|Birch), is ([A-Z][a-z]+) ready for the assigned task\?$', prompt, re.M)
    _require(len(requests) == 1 and prompt.endswith('\nAnswer:'), 'Expected one complete request.')
    epoch, source, entity = requests[0]
    _require(entity in dict(assignment), 'Unassigned query entity.')
    entries = re.findall(r'^(Cedar|Birch) epoch ([12]): ([A-Z][a-z]+) (ready|not ready|no report)\.$', prompt, re.M)
    lines=prompt.splitlines()
    _require(lines[0]=='Use the named source and epoch. Last matching entry wins.' and
             lines.count('Answer:')==1 and len(lines)==len(assignment)+len(entries)+3 and
             re.fullmatch(r'At epoch [12], using source (Cedar|Birch), is [A-Z][a-z]+ ready for the assigned task\?',lines[-2]),
             'Unrecognized or misplaced fixture lines.')
    _require(all(name in dict(assignment) for _, _, name, _ in entries), 'Record names unassigned entity.')
    state = 'no report'; matches = 0
    for src, ep, name, value in entries:
        if (src, ep, name) == (source, epoch, entity): state = value; matches += 1
    status = {'ready':'affirmative', 'not ready':'negative', 'no report':'abstain'}[state]
    verb = dict(assignment)[entity]
    return {'status': status, 'entity': entity, 'verb': verb, 'source': source, 'epoch': int(epoch),
            'matching_entries': matches, 'reference': canonical_response(status, entity, verb)}


def _prompt(assignments, entries, query):
    lines = ['Use the named source and epoch. Last matching entry wins.']
    lines += [f'{name} is assigned to {verb}.' for name, verb in assignments]
    wording = dict(zip(STATUSES, ('ready', 'not ready', 'no report')))
    lines += [f'{src} epoch {ep}: {name} {wording[state]}.' for src, ep, name, state in entries]
    name, src, ep = query
    lines += [f'At epoch {ep}, using source {src}, is {name} ready for the assigned task?', 'Answer:']
    return '\n'.join(lines)


def _record(record_id, prompt, metadata, *, kind='synthetic', reference=None):
    result = oracle(prompt) if kind == 'synthetic' else None
    if reference is None: reference = result['reference']
    prefix = (BOS,)+encode(prompt); answer = encode(reference)+(EOS,)
    ids = prefix+answer
    row = {'id': record_id, 'kind': kind, 'prompt': prompt, 'reference': reference,
           'ids': ids, 'prompt_len': len(prefix),
           'loss_mask': (False,)*(len(prefix)-1)+(True,)*len(answer), 'metadata': metadata}
    if result: row.update({k: result[k] for k in ('status', 'entity', 'verb', 'source', 'epoch')})
    if result:
        row['contrast']=canonical_response(STATUSES[(STATUSES.index(result['status'])+1)%3],result['entity'],result['verb'])
    return row


def _catalog(split):
    _require(split in ('train', 'development'), 'Only TRAIN and public development exist; no holdout API.')
    names = ('Iris', 'Owen') if split == 'train' else ('Ada', 'Bea')
    worlds = 8 if split == 'train' else 4
    output = []
    for world in range(worlds):
        queried = names[world % 2]; other = names[1-world % 2]
        assignments = tuple(zip(names, ('pack', 'sort') if world//2 % 2 == 0 else ('sort', 'pack')))
        if world % 2: assignments = assignments[::-1]
        source = ('Cedar', 'Birch')[world//4 % 2]; alternate = 'Birch' if source == 'Cedar' else 'Cedar'
        epoch = 1+world//2 % 2; other_epoch = 3-epoch
        for index, status in enumerate(STATUSES):
            next_status, last_status = STATUSES[(index+1)%3], STATUSES[(index+2)%3]
            entries = [(src, ep, name, status if (src, ep, name)==(source, epoch, queried) else next_status)
                       for src in ('Cedar','Birch') for ep in (1,2) for name in names]
            if world % 2: entries.reverse()
            query = (queried, source, epoch)
            base = _prompt(assignments, entries, query)
            for family in FAMILIES:
                changed = list(entries); changed_query = query
                if family == 'name_only': changed_query = (other, source, epoch)
                elif family == 'source_only': changed_query = (queried, alternate, epoch)
                elif family == 'epoch_only': changed_query = (queried, source, other_epoch)
                elif family == 'relevant_edit':
                    changed = [(s,e,n,next_status if (s,e,n)==(source,epoch,queried) else v) for s,e,n,v in entries]
                elif family == 'nuisance_edit':
                    changed = [(s,e,n,last_status if (s,e,n)==(source,epoch,other) else v) for s,e,n,v in entries]
                elif family == 'consistent_update': changed.append((source,epoch,queried,status))
                elif family == 'correction': changed.append((source,epoch,queried,next_status))
                variant = _prompt(assignments, changed, changed_query)
                pair = f'{split}-w{world:02d}-{status}-{family}'
                for side, prompt in enumerate((base, variant)):
                    metadata = {'license':LICENSE, 'public':True, 'world':world, 'family':family,
                                'pair':pair, 'side':side, 'base_status':status,
                                'expected_status_change':family not in ('nuisance_edit','consistent_update'),
                                'query_only_edit':family in ('name_only','source_only','epoch_only')}
                    row = _record(f'{pair}-{side}', prompt, metadata)
                    row.update({k:metadata[k] for k in ('world','family','pair','side')})
                    output.append(row)
    return tuple(output)


def records(split='train', seed=12):
    """Deterministic ordering only; seed never changes contents or truth labels."""
    _require(type(seed) is int and 0 <= seed < 2**32, 'Invalid ordering seed.')
    result=tuple(sorted(_catalog(split), key=lambda r: _sha(_wire([seed,r['id']]))))
    for row in result: validate_record(row)
    return result


def iter_groups(split='train', seed=12):
    rows = records(split, seed); groups = {}
    for row in rows: groups.setdefault(row['pair'], []).append(row)
    for pair, members in groups.items():
        yield {'id':pair, 'family':members[0]['family'], 'rows':tuple(sorted(members,key=lambda r:r['side']))}


def minimal_training_records(seed=12):
    """18 balanced rows for an explicitly labeled overfit test, not qualification."""
    return tuple(r for r in records('train',seed) if r['world']==0 and
                 r['family'] in ('name_only','relevant_edit','nuisance_edit'))


def overfit_group(seed=12): return minimal_training_records(seed)


def overfit_manifest():
    rows=overfit_group()
    _require(_sha(_wire(rows))==OVERFIT_SHA256,'Frozen overfit catalog changed.')
    return {'schema':1,'rows':len(rows),'sha256':_sha(_wire(rows)),
            'status_counts':dict(Counter(r['status'] for r in rows)),
            'answer_byte_targets':sum(len(r['ids'])-r['prompt_len']-1 for r in rows),
            'answer_eos_targets':len(rows),'max_input_tokens':max(len(r['ids'])-1 for r in rows),
            'scope':'TRAIN-only balanced overfit check; not development or qualification evidence.'}


ORIGINAL_PROSE = (
    'Morning light reached the workshop. Iris placed a wooden box beside the door. She checked the list and waited for Owen. A paper tag hung from the handle.',
    'Owen sorted the blue cards before lunch. The red cards stayed on the table until the next shift began.',
    'A written record can be incomplete. A missing entry gives no answer to a question about what happened.',
    'The gardeners carried water along the path. After the rain stopped, they put the empty cans in the shed.',
    'Two shelves held different books. Iris read the label on each shelf before putting the small volume away.',
    'The clock struck six. Owen closed the window, folded his coat, and walked home through the quiet square.',
    'A correction replaces an earlier statement about the same event. A report about another event leaves that statement alone.',
    'The team finished a short note. It named the people, described the work, and ended with a complete sentence.',
)


def ordinary_records():
    """Complete original CC0 prose units; EOS means authored unit end, not book end."""
    result = []
    for i, text in enumerate(ORIGINAL_PROSE):
        ids = (BOS,)+encode(text)+(EOS,)
        result.append({'id':f'cc0-prose-{i}', 'kind':'ordinary', 'ids':ids, 'prompt_len':1,
                       'loss_mask':(True,)*(len(ids)-1), 'prompt':'', 'reference':text,
                       'metadata':{'license':LICENSE,'public':True,'source':'original_fixture',
                                   'end_unit':'complete_authored_prose','sha256':_sha(text.encode())}})
    return tuple(result)


def corpus_records(corpus, step, *, seed=12, batch_size=8, seq_len=128):
    """Explicit approved real-source option; TRAIN windows only, never fabricated EOS.

    Caller must enroll the pinned Corpus.load result separately. This function
    reads only its stateless TRAIN sampler; sampled chapter/window EOS is removed
    because this API carries no complete-source boundary attestation.
    """
    from corpus_io import Corpus
    _require(isinstance(corpus, Corpus), 'Require admitted Corpus instance.')
    _require(type(step) is int and step>=0 and 1<=batch_size<=32 and 1<=seq_len<=256, 'Sampler bounds.')
    result=[]
    for i, row in enumerate(corpus.sample_batch(step,seed,batch_size,seq_len)):
        ids=tuple(row['ids']); ids=ids[:-1] if ids[-1]==EOS else ids
        _require(ids[0]==BOS and all(8<=x<264 for x in ids[1:]), 'Unexpected real-source tokens.')
        result.append({'id':f'approved-train-{step}-{i}', 'kind':'ordinary', 'ids':ids,
                       'prompt_len':1, 'loss_mask':(True,)*(len(ids)-1),
                       'metadata':{'source':'approved_train_only','window_sha256':_sha(_wire(ids)),
                                   'source_eos_targets':0}})
    return tuple(result)


def model_record(row):
    """Only language IDs and objective bookkeeping; no oracle/metadata/events."""
    return {key:row[key] for key in ('ids','loss_mask','prompt_len')}


def validate_record(row):
    """Validate next-target alignment without preparing tensors or reading labels as input."""
    ids=row['ids']; mask=row['loss_mask']; boundary=row['prompt_len']
    _require(isinstance(ids,tuple) and len(ids)>=2 and ids[0]==BOS and
             all(type(t) is int and 8<=t<264 for t in ids[1:-1]) and
             (ids[-1]==EOS or 8<=ids[-1]<264), 'Invalid language IDs.')
    _require(isinstance(mask,tuple) and len(mask)==len(ids)-1 and all(type(m) is bool for m in mask), 'Invalid shifted mask.')
    if row['kind']=='synthetic':
        _require(1<=boundary<len(ids) and ids[-1]==EOS and
                 mask==(False,)*(boundary-1)+(True,)*(len(ids)-boundary), 'Prompt loss or shifted answer boundary.')
        _require(decode(ids[1:boundary])==row['prompt'] and
                 decode(ids[boundary:-1])==row['reference'] and
                 oracle(row['prompt'])['reference']==row['reference'], 'Text/oracle/target mismatch.')
    else:
        _require(row['kind']=='ordinary' and boundary==1 and all(mask), 'Ordinary target dropped.')
        if ids[-1]==EOS:
            _require(row['metadata'].get('end_unit')=='complete_authored_prose', 'Unverified source EOS.')
    return {'byte_targets':sum(use and target>=8 for target,use in zip(ids[1:],mask)),
            'eos_targets':sum(use and target==EOS for target,use in zip(ids[1:],mask)),
            'masked_prompt_targets':sum(not m for m in mask)}


def catalog_manifest(split='train'):
    rows=records(split)
    _require(_sha(_wire(rows))==CATALOG_SHA256[split],'Frozen catalog changed.')
    return {'schema':1,'split':split,'public':True,'license':LICENSE,
            'rows':len(rows),'pairs':len(rows)//2,'sha256':_sha(_wire(rows)),
            'status_counts':dict(Counter(r['status'] for r in rows)),
            'family_counts':dict(Counter(r['family'] for r in rows)),
            'unique_prompts':len({r['prompt'] for r in rows}),
            'answer_byte_targets':sum(len(r['ids'])-r['prompt_len']-1 for r in rows),
            'answer_eos_targets':len(rows),'prompt_supervised_targets':0,
            'max_input_tokens':max(len(r['ids'])-1 for r in rows),
            'scope':'Public grammar-sharing diagnostic; no sealed holdout or old model results.'}
