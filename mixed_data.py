# Reader adapted verbatim from reviewed pack reader.py SHA256
# 3c9ff0278cfae3dfba5111111b52eaf1acd71dd8f348fc3d1bfdde2d465581e0.
# No external Python source is imported or executed.
"""Bounded stdlib reader for a pinned train-only uint16 document pack."""
import hashlib
import json
from pathlib import Path
import struct

TOKENIZER_DIGEST='32cadff993c8e0f9810201db3a964944457f86a5009503899b762e19184802fb'
TOKENIZER_SHA='c2fd2d83c6c795cb212938e6d9accbaf3a5b48ed7eb3e21168b6059004c6c376'
MAX_PACK_BYTES=1048576
MAX_TOKEN_BYTES=262144
MAX_DOCUMENTS=64
MAX_CONTENT_BYTES=8192

def sha(raw):return hashlib.sha256(raw).hexdigest()

def pieces_from(raw):
    if sha(raw)!=TOKENIZER_SHA:raise ValueError('exact frozen tokenizer artifact required')
    obj=json.loads(raw)
    if obj.get('tokenizer_digest')!=TOKENIZER_DIGEST or obj.get('vocab_size')!=322 or obj.get('bos')!=257 or obj.get('eos')!=256:raise ValueError('tokenizer schema')
    body={k:v for k,v in obj.items() if k!='tokenizer_digest'}
    if sha(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode())!=TOKENIZER_DIGEST:raise ValueError('tokenizer digest')
    pieces={i:bytes([i]) for i in range(256)}
    if len(obj['merges'])!=64:raise ValueError('frozen merge count')
    for i,pair in enumerate(obj['merges']):
        if type(pair) is not list or len(pair)!=2 or any(type(t) is not int or t not in pieces for t in pair):raise ValueError('merge domain')
        pieces[258+i]=pieces[pair[0]]+pieces[pair[1]]
    return pieces

def validate_payload(index,raw_tokens,tokenizer_raw):
    if type(index) is not dict or index.get('schema')!='broader_prose_pack_v1' or index.get('split')!='public_development_train':raise ValueError('pack schema/split')
    if index.get('document_reset_required') is not True or index.get('no_cross_document_windows') is not True or index.get('tokenizer_fit_performed') is not False:raise ValueError('reset/window/frozen-fit contract')
    if index.get('tokenizer_digest')!=TOKENIZER_DIGEST or index.get('bos')!=257 or index.get('eos')!=256 or index.get('vocab_size')!=322:raise ValueError('pack tokenizer identity')
    if type(raw_tokens) is not bytes or not 0<len(raw_tokens)<=MAX_TOKEN_BYTES or len(raw_tokens)%2:raise ValueError('bounded uint16 payload')
    if index.get('tokens_sha256')!=sha(raw_tokens) or index.get('tokens_bytes')!=len(raw_tokens):raise ValueError('token payload identity')
    tokens=struct.unpack('<'+'H'*(len(raw_tokens)//2),raw_tokens)
    if index.get('token_count')!=len(tokens):raise ValueError('token total')
    pieces=pieces_from(tokenizer_raw);docs=index.get('documents')
    if type(docs) is not list or not 1<=len(docs)<=MAX_DOCUMENTS:raise ValueError('document cap')
    chapters=index.get('chapters')
    if type(chapters) is not list or len(chapters)!=6 or len(set(chapters))!=6:raise ValueError('six source chapters required')
    seen=set();seen_chapters=set();offset=0;content_total=0;ends={}
    for d in docs:
        if type(d) is not dict or type(d.get('id')) is not str or d['id'] in seen:raise ValueError('document identity')
        seen.add(d['id'])
        chapter=d.get('group','')+'/'+d.get('chapter','')
        if chapter not in chapters:raise ValueError('unknown chapter')
        seen_chapters.add(chapter)
        if d.get('split')!='public_development_train' or d.get('eos_synthetic') is not True:raise ValueError('document scope')
        start,end=d.get('normalized_start'),d.get('normalized_end');count=d.get('token_count')
        if type(start) is not int or type(end) is not int or not 0<=start<end or end-start>MAX_CONTENT_BYTES:raise ValueError('normalized byte range')
        if start<ends.get(chapter,0):raise ValueError('overlap/reordered source spans')
        ends[chapter]=end
        if type(d.get('token_offset')) is not int or d['token_offset']!=offset or type(count) is not int or not 3<=count<=MAX_CONTENT_BYTES+2 or offset+count>len(tokens):raise ValueError('contiguous document token partition')
        seq=tokens[offset:offset+count]
        if seq[0]!=257 or seq[-1]!=256 or any(t not in pieces for t in seq[1:-1]):raise ValueError('exact BOS/content/EOS document')
        content=b''.join(pieces[t] for t in seq[1:-1])
        if len(content)!=d.get('content_bytes') or len(content)!=end-start or sha(content)!=d.get('content_sha256'):raise ValueError('decoded content identity')
        content.decode('utf-8',errors='strict')
        content_total+=len(content);offset+=count
    if offset!=len(tokens) or seen_chapters!=set(chapters):raise ValueError('unclaimed tokens or missing chapter')
    if index.get('content_bytes')!=content_total or index.get('synthetic_eos_targets')!=len(docs) or index.get('content_tokens')!=len(tokens)-2*len(docs):raise ValueError('content/EOS accounting')
    return tokens

def load_pack(root,expected_manifest_sha256):
    """Return (index dict, tuple token IDs). Caller must reset at every document."""
    root=Path(root).resolve();path=root/'MANIFEST.json'
    if type(expected_manifest_sha256) is not str or len(expected_manifest_sha256)!=64 or path.stat().st_size>65536:raise ValueError('explicit bounded manifest pin required')
    raw=path.read_bytes()
    if sha(raw)!=expected_manifest_sha256:raise ValueError('manifest identity')
    manifest=json.loads(raw);rows=manifest.get('files')
    if type(rows) is not list or not 1<=len(rows)<=32:raise ValueError('manifest files cap')
    names=set();total=len(raw)
    for row in rows:
        rel=row.get('path');count=row.get('bytes')
        if type(rel) is not str or rel in names or type(count) is not int or not 0<=count<=MAX_PACK_BYTES:raise ValueError('manifest row')
        names.add(rel);p=(root/rel).resolve()
        if not p.is_relative_to(root) or p.stat().st_size!=count:raise ValueError('file boundary/size')
        total+=count
        if total>MAX_PACK_BYTES:raise ValueError('pack cap')
        if sha(p.read_bytes())!=row.get('sha256'):raise ValueError('file hash')
    if not {'tokens.u16le','index.json','tokenizer.json'}<=names:raise ValueError('required payload files')
    index=json.loads((root/'index.json').read_bytes())
    tokens=validate_payload(index,(root/'tokens.u16le').read_bytes(),(root/'tokenizer.json').read_bytes())
    return index,tokens


# Mixed-domain integration below. Interval protocol IDs remain unchanged.
import os
import re
import interval_data as _interval

PAD, UNK, BOS, EOS = _interval.PAD, _interval.UNK, _interval.BOS, _interval.EOS
CONFIRM, DEFER = _interval.CONFIRM, _interval.DEFER
QUERY_A, QUERY_B = _interval.QUERY_A, _interval.QUERY_B
HOLD, CLEAR = _interval.HOLD, _interval.CLEAR
PACK_MANIFEST_SHA256 = 'f7813166e0e7e8edcdc2485dce632a44fc795710c7d792e93811b6f0d1f35bd8'
BYTE_OFFSET = len(_interval.VOCAB)
WINDOW_CONTENT_TOKENS = 128
PROMPT_CONTENT_TOKENS = 32
MAX_WINDOWS = 128
_MERGES = ((101,32),(100,32),(116,104),(32,97),(116,32),(105,110),(115,32),(226,128),
 (101,114),(260,258),(110,259),(44,32),(111,117),(261,268),(121,32),(110,32),
 (263,103),(116,111),(101,259),(104,105),(101,97),(108,108),(111,114),(119,97),
 (46,32),(102,32),(111,110),(104,258),(104,97),(274,32),(101,110),(275,32),
 (44,271),(111,119),(115,116),(111,283),(266,32),(265,156),(265,157),(261,32),
 (97,116),(10,10),(277,264),(105,116),(97,262),(265,153),(101,115),(114,105),
 (101,273),(279,32),(97,114),(111,108),(97,110),(111,109),(115,101),(119,105),
 (103,104),(99,104),(258,97),(32,267),(102,280),(111,273),(111,32),(121,270))
_PIECES = {i: bytes([i]) for i in range(256)}
for _i, _pair in enumerate(_MERGES):
    _PIECES[258 + _i] = _PIECES[_pair[0]] + _PIECES[_pair[1]]
# 256/257 are reserved source framing slots, not content; retain full322 mapping.
VOCAB = list(_interval.VOCAB) + [f'<byte_bpe_{i}>' for i in range(322)]


def encode_bytes(raw):
    """Exact frozen BPE; accepts arbitrary bytes, bounded to128KiB."""
    if type(raw) is not bytes or len(raw) > 131072:
        raise ValueError('bounded bytes required')
    ids = list(raw)
    for i, pair in enumerate(_MERGES):
        new = []
        j = 0
        while j < len(ids):
            if j + 1 < len(ids) and (ids[j], ids[j+1]) == pair:
                new.append(258 + i)
                j += 2
            else:
                new.append(ids[j])
                j += 1
        ids = new
    return [BYTE_OFFSET + i for i in ids]


def encode(text):
    """Ordinary text only: never interprets protocol spellings as special IDs."""
    if type(text) is not str:
        raise TypeError('str required')
    return encode_bytes(text.encode('utf-8'))


def decode_bytes(ids):
    """Strict exact byte-domain decode; reject word/protocol/framing IDs."""
    result = []
    for token in ids:
        token = int(token)
        if token - BYTE_OFFSET not in _PIECES:
            raise ValueError('not a byte content ID')
        result.append(_PIECES[token - BYTE_OFFSET])
    return b''.join(result)


def decode(ids):
    """Exact for pure byte content; readable rendering for mixed generations.

    IDs are the authoritative generation record. Invalid generated UTF8 is
    rendered with replacement characters, never treated as a roundtrip success.
    """
    output = []
    byte_run = []
    word_run = []

    def flush_bytes():
        if byte_run:
            output.append(b''.join(byte_run).decode('utf-8', errors='replace'))
            byte_run.clear()

    def flush_words():
        if word_run:
            output.append(_interval.decode(word_run))
            word_run.clear()

    for token in ids:
        token = int(token)
        if token in (PAD, BOS, EOS):
            continue
        if token - BYTE_OFFSET in _PIECES:
            flush_words()
            byte_run.append(_PIECES[token - BYTE_OFFSET])
        else:
            flush_bytes()
            if 0 <= token < BYTE_OFFSET:
                word_run.append(token)
            else:
                flush_words()
                output.append(f'<invalid_id_{token}>')
    flush_bytes()
    flush_words()
    return ''.join(output)


def _holds(count):
    return [[[HOLD] * 3 for _ in range(2)] for _ in range(count)]


def _prose_row(identifier, content, prompt_end, **metadata):
    raw = decode_bytes(content)
    prompt_raw, response_raw = decode_bytes(content[:prompt_end]), decode_bytes(content[prompt_end:])
    # Strict UTF8 splits: scalar boundaries must be intact at window and prompt.
    text = raw.decode('utf-8', errors='strict')
    prompt = prompt_raw.decode('utf-8', errors='strict')
    reference = response_raw.decode('utf-8', errors='strict')
    ids = [BOS] + list(content) + [EOS]
    reserved = {PAD, UNK, BOS, EOS, CONFIRM, DEFER, QUERY_A, QUERY_B}
    assert not reserved.intersection(content)
    assert all(BYTE_OFFSET <= t < len(VOCAB) for t in content)
    return {'id': identifier, 'kind': 'prose', 'prompt': prompt, 'reference': reference,
            'source_text': text, 'ids': ids, 'prompt_len': 1 + prompt_end,
            'events': _holds(len(ids)), 'target_utf8_bytes': len(raw),
            'response_utf8_bytes': len(response_raw), 'codec': 'frozen_byte_bpe_v1',
            'content_targets': len(content), 'synthetic_eos_targets': 1, **metadata}


def _safe_boundary(content, end, minimum=1):
    while end >= minimum:
        try:
            decode_bytes(content[:end]).decode('utf-8', errors='strict')
            return end
        except UnicodeDecodeError:
            end -= 1
    raise ValueError('cannot make a nonempty UTF8 window boundary')


def _training_prose(target_budget):
    root = os.environ.get('IMPERIUM_PROSE_PACK')
    if not root:
        raise RuntimeError('mixed training requires explicit IMPERIUM_PROSE_PACK pointing to the pinned local train-only pack')
    index, tokens = load_pack(root, PACK_MANIFEST_SHA256)
    tokenizer_raw = (Path(root).resolve() / 'tokenizer.json').read_bytes()
    if pieces_from(tokenizer_raw) != _PIECES:
        raise ValueError('embedded byte codec differs from pinned pack')
    docs = []
    for doc in index['documents']:
        start, count = doc['token_offset'], doc['token_count']
        docs.append([BYTE_OFFSET + t for t in tokens[start+1:start+count-1]])
    offsets = [0] * len(docs)
    rows = []
    count = 0
    # Round-robin exposure covers every source document before taking a second
    # window. Windows are disjoint, reset state, and never cross a document.
    while count < target_budget:
        progress = False
        for doc_i, content in enumerate(docs):
            start = offsets[doc_i]
            remaining = content[start:]
            if len(remaining) < 2:
                continue
            length = min(WINDOW_CONTENT_TOKENS, len(remaining), max(2, target_budget-count))
            end = _safe_boundary(remaining, length)
            if end < 2:
                continue
            window = remaining[:end]
            prompt_end = _safe_boundary(window, min(PROMPT_CONTENT_TOKENS, max(1, end//2)))
            if prompt_end >= len(window):
                raise ValueError('prose continuation must be nonempty')
            row = _prose_row(f'train-pack-{doc_i:02d}-{start:05d}', window, prompt_end,
                             source_domain='private_local_approved_train_only',
                             pack_manifest_sha256=PACK_MANIFEST_SHA256,
                             pack_document_ordinal=doc_i, source_token_start=start,
                             source_token_end=start+end)
            rows.append(row)
            count += len(window)
            offsets[doc_i] = start+end
            progress = True
            if len(rows) > MAX_WINDOWS:
                raise ValueError('bounded window count exceeded')
            if count >= target_budget:
                break
        if not progress:
            raise ValueError('pack exhausted before requested bounded exposure')
    return rows


def examples(split):
    original = _interval.examples(split)
    decisions = [r for r in original if r['kind'] == 'decision']
    if split == 'train':
        # Full next-token decision targets incl EOS; prose budget counts content
        # only. The tiny extra synthetic EOS cost is separately reported.
        budget = sum(len(r['ids']) - 1 for r in decisions)
        prose = _training_prose(budget)
    else:
        prose = []
        for r in original:
            if r['kind'] != 'prose':
                continue
            # Encode the prompt and continuation independently: byte accuracy
            # and their exact boundary matter more than one cross-boundary merge.
            pids, rids = encode(r['prompt'] + ' '), encode(r['reference'])
            prose.append(_prose_row(r['id'] + '-byte', pids+rids, len(pids),
                                     source_domain='self_authored_heldout_CC0'))
    # Deterministic alternating order makes uniform record sampling close to a
    # 1:1 token mixture without task/outcome-dependent sampling.
    result = []
    for i in range(max(len(decisions), len(prose))):
        if i < len(decisions):
            result.append(decisions[i])
        if i < len(prose):
            result.append(prose[i])
    return result


def corpus_manifest():
    receipt = {'schema': 1, 'task': 'mixed', 'pack_manifest_sha256': PACK_MANIFEST_SHA256,
               'tokenizer_artifact_sha256': TOKENIZER_SHA, 'tokenizer_digest': TOKENIZER_DIGEST,
               'vocab_size': len(VOCAB), 'interval_vocab_size': BYTE_OFFSET,
               'byte_codec_slots': 322, 'pack_content_tokens_available': 68439,
               'window_content_token_cap': WINDOW_CONTENT_TOKENS,
               'vocab_sha256': sha(json.dumps(VOCAB).encode()), 'splits': {}}
    for split in ('train', 'dev', 'test'):
        rows = examples(split)
        decisions = [r for r in rows if r['kind'] == 'decision']
        prose = [r for r in rows if r['kind'] == 'prose']
        receipt['splits'][split] = {'examples': len(rows), 'decision': len(decisions),
            'supported': sum(r.get('supported', False) for r in decisions),
            'prose_windows': len(prose), 'decision_targets': sum(len(r['ids'])-1 for r in decisions),
            'prose_content_targets': sum(r['content_targets'] for r in prose),
            'prose_utf8_bytes': sum(r['target_utf8_bytes'] for r in prose),
            'synthetic_prose_eos_targets': len(prose),
            'max_tokens': max(len(r['ids']) for r in rows),
            'sha256': sha(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode())}
        if split == 'train':
            receipt['splits'][split]['source_spans'] = [{
                'document_ordinal': r['pack_document_ordinal'],
                'token_start': r['source_token_start'], 'token_end': r['source_token_end']}
                for r in prose]
    return receipt


if __name__ == '__main__':
    # Metadata only. Never print private local pack text or token arrays.
    print(json.dumps(corpus_manifest(), indent=2))
