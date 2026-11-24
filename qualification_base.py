"""Corpus-only shifted next-token objective shared by all three architectures.

No training is dispatched here. The supplied corpus must have passed the existing
rights and document split checks. Training examples carry no trusted events.
"""
import hashlib
import struct


def base_plan(corpus, step, *, seed=12, batch_size=8, seq_len=128):
    from pilot_data import MANIFEST_PIN
    from qualification_data import corpus_records
    if corpus.manifest_sha256 != MANIFEST_PIN:
        raise ValueError('This local qualification supports the previously approved corpus manifest only')
    rows = corpus_records(corpus,step,seed=seed,batch_size=batch_size,seq_len=seq_len)
    return prepare_base_rows(rows)


def prepare_base_rows(rows):
    from qualification_data import BOS, EOS
    digest = hashlib.sha256()
    source_targets = 0
    sequences = []
    for row in rows:
        ids, mask = tuple(row['ids']), tuple(row['loss_mask'])
        if (row['kind']!='ordinary' or row['prompt_len']!=1 or not 2 <= len(ids) <= 1025
                or ids[0]!=BOS or len(mask)!=len(ids)-1 or not all(m is True for m in mask)
                or not all(type(t) is int and 8<=t<=263 for t in ids[1:-1])
                or not (ids[-1]==EOS or 8<=ids[-1]<=263)):
            raise ValueError('Expected fully supervised ordinary byte records with optional genuine ending')
        digest.update(struct.pack('<I',len(ids)) + struct.pack('<'+'H'*len(ids),*ids))
        sequences.append(ids)
        source_targets += len(ids)-1
    if not sequences:
        raise ValueError('No training records supplied')
    return sequences, source_targets, digest.hexdigest()


def objective(model, plan, device):
    """One globally token-normalized base loss. BOS resets state per document row."""
    import torch
    import torch.nn.functional as F
    sequences, expected, _ = plan
    length = max(len(ids)-1 for ids in sequences)
    x = torch.zeros((len(sequences),length),device=device,dtype=torch.long)
    y = torch.full_like(x,-100)
    for i,ids in enumerate(sequences):
        n = len(ids)-1
        x[i,:n] = torch.tensor(ids[:-1],device=device)
        y[i,:n] = torch.tensor(ids[1:],device=device)
    if int(y.ne(-100).sum())!=expected:
        raise ValueError('Token accounting mismatch')
    scores = model(x)
    return F.cross_entropy(scores.flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum')/expected
