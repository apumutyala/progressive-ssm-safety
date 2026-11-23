# Mixed ordinary-language and certified-decision pilot

`mixed_data.py` is an optional dataset for `--task mixed` using the same
`IntervalModel` and primitive-fact decision examples. It adds normal-language
next-token training from an existing, approved, immutable local train-only pack.
This is an exploratory integration arm; its different data and output alphabet
must not be presented as a matched result from the original interval experiment.

## Local-only corpus and exact validation

Set `IMPERIUM_PROSE_PACK` explicitly to the existing `broader_prose_pack_v0_1`
directory before mixed training. There is no automatic path discovery, network
access, download, tokenizer fitting or fallback corpus. Missing/wrong data fails
closed. The source repository contains no corpus, token payload, tokenizer
artifact, checkpoints or raw private local training text.

Required pack manifest SHA256:

`f7813166e0e7e8edcdc2485dce632a44fc795710c7d792e93811b6f0d1f35bd8`

Required tokenizer artifact SHA256:

`c2fd2d83c6c795cb212938e6d9accbaf3a5b48ed7eb3e21168b6059004c6c376`

Frozen tokenizer semantic digest:

`32cadff993c8e0f9810201db3a964944457f86a5009503899b762e19184802fb`

The bounded reader is copied into `mixed_data.py` from reviewed pack `reader.py`
(SHA256 `3c9ff0278cfae3dfba5111111b52eaf1acd71dd8f348fc3d1bfdde2d465581e0`).
It never imports external Python code. It verifies the exact manifest; all
manifest file hashes, sizes and path boundaries; total pack size below 1 MiB;
bounded uint16 payload; frozen tokenizer bytes/schema/digest; 64 merge
definitions; vocabulary, BOS/EOS and split identity; contiguous document token
partitions; non-overlapping indexed source spans; exact decoded UTF8 byte counts
and hashes; and the declared six source chapters. The embedded byte-piece map
must equal the pinned tokenizer's map. The pack declares
`public_development_train`; its 18 document fragments contain 68,439 content
tokens and 110,618 bytes. This loader neither searches nor opens holdout data.
Corpus ownership/provenance approvals remain those of that exact local pack;
the synthetic examples and integration code do not relicense the source prose.

## Two token domains, one trained output distribution

The original 261 interval IDs are unchanged: global framing and decision/query
sentinels remain IDs 0 through 7. A second domain appends all 322 slots of the
frozen byte BPE at offset 261, giving a 583-logit distribution. It contains all
256 byte fallbacks and 64 merge pieces. Source byte-codec framing slots 256/257
are retained as reserved unused output slots; all actual framing uses the
existing global BOS=2 and EOS=3. No new safety token or certificate is introduced.

`encode(text)` always encodes general text as UTF8 bytes followed by frozen BPE
merges. It never interprets text such as `<confirm>` or `<query_a>` as protocol
IDs. `encode_bytes`/`decode_bytes` additionally support arbitrary bytes exactly.
The protocol task builder supplies the original interval token IDs and trusted
fact events. General text alone cannot forge protocol sentinels or trusted fact
events. All prose events are HOLD=-2 and prose input contains no decision/query
IDs. The same words can have distinct protocol-word and byte representations;
shared recurrent weights do not guarantee transfer between those embeddings.

The learned softmax sees all 583 logits for every next-token target, including
the extra alphabet's unused and out-of-domain possibilities. This increases
output competition and changes loss/parameter counts relative to the 261-token
arm. Never compare per-token perplexity across the two alphabets as a language
quality claim. Report ordinary-language lexical NLL in bits per exact UTF8 byte,
with EOS cost separately, plus raw generation and protocol task metrics. New
word/byte mixing, invalid UTF8 and reserved-slot outputs remain possible task
failures. `decode` renders mixed generations readably and replaces invalid UTF8
for display; saved token IDs are authoritative. `decode_bytes` is strict and is
used for exact training/evaluation accounting.

## Deterministic exposure and reset rules

Training retains all 64 primitive-fact decisions (32 feasible/32 infeasible) and
replaces the original tiny authored training prose with bounded pack windows.
It sets the ordinary prose content-token budget equal to the complete decision
next-token target count, including decision EOS. With the pinned inputs this is
7,304 targets per domain. The 58 selected prose windows contain 11,849 exact UTF8
bytes, about 10.67% of the available pack content tokens. Selection round-robins
through the immutable indexed documents, taking at most 128 content tokens from
each before returning for another window; all 18 documents are represented.
Selection uses no development/test outcomes, losses, scores or model state.

Windows never cross document boundaries, never overlap, and reset recurrent
state at every window. Start/end and prompt boundaries preserve UTF8 scalars.
Each window uses up to 32 content tokens as a generation prefix and the rest as
continuation. Full ordinary-language next-token loss still covers prefix and
continuation. Each window has global BOS and one synthetic fragment EOS; these
58 EOS targets are separately counted and are not natural book-ending evidence.
Document ordinal and source token spans are included in local record metadata
and in the corpus receipt. Neither continuation state nor learned activation
crosses a window boundary. Deterministically alternating decision/prose records
produces a 122-record training set with an approximately 1:1 target mixture;
finite random batches can differ. Report actual cumulative target exposure by
kind from the experiment runner rather than assuming the nominal mixture.

Development and test each retain 64 frozen interval decisions and four distinct
self-authored CC0 prose continuations from the original dataset. The eight
held-out prose passages never appear in training. They use the same byte codec,
not word IDs. Development has 360 prose content tokens/553 bytes; test has 360
tokens/570 bytes. Their exact strings are already visible in the public
synthetic source. This is a very small ordinary-language evaluation sample,
without representative corpus coverage or a statistical fluency claim. It
cannot establish novel-domain generalization or justify a scaling trajectory.

The maximum complete sequence is 131 tokens. `ids` includes global BOS/EOS;
`prompt_len` includes BOS. For prose, `source_text` is the exact modeled prompt
plus reference, `target_utf8_bytes` is its UTF8 length and
`response_utf8_bytes` is the exact continuation's UTF8 length. These exclude
framing. Roundtrip tests require `decode(body_ids)==source_text` and
`decode(response_ids)==reference`. The evaluation prefix and continuation are
encoded independently to retain their precise boundary; a possible BPE merge
across that boundary is deliberately not applied.

## Receipts and reproduction

`corpus_manifest()` validates local data and returns the pinned source/tokenizer
identities, vocabulary hash, per-split full-record hashes, counts, byte totals,
synthetic EOS counts and selected source spans. Its CLI prints only this receipt;
it never prints private local training strings or token arrays. Full examples
are available in memory for local training. Keep any serialized training data,
local dataset receipts containing text, checkpoints and corpus excerpts outside
the source repository. Publish only source, public synthetic examples and
reviewed aggregate results unless further corpus release is explicitly allowed.

PowerShell example (dataset validation only; no Torch or GPU):

```powershell
$env:IMPERIUM_PROSE_PACK = 'D:\research-data\broader_prose_pack_v0_1'
python -B mixed_data.py
```

Verified stdlib-only properties: exact pinned pack acceptance; token/event
alignment; 66 prose row roundtrips and exact byte denominators; all 256 byte
fallbacks; arbitrary Unicode roundtrip; literal protocol spelling isolation;
all-HOLD prose events; and preservation of the original decision IDs. These are
data/codec checks, not numerical-model certificate validation or evidence of
language quality. The optimizer path, finite-precision certificate, useful
decision behavior and actual generation quality require the separate experiment
and independent review.
