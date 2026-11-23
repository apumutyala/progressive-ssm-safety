"""Curate an external local corpus manifest before training; makes no network requests."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import corpus_io as io


def read_local(root, relative, cap):
    if not isinstance(relative,str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Require a relative path under the approved root")
    path = (root/relative).resolve(strict=True)
    if not path.is_relative_to(root) or not 0 < path.stat().st_size <= cap:
        raise ValueError("File outside approved root or size bounds")
    with path.open("rb") as stream:
        data = stream.read(cap+1)
    if not 0 < len(data) <= cap:
        raise ValueError("File grew beyond cap")
    return data


def prepare(spec, root, output):
    root = root.resolve(strict=True)
    if output.resolve().is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("Keep the private manifest outside the source checkout")
    with spec.open("rb") as stream:
        raw = stream.read(io.MAX_MANIFEST_BYTES+1)
    if len(raw)>io.MAX_MANIFEST_BYTES:
        raise ValueError("Specification exceeds cap")
    records = json.loads(raw)["documents"]
    if not 3 <= len(records) <= io.MAX_DOCUMENTS:
        raise ValueError("Expected 3–32 explicitly named documents")
    docs, totals = [], dict(train=0,dev=0,test=0)
    keys = ("id","path","split","group","author","source_id","license_id","rights_status",
            "model_fit_allowed","evaluation_allowed")
    for record in records:
        row = {k:record[k] for k in keys}
        blob = read_local(root,row["path"],io.MAX_DOCUMENT_BYTES)
        fp = io.content_fingerprint(blob)
        row.update(bytes=len(blob),sha256=io.sha(blob),canonical_sha256=fp["canonical_sha256"],
                   shingles_sha256=fp["shingles_sha256"],evidence=[])
        totals[row["split"]] += len(blob)
        if sum(totals.values())>io.MAX_CONTENT_BYTES:
            raise ValueError("Aggregate data exceeds approved loader cap")
        for rel in record["evidence_paths"]:
            proof = read_local(root,rel,io.MAX_EVIDENCE_BYTES)
            row["evidence"].append(dict(path=rel,bytes=len(proof),sha256=io.sha(proof)))
        docs.append(row)
    comparisons = sum(1 for i,a in enumerate(docs) for b in docs[:i] if a["split"]!=b["split"])
    manifest = dict(schema="corpus_first_external_v1",documents=docs,content_bytes=totals,
        content_byte_cap=sum(totals.values()),duplicate_audit=dict(method="NFKC_casefold_whitespace_and_word5",
        cross_split_duplicates=0, cross_split_pairs=comparisons))
    buf = (json.dumps(manifest,sort_keys=True,indent=2)+"\n").encode()
    pin = io.sha(buf)
    # During explicit curation only, validate all cross-split fingerprints/rights.
    # An independent training process loads split bytes lazily from this pin.
    with TemporaryDirectory() as tmp:
        draft = Path(tmp)/"manifest.json"
        draft.write_bytes(buf)
        corpus = io.Corpus.load(draft,pin,root)
        for split in ("train","dev","test"):
            corpus.documents(split)
    with output.open("xb") as stream:
        stream.write(buf)
    return pin


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("spec","root","output"):
        p.add_argument("--"+key,required=True,type=Path)
    args=p.parse_args()
    print(prepare(args.spec,args.root,args.output))
