"""Compare actual optimizer, parameter and sampler contents of two private endpoints."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import torch
from pretrain import compare_checkpoint_states, LOAD_CAP


def checked_read(folder, name, cap):
    with (folder/name).open("rb") as stream:
        blob = stream.read(cap+1)
    if not 0<len(blob)<=cap:
        raise ValueError("Artifact exceeds expected size")
    return blob


def load(folder):
    proof = json.loads(checked_read(folder,"manifest.json",65536))["checkpoint.pt"]
    buf = checked_read(folder,"checkpoint.pt",LOAD_CAP)
    if len(buf)!=proof["bytes"] or hashlib.sha256(buf).hexdigest()!=proof["sha256"]:
        raise ValueError("Checkpoint manifest mismatch")
    result = torch.load(io.BytesIO(buf),map_location="cpu",weights_only=True)
    if result["step"]!=128 or result["arm"]!="certified":
        raise ValueError("Expected the two certified fixed endpoints")
    return result, proof["sha256"]


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resumed",type=Path,required=True)
    p.add_argument("--uninterrupted",type=Path,required=True)
    args=p.parse_args()
    resumed, resumed_pin = load(args.resumed)
    uninterrupted, uninterrupted_pin = load(args.uninterrupted)
    answer = compare_checkpoint_states(resumed, uninterrupted)
    answer.update(resumed_checkpoint_sha256=resumed_pin,
                  uninterrupted_checkpoint_sha256=uninterrupted_pin)
    print(json.dumps(answer,sort_keys=True))
    if not answer["exact_equal"]:
        raise SystemExit(1)
