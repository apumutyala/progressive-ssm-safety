"""Export synthetic evaluation rows and a small source-shareable run receipt."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def persist(path, payload):
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Different evidence already exists: {path}")
        return
    with path.open("xb") as stream:
        stream.write(payload)


def export(run, destination):
    manifest = read(run / "manifest.json")
    for name in ("checkpoint.pt", "results.json"):
        info = manifest[name]
        path = run / name
        if path.stat().st_size != info["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Source receipt mismatch for {name}")
    r = read(run / "results.json")
    supervisor = read(run.with_name(run.name + "-supervisor.json"))
    if supervisor["exit_code"] or supervisor["stop_reason"] or r["status"] != "complete":
        raise ValueError("Exporter requires a complete observed run")
    stage = r["config"]
    if stage["task"] not in {"planning","interval","mixed"}:
        raise ValueError("Expected the included synthetic study")
    rows = []
    checks = {}
    for split in ("dev", "test"):
        raw = r["raw_evaluation"][split]
        if not raw["complete"] or len(raw["rows"]) != raw["summary"]["rows"]:
            raise ValueError(f"Incomplete {split} evaluation")
        checks[split] = dict(raw["summary"])
        decisions = [x for x in raw["rows"] if x["kind"] == "decision"]
        if stage["task"] == "mixed":
            # These two splits consist exclusively of authored fixtures and
            # authored prose. Never load or publish private training passages.
            import mixed_data
            approved = {x["id"]:(x["prompt"],x["reference"]) for x in mixed_data.examples(split)}
            observed = {x["id"]:(x["prompt"],x["reference"]) for x in raw["rows"]}
            if approved != observed or len(raw["rows"]) != len(approved):
                raise ValueError("Exported rows must exactly match authored evaluation split")
        for support, group in ((True, "supported"), (False, "unsupported")):
            subset = [x for x in decisions if x["semantically_supported"] == support]
            checks[split][group+"_first_token_correct"] = sum(x["raw_ids"][:1] == [x["expected_marker"]] for x in subset)
        for x in raw["rows"]:
            # Prompts, reference IDs, primitive events and vocabulary can also
            # be regenerated exactly from task + split + id in this checkout.
            rows.append({k:x[k] for k in ("id","kind","prompt","reference","expected_marker",
                                          "raw_ids","raw_text","token_probabilities","length_capped")})
    resource = r["resources"]
    summary = dict(task=stage["task"], arm=stage["arm"], seed=stage["seed"], step=r["step"],
                   parameters=r["parameters"], blocks=r["blocks"], targets=r["cumulative_training_targets"],
                   parameter_target_products=r["parameter_target_products"], loss=r["loss_evaluation"],
                   greedy=checks, growth=r["growth"], new_parameter_max_abs_deltas=r["new_parameter_max_abs_deltas"],
                   checkpoint_sha256=manifest["checkpoint.pt"]["sha256"], local_results_sha256=manifest["results.json"]["sha256"],
                   model_sha256=r["final_model_sha256"], dataset_sha256=r["dataset_sha256"],
                   source_sha256=resource["source_sha256"], device=resource["gpu_name"],
                   peak_reserved_bytes=resource["cuda_peak_reserved_bytes"], wall_seconds=supervisor["seconds"],
                   worker_cumulative_seconds=r["cumulative_worker_wall_seconds"],
                   ablations={k:v for k,v in r.items() if "ablation" in k})
    if stage["task"] == "mixed":
        summary["training_targets_by_kind"] = r["cumulative_training_targets_by_kind"]
    directory = destination / run.name
    directory.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(x, separators=(",",":"), allow_nan=False)+"\n" for x in rows).encode()
    summary["predictions_sha256"] = hashlib.sha256(payload).hexdigest()
    persist(directory / "predictions.jsonl", payload)
    persist(directory / "summary.json", (json.dumps(summary, indent=2, allow_nan=False)+"\n").encode())
    settings = {k:v for k,v in stage.items() if k not in {"output","resume"}}
    receipt_keys = ("exit_code", "stop_reason", "seconds", "peak_worker_private_bytes", "artifact_bytes")
    recipe = {"config":settings, "supervisor":{k:supervisor[k] for k in receipt_keys}}
    persist(directory / "reproduce.json", (json.dumps(recipe, indent=2, allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.runs:
        export(path.resolve(), args.destination.resolve())
