"""Fixed training-case probes. Raw learned selection, witness and sink never share scores."""
from routed_cases import make_cases
from routed_measure import score_predictions


def measure(arm, check, device):
    from routed_fit_neural import run_arm
    cases = make_cases()
    fit = run_arm(arm, cases, check, device)
    guesses = fit.pop("probes")
    if [p["step"] for p in guesses] != [0, 4, 8] or fit.pop("predicted_routes") != guesses[-1]["predicted_routes"]:
        raise ValueError("Incomplete fixed probe sequence")
    probes = []
    for probe in guesses:
        sample = score_predictions(arm, cases, probe["predicted_routes"], check)
        sample.pop("neural")
        sample["step"] = probe["step"]
        probes.append(sample)
    training = fit["training"]
    if (training["steps"], training["selector_targets"], training["prose_targets"], training["eos_targets"]) != (8,136,1216,48):
        raise ValueError("Exposure mismatch")
    if len(fit["prose"]) != 8 or len({p["dataset_sha256"] for p in probes}) != 1:
        raise ValueError("Incomplete or mixed evaluations")
    return dict(arm=arm, device=device, dataset_sha256=probes[0]["dataset_sha256"], probes=probes, neural=fit)
