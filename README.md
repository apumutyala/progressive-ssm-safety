# Controlled state-space language pretraining

This branch trains an autoregressive state-space language model from random initialization on ordinary UTF-8 text. It predicts the next byte. The learned state transition and output readout are constrained at initialization, during Adam updates and at zero-connected growth. A matched comparison model has the same parameter shapes and data access and relaxes both constraints.

The central question is whether an architecture can learn to predict language while retaining a property proved over **every admitted set of weights**. The first property concerns the hidden state throughout ordinary prose. The second concerns a reserved decision token when the input contains authenticated facts. Neither establishes the truth or safety of arbitrary generated English.

## Where to start

- [Smoke protocol](SMOKE_PROTOCOL.md) describes the corpus-first comparison, checkpoint-resume test and held-out evaluations.
- [Control argument](CONTROL.md) defines the state set, assumptions, optimizer closure, influence of prior context and proof-preserving insertion.
- [Measured results](CORPUS_RESULTS.md) reports the corpus split, actual update counts, training and held-out losses, state bounds, free-language failures and limits.
- `corpus_io.py` validates externally supplied documents, rights metadata and split assignments. `corpus_model.py` implements byte encoding, prefill, cached decoding and the actual constrained and comparison recurrent dynamics.
- `pretrain.py` is the training entry point. `semantic_eval.py` is a separate downstream evaluator. `corpus_sensitivity.py` tests the influence of earlier hidden state on ordinary byte sequences. Evaluation cases are never passed to the optimizer.

Provide an authorized local corpus and an external manifest. Source and work identity determine train/development/evaluation splits; the tokenizer does not learn a vocabulary from any split. The smoke uses a single small literary training source. Larger, more diverse corpora would be needed for general language capability.

## Run it on the Omen

The supported numerical environment is Windows, Python 3.12.10 and PyTorch 2.14.1+cu130. The launcher verifies the installed overlay and GPU, bounds each worker to 110 seconds, and checks shared RAM, commit, free disk, VRAM and retained artifact capacity. Run one worker at a time. The artifact directory `../certified-runs` must exist beside the checkout. Set the interpreter and overlay to the installations already approved on your machine.

For a newly curated dataset, place a JSON specification outside this checkout. Its `documents` list names local UTF-8 files. Each record supplies `id`, `path`, `split` (train/dev/test), `group` (work), `author`, `source_id`, `license_id`, `rights_status`, `model_fit_allowed`, `evaluation_allowed` and a list of local `evidence_paths` for the actual rights/provenance receipts. Keep authors and works in one split each. Currently approved rights-status enums are `CC0_original_fixture` and `publisher_US_PD_belief_CC0_contributions`; use them only when the rights evidence actually supports the description. The constructor computes hashes and checks text overlap, but it cannot establish copyright status. Texts, licenses, per-example outputs and the private manifest stay outside Git.

```powershell
$pin = & $py -B prepare_corpus.py --spec $localSpec --root $corpusRoot --output $privateManifest
& $py -I -S -B launch.py --site $torchSite --mode test --receipt ../certified-runs/new-cpu-tests.json
& $py -I -S -B launch.py --site $torchSite --mode corpus --receipt ../certified-runs/new-cert-64-supervisor.json -- --arm certified --device cuda --steps 64 --manifest $privateManifest --manifest-sha256 $pin --corpus-root $corpusRoot --output ../certified-runs/new-cert-64
```

To resume, use a fresh receipt and output directory, supply `--resume` pointing at the preceding `checkpoint.pt`, and pass its exact SHA-256 using `--resume-sha256`. The additional step count must keep the total at or below the fixed 128-update smoke endpoint. For a comparison from scratch select `--arm unconstrained`. The actual run specification, source counts, stop criteria and evidence are in [the protocol](SMOKE_PROTOCOL.md) and [the results](CORPUS_RESULTS.md). Terminal downstream evaluation requires both `--semantic-fixture` and `--semantic-sha256`. Its schema is documented in the scorer. Those fictional prompts play no part in training.

## Comparing outputs honestly

A tiny model can reduce its prediction loss on prose and still produce unreadable text. Evaluate byte prediction, intelligible free generation, decision coverage, semantic contradictions and the formal invariant separately. Greedy generation, typed provenance, a fixed first query and finite FP32 execution are part of the decision contract. Stochastic decoding, misleading synonyms and a false source record require different arguments.

Earlier fictional route experiments are available at the repository's parent release. This branch's tree intentionally excludes their raw answers and all local texts, checkpoints and private evaluation material.

