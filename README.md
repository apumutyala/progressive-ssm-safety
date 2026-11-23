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

## Comparing outputs honestly

A tiny model can reduce its prediction loss on prose and still produce unreadable text. Evaluate byte prediction, intelligible free generation, decision coverage, semantic contradictions and the formal invariant separately. Greedy generation, typed provenance, a fixed first query and finite FP32 execution are part of the decision contract. Stochastic decoding, misleading synonyms and a false source record require different arguments.

Earlier fictional route experiments are available at the repository's parent release. This branch's tree intentionally excludes their raw answers and all local texts, checkpoints and private evaluation material.

