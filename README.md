# Routed intent qualification

A byte language model sees the request text and predicts a route. A separate typed state records evidence and grants by subject, task, source and epoch. The grammar witness checks that the predicted route names the exact request. It can reject a mismatch. A lawful ALLOW additionally requires affirmative evidence and current authority for that route.

## Run the synthetic boundary tests

From this directory, run the following standard-library command with Python 3.12.

```sh
python -B -m unittest test_routed_qualification
```

The tests exercise valid approvals, negative and missing evidence, wrong people, missing authority, revocations, stale intents, nuisance edits, cache restoration and width growth. They include a wrong-referent ALLOW from a pre-witness proposal.

## Inspect the learning path

`routed_cases.make_cases` builds fictional input ledgers and requests. `routed_state.candidate`, `propose` and `sink` expose the raw candidate, witnessed pre-sink intent and Boolean acceptance respectively. The sink has no external effects.

The CPU-only `routed_neural.RoutedByteLM` uses separate forwards for request selection and ordinary autoregressive prose with shared weights. `routed_measure.measure` takes an arm name and a caller-supplied guard and includes the original two-step learning trial. Its numerical path requires PyTorch. The grammar has a finite set of request spellings and typed events assume authentic delivery. Passing the boundary tests establishes properties of the typed observer and grammar witness. Learned language behavior requires separate measurement.

The source manifest identifies the exact original modules and tests. Machine-specific orchestration, measurements and trained artifacts remain outside this source snapshot.
