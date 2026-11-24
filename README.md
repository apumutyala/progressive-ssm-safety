# Routed evidence and intent fit

This eight-update experiment asks whether a small byte language model can select the subject, task, source and epoch named in a request and generate an evidence-dependent answer. It reuses the controlled grammar and 74 synthetic cases from the two-update CPU qualification. The device-aware fit has three arms: a constrained SSM with attention, a relaxed SSM with attention and a causal decoder.

## Check the scorer

Run the following standard-library checks in a fresh Python 3.12 interpreter.

```sh
python -B -m unittest test_routed_qualification test_routed_fit.TestRoutedFit.test_grammar_refusal_is_separate_from_learned_abstention test_routed_fit.TestRoutedFit.test_exact_diagnostic_route_satisfies_all_pairs
```

The two additional tests in `test_routed_fit` inspect the data schedule and compact event codec. They import PyTorch through the fitting module.

## Read the experiment

`routed_fit_neural.training_plan` enumerates eight batches. Each update supervises 17 selector targets and six responses. Each of the 16 supported routes appears once with each of ALLOW, NO and UNKNOWN in the response targets. Response training includes 1,216 next-token targets, of which 48 are EOS. The selector consumes the request bytes. The prose forward receives the chronological ledger followed by those request bytes.

`routed_fit_measure.measure(arm, check, device)` is the guarded experimental entry point for deterministic, single-thread FP32 execution. It probes route predictions before training and after updates four and eight. The final probe includes eight raw full-vocabulary greedy replies, their EOS or truncation status, first divergence and teacher-forced metrics. All scored cases come from the synthetic study. These measurements characterize fixture fit.

`routed_measure.score_predictions` reports the learned scope, proposal before witness validation, exact pre-sink intent, and acceptance separately. The verifier uses an exact finite request grammar and trusted typed event delivery. A wrong route can be rejected even when the model itself would proceed. The sink is a Boolean simulation.

## Reproduction scope

The source manifest pins the model, fixture, fitting and scoring modules. Machine-specific admission, dispatch, reports and checkpoints remain outside this release. A caller running the numerical fit must supply its own admitted runtime and resource checks. There is no GPU result in this snapshot.
