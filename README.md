# Corpus growth and evidence grounding archive

This snapshot preserves three experiments that preceded query-bound qualification. It contains source, synthetic fixtures and tests. The source manifest maps each file or excerpt to its original Git object.

| Question | Entry point | What it checks |
| --- | --- | --- |
| Can an extra trainable SSM branch be inserted without changing the predictions? | `growth_transfer.insert` | Compares full FP32 logit bits before and after silent insertion, probes two typed decisions and checks the existing Adam state and parameter groups. The caller supplies a disposable model, optimizer and ordinary rows. |
| Can the fitted readout certify a particular ordinary byte? | `grammar_prefix_bound.bound_expected_byte` | Given a hidden-state enclosure, computes a sufficient real-arithmetic margin and reports the numerical error bound needed to transfer it to an implementation. A nonpositive bound is inconclusive. |
| Does truthful evidence supervision help relative to shuffled supervision? | `evidence_recipe.EvidenceRecipe.paired_plan`, `response_step.accumulate_gradients` | The paired curriculum matches prompts, target marginals and ordinary training windows. All supervised bytes share one normalization across microbatches. The matched GRU and SSM produce raw byte logits. |
| Can a learned fact affect a later answer? | `evidence_control.analyze_pair`, `causal_trace_core.trace_pair` | Compare probe predictions, score-margin sensitivities and state swaps while keeping the recipient's typed records and query unchanged. The trace records raw responses separately. Mixed states can be outside the reachable set. |

`pilot_evaluation.development_panel` builds the public development cases. `pilot_evaluation.evaluate` measures free responses, EOS, likelihood and a separate typed authority regression. A favorable margin or a valid typed decision alone cannot establish learned English grounding.

## Run source checks

With Python 3.12, run the synthetic contracts from this directory.

```sh
python -B -m unittest test_grammar_prefix_bound test_comparator_contract test_evidence_recipe test_response_step test_evidence_control.PureEvidenceControlTests test_causal_input_audit test_pilot_evaluation
```

The numerical libraries also require PyTorch. The tests above construct their artificial records in memory. They need no corpus, checkpoints or numerical worker. Fit and tracing require a separately admitted runtime. This release omits the device-specific campaign runner, materialized experiments, private evaluations and all model states.
