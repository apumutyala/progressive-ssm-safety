# Paired route and evidence binding archive

This source snapshot preserves two experiments with a small autoregressive selective state-space language model. The paired-route comparison changes the learned residual on the reserved decision readout while keeping transitions, signed premise prior and parameter count matched. The evidence-binding comparison changes how a prompt memory contributes to vocabulary scores. One arm uses learned attention over seven-token key windows to copy prompt tokens. Its control uses a uniform histogram over the same visible tokens. Both compute the same address logits, allowing equal address supervision.

These components answer different questions from selecting a route for a typed intent. A model can order the two decision tokens correctly and still copy the wrong entity or destination. The conditional ordering assumes finite admitted FP32 parameters, the intended framed query token, and timely, trustworthy typed receipts in common units. The ordering does not establish which full-vocabulary token wins, whether generated prose is correct, or whether either trained model learned to bind evidence. This archive contains no fitted weights or performance measurements.

## Use

Run `python -B -m unittest -v test_metrics` in this directory to exercise scoring with a newly constructed pair. This standard-library smoke checks exact paired answers, a constant lookup shortcut, first divergence, field agreement and truncation. It does not execute the neural models.

The original model modules require an existing PyTorch installation. `binding_model.BindingModel` offers the `addressed` and `pooled` arms, `teacher` with a shared public prompt boundary, and incremental `prefill` and `decode_step`. The caller provides token IDs and any trusted receipt events. `paired_model.make_model` constructs the certified and comparator readouts. The comparator relaxes only the decision residual. Its transition remains bounded. The adverse-update fixture is separate from these constructors.

For scoring caller-supplied records, pass token spellings and reserved IDs through `paired_metrics.Vocabulary`. Marker spellings must be lowercase. `evaluate` accepts a prediction for each row and requires two distinct sides per pair. It records exact agreement, response changes and a narrowly specified recommendation grammar. `distance_metrics.first_error` reports the first mismatched or missing token position. Model fitness and a historical run cannot be reconstructed from this module smoke.

## Provenance

The paired-route source commit is `41253df0724bf356e151b3825461c32550f521f3`. The evidence-distance source commit is `5a8cc0eca124b20148e93148004bc2d68bfe358d`. Git blob IDs below identify the original files.

| Archive file | Original pin | Original Git blob | Export |
|---|---|---|---|
| model.py | paired-route | `05e4bc5f97529d9bdca2603e6444ef59fc3f17ab` | Identical |
| interval_model.py | paired-route | `e85fb2f791a206e81f70dc021293e980103002e0` | Identical |
| paired_model.py | paired-route | `ab1ae9e9f88d9e79b23a6f8b92dee980285e481d` | Identical |
| binding_model.py | evidence-distance | `aa823d1e711031bd62d42ca3bf688dd2f37f74ed` | Identical |
| paired_metrics.py | paired-route | `6dc5dd96f16c0c779e5e6e1e1a7c56912e0b7c44` | Original scoring logic with an explicit vocabulary argument replacing the dataset import |
| distance_metrics.py | evidence-distance | `eba8a455062d427f5989639715aeaaf3844852f5` | Two sequence helpers from distance_score.py without its run-specific reader |

The historical dataset generators, run supervisors and evaluation records are outside this snapshot. The README and smoke are new.
