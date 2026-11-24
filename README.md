# Query-bound SSM qualification

Three byte-level autoregressive models share a 264-token vocabulary and train on the same next-token targets. The two SSM arms combine selective recurrence with causal attention. The decoder arm uses causal attention without the recurrent contribution. The small synthetic task asks the model to bind a named subject, source and epoch to the correct record and finish a reply with EOS.

## Run the fixture checks

With Python 3.12, run the data and schedule checks from this directory in a fresh interpreter. These checks use only the standard library.

```sh
python -B -m unittest test_qualification_data test_qualification_overfit.ContractTests
```

The optional model and scoring tests require PyTorch. The original numerical experiment used eager FP32 with a fixed backend. A source checkout alone cannot reproduce its trained weights or measurements.

## Where to start

- `qualification_model.make_model` constructs the certified SSM, relaxed SSM or decoder using the arm names in `ARMS`. The model exposes full-vocabulary logits, prefill, cached decoding and silent parallel width growth.
- `qualification_base.prepare_base_rows` and `objective` implement the ordinary-language next-token objective. Eight short CC0 fixture passages are available from `qualification_data.ordinary_records`. The optional `base_plan` path requires a separately enrolled, hash-pinned corpus.
- `qualification_overfit.run_slice` implements a supervised in-memory eight-update slice of a synthetic learning check. It requires the caller to admit the workload and supply the runtime guard. `qualification_eval.evaluate` scores free generation, first divergent token and EOS separately.

The protected comparison is between reserved CONFIRM and DEFER logits. It depends on finite FP32 execution, controlled query framing and authentic, timely typed receipts. Ordinary prose is evaluated empirically. A successful fixture test establishes the language-target wiring and cannot certify a generated English statement.

The source manifest identifies exact original files and the single declaration extracted into a portable dependency. This release contains the experiment modules and tests. Machine-specific orchestration, corpora, measurements and checkpoints remain outside the release.
