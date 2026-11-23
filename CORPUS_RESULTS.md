# What survived ordinary-text pretraining

The constrained model learned regularities in English while its recurrent state stayed inside the proved box at every measured training step. On a held-out work, the comparator predicted the next byte slightly better. Neither model answered any of the downstream questions usefully. This experiment demonstrates training and exact resumption with a specific architectural guarantee. It leaves open how to make that guarantee matter for ordinary language.

## Paired GPU results

We trained two 17,432-parameter autoregressive models from the same random tensors on the same 1,024 sampled windows. Each saw 131,072 ordinary UTF-8 byte targets across 128 Adam updates. There were zero synthetic or framing targets in these samples. Of 46,411 available training bytes across four chapters of *Treasure Island*, the windows covered 43,410 distinct **positions**. Total exposure includes repeated reading.

| Measure | Constrained | Comparison |
|---|---:|---:|
| Initial development loss, nats per byte | 5.5973 | 5.5991 |
| Final development loss, nats per byte | 3.0422 | 3.0069 |
| Final test loss, nats per byte | 3.1028 | 3.0583 |
| Final test bits per byte | 4.4763 | 4.4122 |
| Final test byte perplexity | 22.26 | 21.29 |
| Hidden coordinates outside the box, pre-update / post-update | 0 / 0 | 24 / 29 |
| Correct first typed decision among six raw continuations | 0 / 6 | 0 / 6 |
| Accepted free answers to six prose questions | 0 / 6 | 0 / 6 |

The development set uses *The Wind in the Willows* and the test set uses *Little Women*. These authors and works are distinct from training. Each score uses 2,048 byte targets in 16 disjoint windows. The test work was publicly known before training. The figures come from one seed and one training work. The comparator relaxes the drive, recurrent box and decision adjustment together, so this comparison cannot isolate their individual costs. The identical initial tensors also have different effective functions.

## Control on the trained model

- Across 128 pre-update and 128 post-update passes, the constrained hidden extrema stayed between −0.7330 and 0.7442. The comparator reached 1.0542. Its 29 post-update excursions count coordinate observations, so repeated excursions need not be independent.
- Ordinary prose cannot supply a trusted receipt. On those batches the constrained raw DEFER score exceeded CONFIRM by at least 3.0126 before updates and 3.0199 after updates. Neither arm emitted a reserved unsupported CONFIRM in these training measurements. The constrained family also excludes that greedy decision at **any** admitted weights when the selected plan lacks support. The proof uses the trusted record and bounded adjustment in the readout. It relies on the accuracy and attribution of the record. [Control argument](CONTROL.md)
- With identical bytes but a different trusted capacity, the constrained model's CONFIRM-minus-DEFER margin changed from −1.4580 to +0.5420. Learned state and ordinary byte scores were bit-identical. An ordinary byte won in both cases. The architectural channel retained the fact and changed the ordering, but the LM did not choose to use it.
- On a fixed public byte substitution, the difference in learned hidden state fell from 0.4096 immediately to 0.03057 after 16 common bytes and 0.00311 after 64. The next greedy token was identical at all four measured lags. A changed fact entering hidden state therefore did not suffice to change the chosen output in this probe.
- On one 16-byte development trajectory, the local controllability and observability Gramians had numerical ranks 15 and 14 out of 16 for the declared continuous drive perturbation and ordinary logit contrast. Distinct ordinary bytes are discrete interventions. These ranks describe the linearization around that trajectory.
- Across eight other development windows, a common 128-byte suffix shrank the worst endpoint-enclosed coordinate width from 2 to 0.69373. In 64 of 128 coordinates the machine endpoints met; no complete row established independence from every possible prior hidden state. On the same model, different coordinates can lose or retain information for quite different lengths of time.

## Where the model failed

All 12 downstream raw generations in each arm reached the 48-token cap. The scoring rule could classify none as a correct or explicitly contradicted complete response. Neither model supplied the expected marker on any of the six typed cases, including two with enough evidence to approve. The certified model ranked the correct reserved marker above its opponent under teacher forcing in all six typed cases. Its failure was generating an appropriate response from the whole vocabulary. On the six English cases the constrained model preferred the supplied correct continuation by mean likelihood in 2/6; this metric can depend on candidate length and wording.

There is a mismatch worth correcting in the next experiment. Our scorer requires a complete answer ending at EOS, but none of the sampled training windows supplied an EOS target. A fresh reviewer independently decoded and rescored every private continuation. All 12 continuations in each arm contained only whitespace. A more forgiving termination rule would still leave no answer to evaluate. The entire between-arm difference in mean candidate preference came from two supported **typed** cases. Both arms preferred the correct English continuation in the same two of six cases. Future evaluation should report content quality separately from termination and teach termination on genuine document boundaries.

We retain the exact per-case prompts, references and raw continuations privately for error analysis. The released aggregate contains no raw text. Passing a reserved action-token guarantee while producing an unusable answer is possible in practice. Neither sampled nonviolation nor a language loss of 3.10 demonstrates aligned reasoning.

## Growth, resumption and local cost

At the constrained endpoint we doubled the model to 34,864 parameters by appending a block with a zero output projection. All 270,336 forward logits and all 270,336 prefill logits in the comparison matched at insertion. The moved model was a disposable copy. We did not train this larger checkpoint or observe a plateau. In fact the constrained development loss still improved from 3.1629 at update 64 to 3.0422 at 128.

Independently training to 64, saving, and continuing to 128 reproduced the uninterrupted endpoint. A second verifier loaded both checkpoint byte streams and checked the model tensors, Adam, Python/Torch/CUDA RNG, sampler windows, input identities, recipe and target counts. It bound its answer to the exact two checkpoint hashes. The two serialized files contain different wall-clock metadata.

Four serial supervised CUDA jobs completed on an NVIDIA GeForce RTX 5070 Ti Laptop GPU. Supervisor durations were 10.06 seconds (first 64), 11.66 seconds (continuation), 15.49 seconds (uninterrupted constrained) and 12.47 seconds (uninterrupted comparison). Peak reserved CUDA memory in each full run was 88 MiB. Durations include setup, audits, evaluation and writing, so they measure end-to-end time for this recipe. The laptop has substantial headroom at this size.

## Next decision

**First, separate the cost of the recurrence from the cost of the readout.** A four-arm comparison should independently toggle the two constraints while retaining the exact ordinary-text objective and splits. Use new approved works, more than one seed, and a fresh unseen evaluation. Compare with a byte-frequency or n-gram LM trained on exactly the same bytes before attributing a loss improvement to learned reasoning.

**Next, test which information deserves persistent state.** Keep ordinary text as the majority of the token stream. In a separately budgeted treatment, mix in paired, multi-token examples where identical wording refers to different entities or a newer observation changes the answer. Give the comparator the same supervision. Require useful affirmative answers as well as correct deferrals, including held-out evidence changes and wording. Compare the learned causal state change with the behavior obtained from a trusted record. Record the exact raw generations privately, including incorrect fluent assent.

**Grow on development evidence.** For a follow-on protocol, one candidate budget is 2 million training byte targets per seed and arm, evaluated every 128 updates. At that rate a four-arm, three-seed sweep needs 24 million total byte targets before growth comparisons. Freeze the corpus, thresholds and held-out families before that study. A candidate growth trigger is three consecutive checkpoints with under 0.01 nat/byte improvement against the best preceding checkpoint and no gain in development task coverage. Compare an insertion against same-size continuation, a learning-rate restart and a larger model trained from scratch. Report cumulative targets and measured compute including all pre-growth work. Authorize more compute only from measured throughput, memory and the number of distinct runs required for a predeclared effect. This smoke gives no basis for pricing H100 time or calling the laptop exhausted.
