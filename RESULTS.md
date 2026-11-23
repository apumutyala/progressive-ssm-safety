# What the laptop experiments established

Training and the matched comparisons ran on the RTX 5070 Ti Laptop GPU. Separate diagnostics below identify their CPU or GPU execution path. Each arm used raw greedy decoding. There was no checker or constrained decoding applied after the model produced scores. Paths under [evidence](evidence/) contain sanitized training settings, per-prompt expected and actual answers, generated token IDs, and probabilities.

## First 256 updates

The interval experiment used one seed (12), the same 320,886 sampled training targets per arm, and 64 synthetic held-out decisions, half supported. Growth appended a second block at update 128 in a deliberate test. At insertion, all **6,741,108** tested vocabulary logits matched exactly on the same execution path. Old Adam moments were retained. After training, zeroing only the new block's feature-to-output weights while keeping its learned biases increased development next-token loss from 1.793 to 2.357. Its learned recurrent features affected prediction.

| Training arm | Parameters | Test next-token loss | Correct first decision | Exact useful continuation | Expected route appears anywhere |
|---|---:|---:|---:|---:|---:|
| Small continuation | 17,237 | 2.291 | 64/64 | 0/64 | 0/64 |
| Small, reset Adam | 17,237 | 2.215 | 64/64 | 0/64 | 0/64 |
| Grow at 128 | 34,474 | 2.103 | 64/64 | 0/64 | 7/64 |
| Larger from scratch | 34,474 | 2.043 | 64/64 | 0/64 | 8/64 |

The test loss includes prompts as well as continuations. All four runs made zero unsupported confirmations. The larger-from-scratch arm had the lowest loss. The grown arm used the larger network for only half the updates. This comparison does not establish a compute advantage or a plateau. The route column is a substring diagnostic; an answer that names several routes can score a hit.

**An actual held-out failure.** For B in the `test-interval-0-slack` query pair, the reference was “&lt;defer&gt; hold plan b . intended route is garden by boat . resolve the capacity shortfall .” The grown model generated “&lt;defer&gt; hold plan a capacity lower 1000000 upper 1000000 units. response: &lt;defer&gt; hold plan a capacity lower 1000000 upper 1000000 units.” The first token was correct. The named plan and figures were wrong. That distinction is the central empirical result of the first comparison.

An earlier two-flag pilot had 0/96 correct first decision tokens after 128 updates. On that task an absence of unsafe confirmations was entirely compatible with failure to emit an answer. Its exact executable worker predates a JSON-serialization-only change, at commit `36ff1313cbc6b50cf7549558b241d1a8f40321a8`.

## Continuing to 1,024 updates

| Arm | Exact decision continuations at 256 | At 512 | At 1,024 | Test all-token loss at 1,024 | Cumulative parameter × target count |
|---|---:|---:|---:|---:|---:|
| Small continuation | 0/64 | 0/64 | 45/64 | 2.098 | 22.14 billion |
| Small, reset Adam | 0/64 | 32/64 | 45/64 | 2.210 | 22.14 billion |
| Grow at 128 | 0/64 | 45/64 | 45/64 | 2.535 | 41.53 billion |
| Larger from scratch | 0/64 | 45/64 | 45/64 | 2.302 | 44.27 billion |

At 1,024 updates each arm had seen the same 1,284,254 next-token targets. All got the first decision right on 64/64 prompts. Their per-example mistakes differ. The larger models acquired the common answer patterns earlier. The small continuation eventually matched their exact-answer count, with the lowest held-out loss and roughly half the parameter-target product. More parameters and lower training loss do not establish that the model uses the evidence in its prose.

A retrospective diagnostic exposed a shortcut. Fit just four strings to the training data, one per (requested plan, known feasibility) pair, by choosing the most common reference for each pair. That table also scores **45/64 exact**. It receives the feasibility bit directly; the interval model derives that bit from primitive receipts. The diagnostic tests what the model learns *in addition* to that hardwired calculation. At 1,024 updates, 44–50 of the 64 outputs per arm reproduce that table's selected strings exactly. Run [shortcut.py](shortcut.py) for all answers. This benchmark cannot establish robust learned route selection from its aggregate score.

**Swapping the actual route falsified the stronger interpretation.** We reran the grown 1,024-update checkpoint on CPU. For each of 64 original test prompts, we swapped the destination and transport between the two plans, keeping the query, all receipt values and every other token fixed. On the original prompts the correct route occupied its required position in 48/64 answers. On the counterfactuals it occupied that position in 0/64. **Exactly 0/64 generated answers changed.** The model still emitted the correct decision token on both sides of all 64 pairs. The experiment isolates a specific failure: in this test the learned route description did not respond to the information the user had changed. See [paired results](evidence/route-counterfactual-summary.json) and [every original and swapped generation](evidence/route-counterfactual-pairs.json).

The next design should balance destination and transport independently of feasibility, requested plan and explanation template. Train on counterfactual pairs with genuinely different next-token targets and hold out new templates, route combinations and longer delays. Compare the same recurrent backbone against a learned memory that binds each route to its plan. Require correct route changes on both members of a pair, positive decision coverage, and the same guaranteed margin before increasing model size again. More training on the present examples is a poor test of that hypothesis.

## Ordinary-language pretraining

The mixed model has 38,167 parameters and 583 output IDs. Its byte-level channel can encode arbitrary UTF-8. It trains on the same primitive planning cases plus 58 disjoint prose windows sampled from 18 local train-only document fragments. The distinct normal-prose content in this pilot is 7,304 tokens. At 1,024 updates, the model had seen 724,562 planning and 754,287 prose next-token targets, including repetitions. The four authored test prose examples contain only 570 UTF-8 bytes in all. These metrics describe that tiny sample.

| Updates | Test prose bits per UTF-8 byte (lower is better) | Correct first decision | Exact decision continuation |
|---|---:|---:|---:|
| 128 | 4.081 | 0/64 | 0/64 |
| 512 | 3.528 | 64/64 | 0/64 |
| 1,024 | 3.534 | 64/64 | 45/64 |

Free prose remained repetitive. The mixed model did not trigger the preregistered plateau growth rule by update 1,024. Its useful size remains an empirical question. An independent CPU prompt and its full continuation are in [the ordinary inference receipt](evidence/ordinary-interactive.json). The same checkpoint generated a correctly scoped confirmation followed by an incorrect route in [the planning example](evidence/decision-interactive.json).

## What the checks establish

The integrated CPU suite passed 25 tests. On the actual GPU, 384 positions per parameter setting were checked under initial, extreme positive, extreme negative, and newly grown weights. The smallest observed signed margin was 0.5 and all 7,296 compared logits matched at growth in that stress fixture. The scope and assumptions of the analytic guarantee remain those in [PROOF.md](PROOF.md). See [GPU results](evidence/cuda-device-check.json).

We also tested whether the trained learned state could be proved independent of its unknown prior value after each of two eight-token route descriptions. Neither suffix erased all possible prior learned states on the tested CPU path. Maximum hidden-coordinate widths remained about 1.86 and 1.85 on a total allowed width of two. This is an inconclusive certificate of forgetting; it does not tell us which past fact the readout used. See [the two reports](evidence/prefix-probe.json).
