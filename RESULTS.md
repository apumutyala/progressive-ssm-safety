# What the laptop experiments established

All numbers here are from the RTX 5070 Ti Laptop GPU on the same Windows machine. Each arm used raw greedy decoding. There was no checker or constrained decoding applied after the model produced scores. Paths under [evidence](evidence/) contain sanitized training settings, per-prompt expected and actual answers, generated token IDs, and probabilities.

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
