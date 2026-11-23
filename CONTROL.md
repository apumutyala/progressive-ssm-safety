# A state invariant during language-model training

Let each block of the recurrent model have width `w`. The learned state space we control is the Cartesian product of coordinate boxes `[-1,1]^w`. A trusted integer record for each modeled plan and a latched query live alongside that learned state. The records have different semantics from the language memory. Containing a hidden activation inside a box does not make the English it represents safe.

## Transition and optimizer

For each observed byte or reserved token, two trainable embedding tables provide an input `u` and a retention coefficient `a` for every coordinate. The certified arm maps its raw values through bounded functions so that `|u| ≤ 1` and `1/256 ≤ a ≤ 255/256`. It executes the elementwise update

```text
h_next = clip(a*h + (1-a)*u, -1, 1)
```

At initialization the hidden state is zero. If the previous state lies in the box, the unclipped real-valued update is a convex combination of two members of the box. The final clamp defends that range under rounding. Induction covers every input sequence length, including freely generated continuations. Multiple blocks use the same argument independently. Data need not belong to the training distribution.

Adam modifies the raw weights. Every finite admitted set of those weights still maps to the same allowed coefficient and drive sets. This is architectural closure; the optimizer does not project a learned notion of human values or search a specially verified loss basin. The runner checks finite values and instruments the forward computation that actually supplies the training loss and the post-update recheck. The matched comparator can drive a coordinate outside the box and admit retention above one. Checking how often it actually does so on through-training batches is a measurement distinct from showing such a transition is reachable.

## What survives a change in context

For two starting states exposed to the same subsequent bytes, their real-arithmetic hidden-state difference along a coordinate is bounded by the product of that suffix's retention coefficients times the initial difference. The global upper bound across `k` steps is `(255/256)^k` times the starting infinity-norm difference. Clipping is nonexpansive. A bound on the distance between real-valued activations is not an exact machine-erasure theorem.

The finite-precision diagnostic runs the executed monotone update from lower and upper endpoints with equal protected records and a common ordinary suffix. Where endpoints become provably identical under its numerical assumptions, intervening hidden states produce the same endpoint. Otherwise the remaining box width measures uncertainty about earlier context. This property concerns influence through the recurrent state. It can forget a secret or it can forget a fact the user needed. Whether the model preserves the *right* facts requires separate evidence.

## Protected output ordering

For the first authenticated query, the model checks permit and a conservative capacity-vs-demand comparison from typed, correctly attributed integer receipts. Words in normal input or generated prose cannot write a receipt; literal UTF-8 spellings of reserved symbols have distinct byte IDs. With two Boolean premises, let `s = 2 × (permitted + sufficient) − 3`. A bounded learned adjustment in `[-0.5,0.5]` is added to `s` and then to the DEFER logit to obtain the CONFIRM logit. Both compete against the entire vocabulary.

If either premise fails, the CONFIRM logit is at least approximately 0.5 below DEFER after FP32 rounding, so it cannot be a greedy winner. Other words can still be generated. With both premises true, CONFIRM outranks DEFER, but an ordinary word may still win. This distinction is why we report both support-conditioned coverage and unauthorized confirmations. We assume current authentic receipts, correct numerical bounds and query identity, finite admitted FP32 weights, and the supported decoding path. No part of the proof turns matching English phrases into trusted facts.

## Growth and semantic alignment

Growth appends an independent recurrent block whose vocabulary projection starts at zero. The existing block is evaluated on its original shape. On an unchanged execution path the aggregate vocabulary scores and the protected decision ordering are preserved at insertion. New weights can subsequently change ordinary predictions but still produce bounded effective transitions and pass through the same signed decision head. The insertion test does not establish that the enlarged model learned more or that training had reached a plateau.

The harder step is to verify claims about meaning. One concrete experiment would define a small language of factual claims with a public interpretation, vary evidence independently of wording, and ask the model to generate supported, contradicted and unknown statements under held-out paraphrases. An exact verifier could certify a restricted language or prove a particular generation, with independent coverage and abstention measurements. A guarantee on unrestricted English would also need a sound interpretation covering that English. Neither low byte loss, latent-state contraction nor a lexical contradiction test supplies it.
