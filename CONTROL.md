# A state invariant during language-model training

Let each block of the recurrent model have width `w`. The learned state space we control is the Cartesian product of coordinate boxes `[-1,1]^w`. A trusted integer record for each modeled plan and a latched query live alongside that learned state. The records have different semantics from the language memory. Containing a hidden activation inside a box does not make the English it represents safe.

## Transition and optimizer

Write the complete state as `x=(h,r,q)`, where `r` contains two sets of three trusted integer fields and `q` is unbound, A or B. An input `u=(token,event)` contains a byte or explicitly framed reserved ID and an optional typed replacement/revocation of a primitive field. On the normal training path the event is HOLD. `F_theta` updates the real coordinates by their controlled recurrence, replaces or retains the named integer coordinates, and admits only the first query into the latch. In free generation, the greedy output is fed back as the next token with HOLD events. This is the closed loop whose protected output ordering is studied. It is not a model of a human or institution correcting an arbitrary English instruction.

For each observed byte or reserved token, two trainable embedding tables provide an input `u` and a retention coefficient `a` for every coordinate. The certified arm maps its raw values through bounded functions so that `|u| ≤ 1` and `1/256 ≤ a ≤ 255/256`. It executes the elementwise update

```text
h_next = clip(a*h + (1-a)*u, -1, 1)
```

At initialization the hidden state is zero. If the previous state lies in the box, the unclipped real-valued update is a convex combination of two members of the box. The final clamp defends that range under rounding. Induction covers every input sequence length, including freely generated continuations. Multiple blocks use the same argument independently. Data need not belong to the training distribution.

The discrete-time barrier candidate `B(h)=1−||h||∞` therefore satisfies `B(h)≥0 ⇒ B(F_hidden(h,u))≥0` uniformly in token and admissible weight. The reachable enclosure is the product of this box, the legal typed field values (including unknown), and the three query states. The unsafe *state–output* set is a state with unsupported selected evidence paired with the emitted reserved CONFIRM ID. A point in the box can represent malicious intent or incorrect prose. Safety here follows from the enclosure **and** the signed readout on the protected record, not from any interpretation of the box as aligned thought.

Adam modifies the raw weights. Every finite admitted set of those weights still maps to the same allowed coefficient and drive sets. This is architectural closure; the optimizer does not project a learned notion of human values or search a specially verified loss basin. The runner checks finite values and instruments the forward computation that actually supplies the training loss and the post-update recheck. The matched comparator can drive a coordinate outside the box and admit retention above one. Checking how often it actually does so on through-training batches is a measurement distinct from showing such a transition is reachable.

## What survives a change in context

For two starting states exposed to the same subsequent bytes, their real-arithmetic hidden-state difference along a coordinate is bounded by the product of that suffix's retention coefficients times the initial difference. The global upper bound across `k` steps is `(255/256)^k` times the starting infinity-norm difference. Clipping is nonexpansive. A bound on the distance between real-valued activations is not an exact machine-erasure theorem.

For identical trusted updates, the protected records have a different metric. With equal latched queries, Hamming distance between two record arrays cannot increase: a HOLD keeps that coordinate's disagreement, and an overwrite with the same new value erases it. Thus initial record disagreements can persist indefinitely, and identical authoritative records remain identical. Changing the receipt itself intentionally changes the decision comparison. Ordinary byte perturbations cannot cross that information boundary. This non-amplification claim assumes equal framing and trusted inputs; an incorrect source update can still authorize the wrong action.

The finite-precision diagnostic runs the executed monotone update from lower and upper endpoints with equal protected records and a common ordinary suffix. Where endpoints become provably identical under its numerical assumptions, intervening hidden states produce the same endpoint. Otherwise the remaining box width measures uncertainty about earlier context. This property concerns influence through the recurrent state. It can forget a secret or it can forget a fact the user needed. Whether the model preserves the *right* facts requires separate evidence.

## Which changes are visible at the output?

The [local systems probe](control_probe.py) evaluates a finite horizon of the fitted model. For each input position it forms the diagonal state Jacobian `A_t` from that token's effective retention and the local derivative of the projection. A discrete token ID has no ordinary derivative. The probe uses a stated **continuous hypothetical perturbation to the learned input drive** in one direction; its state derivative is `B_t`. It separately measures an actual substitution of one byte for another. `C_t` is the local gradient of an ordinary byte-logit contrast with respect to the hidden state, through the model's actual vocabulary head.

With `Φ(t,s) = A_t ... A_(s+1)` and identity for an empty product, the observability Gramian sums `Φ(t,0)ᵀ C_tᵀ C_t Φ(t,0)` over measured outputs. A terminal controllability Gramian sums `Φ(T,t) B_t B_tᵀ Φ(T,t)ᵀ` over the hypothetical input pulses. Eigenvalues and numerical rank depend on the chosen trajectory, input direction, output contrast, tolerance and horizon. A large reachable direction need not be visible to that output. No rank here certifies global controllability, full observability or the meaning of a learned coordinate. The finite suffix enclosure above is the uniform reachability argument; these local measurements answer a different question.

## Protected output ordering

For the first authenticated query, the model checks permit and a conservative capacity-vs-demand comparison from typed, correctly attributed integer receipts. Words in normal input or generated prose cannot write a receipt; literal UTF-8 spellings of reserved symbols have distinct byte IDs. With two Boolean premises, let `s = 2 × (permitted + sufficient) − 3`. A bounded learned adjustment in `[-0.5,0.5]` is added to `s` and then to the DEFER logit to obtain the CONFIRM logit. Both compete against the entire vocabulary.

If either premise fails, the CONFIRM logit is at least approximately 0.5 below DEFER after FP32 rounding, so it cannot be a greedy winner. Other words can still be generated. With both premises true, CONFIRM outranks DEFER, but an ordinary word may still win. This distinction is why we report both support-conditioned coverage and unauthorized confirmations. We assume current authentic receipts, correct numerical bounds and query identity, finite admitted FP32 weights, and the supported decoding path. No part of the proof turns matching English phrases into trusted facts.

## Growth and semantic alignment

Growth appends an independent recurrent block whose vocabulary projection starts at zero. The existing block is evaluated on its original shape. On an unchanged execution path the aggregate vocabulary scores and the protected decision ordering are preserved at insertion. New weights can subsequently change ordinary predictions but still produce bounded effective transitions and pass through the same signed decision head. The insertion test does not establish that the enlarged model learned more or that training had reached a plateau.

The harder step is to verify claims about meaning. One concrete experiment would define a small language of factual claims with a public interpretation, vary evidence independently of wording, and ask the model to generate supported, contradicted and unknown statements under held-out paraphrases. An exact verifier could certify a restricted language or prove a particular generation, with independent coverage and abstention measurements. A guarantee on unrestricted English would also need a sound interpretation covering that English. Neither low byte loss, latent-state contraction nor a lexical contradiction test supplies it.
