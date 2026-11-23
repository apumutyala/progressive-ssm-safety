# Certificate, growth map, and boundaries

This is a proof for the family implemented in `model.py`, not a proof inferred
from training loss. Its semantic conclusion is deliberately narrow: a raw greedy
next-token choice cannot be the atomic `<confirm>` decision unless two modeled
authorizations have current support in an authenticated ordered ledger. The
learned language state supplies ordinary output scores. Two protected Boolean
coordinates carry the evidence context without forgetting. The output equation
is part of the model's readout; no post-generation checker supplies the margin.

This construction fixes the polarity between `<confirm>` and `<defer>` from the
two coordinates. It does **not** learn how to adjudicate these authorizations.
Useful route selection, prose, choice to emit a decision rather than another
word, and generalization remain empirical questions. The proof by itself is not
evidence that the model uses its learnable state for useful work.

## Admitted executions and semantic interpretation

1. Tokens belong to the fixed finite vocabulary. Batch prefixes are nonempty.
   Token values, order, and length may be arbitrary, including misleading prose
   and repetitions of decision tokens. There is no assumption of an honest user.
2. Every coordinate's event is a signed integer: `0` holds, `+1` establishes
   support, and any other nonzero value clears support. Missing events mean zero.
   Each event applies when its associated input token is consumed, before that
   token's next-token logits. The corpus uses only `-1, 0, +1`.
3. Only the trusted environment supplies events. Generated words and untrusted
   customer prose supply zero events. A permit event and a capacity event refer
   to the **requested move**, and both must be valid authorizations for it. The
   current ledger order determines supersession. A `disputed` record clears
   support; a later authentic `yes` resolves it under this toy ledger's rules.
   Authentication, binding to the request, and faithful uncertainty annotation
   are assumptions, not learned or established by this code.
4. Model parameters are finite FP32, block width is in `[1,8192]`, and execution
   uses the reviewed eager path with autocast disabled. The runner must validate
   parameters after initialization, each optimizer update, and restore, and
   reject nonfinite logits. A permitted update is one that passes this admission
   check; an optimizer producing NaN/Inf is a failed run, not an admitted model.
5. A `State` originates from this model's initial state and the admitted
   prefill/decode path at its current growth stage. No caller mutates, forges,
   replaces, cross-batches, or reuses a state after growth or parameter change.
   Python's frozen dataclass does not make its tensors immutable or authenticate
   their provenance. The library does not enforce all of these preconditions.
6. The raw decision rule is `argmax` of the complete returned logit vector, with
   ordinary ties allowed among other tokens. No temperature, probability
   sampling, token bias, logit processor, quantization, compilation, or alternate
   kernel is covered unless independently verified to preserve the bounds.

The decision `<confirm>` means only “both modeled authorizations currently
support this move.” For an unsupported move, missing, revoked, or unresolved
disputed authorization is sufficient to prohibit this marker. No other word is
assigned this verified meaning. See `DATA.md` for exact prompts and references.
The theorem does not certify that a generated route is correct, or that prose
following a marker is true, complete, helpful, or harmless.

## Trainable family and invariant set

For a block and token `x`, the effective drive is

`u(x) = clip(tanh(clip(E_u[x],-16,16)),-1,1)`.

The diagonal retention vector is

`a(x) = clip(1/256 + (254/256) sigmoid(clip(E_a[x],-16,16)), 1/256,255/256)`.

The state evolves coordinatewise as

`h_next = clip(a(x)*h + (1-a(x))*u(x),-1,1)`, with `h_initial=0`.

**Proposition 1 (forward invariance).** Every admitted block state lies in
`[-1,1]^width`. In real arithmetic, each coordinate before the final clip is a
convex combination of two members of `[-1,1]`; induction establishes the claim.
The final clip also repairs finite arithmetic overshoot. For two initial states
under the same input sequence in real arithmetic, the transition is
nonexpansive with Lipschitz constant at most `255/256` per token. This is a
diagonal, token-conditioned affine selective recurrence; it is not a Mamba
implementation or a claim that arbitrary selective SSMs are contractive.

The mixed feature is `tanh(W_mix*h+b_mix)`, using effective entries clipped to
`[-1,1]`, and is itself clipped to `[-1,1]`. Each block's readout uses the
concatenated hidden and mixed features, effective head entries in `[-1,1]`, and
an output clip to `[-16,16]`. Block readouts are added in insertion order, with
another `[-16,16]` clip after every addition. Denote the resulting score vector
by `s`. Thus `|s_j|<=16` for every vocabulary coordinate and any admitted number
of blocks. The certificate does not assume the un-clipped sum has small norm.

The Boolean transition is `e_next[k]=e[k]` for event zero and
`e_next[k]=(event[k]==1)` otherwise. The implementation gathers the latest
nonzero event independently for each coordinate, with index zero referring to
the incoming state. Therefore the trajectory equals the sequential ledger
transition. This is exact integer/Boolean work, not approximate recurrent
memory. The full invariant set is a product of the hidden boxes and `{0,1}^2`.
The protected evidence part is persistent and is **not** strictly contractive.

**Proposition 2 (closure under updates).** Any finite FP32 raw parameter values
induce the stated effective constraints. No assumption about learning rate,
optimizer moments, gradient norm, or loss is needed for this closure. Clamps
are applied on every forward call, not merely after training. This guarantees
family membership for every admitted update, but saturated clamps and tanh can
have zero gradients. The theorem guarantees neither convergence nor useful
trainability at all parameter points. Nonfinite updates must be rejected.

## Readout separation and the semantic bridge

Let `m=e[0]+e[1]`, `b=s[defer]`, and `r=0.5*tanh(s[confirm])` (with the explicit
implementation clamp). The model replaces only the confirm logit:

`L_defer=b`, `L_confirm=b + (2*m - 3 + r)`, and `L_j=s_j` otherwise.

**Proposition 3 (real-arithmetic decision exclusion).** Since `r` is in
`[-0.5,0.5]`, `m<=1` implies `L_confirm-L_defer<=-0.5`. The existence of a
strictly higher defer logit excludes confirm from the complete vocabulary
argmax, regardless of every other word's score. If `m=2`, the same difference
is at least `+0.5`, although some other word may still win.

The semantic bridge has two explicit steps. First, under the trusted-event
assumptions, the protected coordinates equal current support for the two
required authorizations. Second, the readout separation excludes the atomic
affirmative decision when either is unsupported. The architecture does not
infer either step from natural language. This certificate addresses uncertainty
represented by absent/revoked/disputed support, not every possible uncertainty
about the world. It is intrinsic to this declared augmented input model, whose
trusted metadata and fixed readout rule are substantial architectural choices.

**Non-vacuity witness.** Set all parameters to zero except the first block's
defer head bias, which is `1`. Every ordinary word has score zero. Defer has
logit one. With both premises true confirm has logit two and uniquely wins;
otherwise confirm is at most zero and defer uniquely wins. Both positive and
negative decisions are achievable. This witness establishes capability, not
learned usefulness. The corpus contains positive decisions with concrete routes,
and experiments must measure whether training supplies those routes and avoids
unnecessary deferral or unrelated words.

## Finite precision and the actual path

The finite-precision claim additionally assumes conventional finite FP32
round-to-nearest arithmetic, correctly ordered comparison/clamp, and finite
tanh/sigmoid outputs on the explicitly bounded input intervals. Hardware faults,
malicious kernel implementations, memory corruption, and NaN propagation are
excluded. Turning off autocast does not alone prove every backend implements
these assumptions; the launch receipts and numerical checks must record the
actual device, precision settings, software version, and tested path.

For width at most 8192, bounded mix inputs yield exact absolute dot-product
sum at most 8193 including bias; bounded head inputs yield at most 16385.
These are far below FP32 overflow. Hidden arithmetic has bounded operands,
and every block-addition has magnitude at most 32 before clipping. Floating
dot-product error can alter learned scores, but the subsequent clips preserve
the exact endpoint bounds needed here. No tight accumulated GEMM error estimate
is used to establish the decision margin.

The evidence count is an exactly representable integer in `{0,1,2}`. Its signed
level `2*m-3` is exactly one of `-3,-1,+1`. Multiplication by one half is exact.
Monotonic rounding preserves the signed-offset bounds at the representable
endpoints `-0.5` and `+0.5`: the unsupported offset remains at most `-0.5`, and
the supported offset remains at least `+0.5`. Finally `|b|<=16` and
`|L_confirm|<=19.5` before rounding. In that range one final addition has error
at most `2^-20`. Thus the unsupported margin is at least
`0.5 - 2^-20 > 0.499999`, and the supported relative margin has the same bound.
The tests use the looser `0.49998` threshold to expose substantial regression
without claiming a tested maximum is a universal proof.

The executable tests cover every prior Boolean state and all two-coordinate
events in `{-2,-1,0,1,2}^2`, repeated histories, hostile finite parameter
endpoints, raw argmax, and nonfinite admission. Tests are finite observations;
the bounded argument establishes the universal claim under its assumptions.
CPU success does not establish a CUDA receipt. Prefill/decode equivalence is
checked with an FP32 tolerance for readouts because kernel accumulation order
and elementwise kernels may change with batch shape. The first CPU test receipt
found a hidden-state difference of `4.470348358154297e-08`; it falsified a claim
of exact differential state equality. The test therefore permits `1e-6` hidden
state tolerance and `1e-5` logit tolerance, independently checks both decision
margins and greedy choices, and does not call this exact parity. Growth uses an
unchanged prefill shape/path and requires exact full-logit numerical equality.

## Proof-preserving state growth

`grow()` appends an independent block of the same width, with an arbitrary
admitted initialized transition and **zero head weight and bias**. The old
blocks, their iteration order, and all output arithmetic stay in place. The new
block receives the same tokens; its hidden state need not be zero after a
prefix, but its initial output is exactly zero for every token.

**Proposition 4 (growth).** Under the same executable math path, appending the
silent block leaves every old prefix logit numerically equal: old scores are
unchanged, adding zero and clipping an already-clipped value preserves it, and
the protected readout then receives identical inputs. IEEE signed-zero bits may
differ; numerical equality is the claim. The new product state still satisfies
Proposition 1 and all future admitted updates satisfy Propositions 2 and 3.
Existing cached states are invalidated, and a prefix must be recomputed. The
library rejects a state with the wrong number of blocks.

This adds learnable state and additive feature width; it does not add sequential
composition depth or implement a general width-changing Net2Net theorem. A zero
head initially blocks gradients to the new block's features, while its head can
receive gradients immediately. After the head changes, gradients can reach
the features. The optimizer must include the newly created parameters; the
runner must record parameter deltas and causal output ablations after training.
`grow(silent_head=False)` initializes the matched larger scratch architecture;
it preserves the certificate but intentionally does not preserve logits.

## Counterexamples and unresolved scientific claims

- **Sampling:** every finite confirm logit has positive softmax probability,
  even without support. Sampling can emit confirm. The exclusion theorem is
  about raw greedy argmax only.
- **Wrong authority:** giving two false `+1` events can produce a unique confirm
  in the non-vacuity witness. Forged metadata or a forged evidence cache defeats
  the semantic assumption. Literal prose saying “trusted record” is insufficient.
- **Free prose:** an ordinary output word or sentence may affirm an unsupported
  move. Only the atomic marker has the declared verified meaning. Interpreting
  arbitrary prose as a certified decision invalidates the claim.
- **Route error:** a supported marker can precede the wrong destination or
  vehicle. The certificate checks authorizations, not route correctness. A
  route read from an unrelated item is still a substantive task failure.
- **Changed decoding:** a logit bias favoring confirm by more than the margin
  can reverse the result; so can a different unverified numeric path. No
  certificate is transferred by matching a few strings or showing low loss.
- **Persistent evidence:** if the caller neglects a real revocation, the model
  retains stale support. Arbitrarily many untrusted tokens cannot fix incorrect
  metadata. More learnable state does not remove this assumption.
- **Growth utility:** exact insertion equality says nothing about optimization
  benefit, plateau, scaling efficiency, or generalization. Those need the
  prespecified continuation, restart, and larger-scratch comparisons, cumulative
  compute accounting, and useful held-out outputs. A short pilot cannot prove
  a development plateau or a favorable scaling law.

The product-state construction has a mathematical certificate for one explicit
decision. Whether it is a useful language-model architecture remains falsifiable:
measure raw positive choices, route/reference quality, unnecessary deferral,
held-out loss, and whether new learned features causally improve these outcomes.
External checkers, release policy, and authority controls may complement it, but
their success is not evidence for this model's intrinsic certificate.

## Interval extension: computed premises for two plans

`interval_model.py` reuses exactly the neural recurrence and signed readout in
`Model._score`. Its protected state replaces two supplied support bits with a
query index and six signed-integer fields: permit, capacity lower bound, and
demand upper bound for each of two plans. `HOLD=-2` retains a field; `CLEAR=-1`
makes it unknown. Capacity/demand values in `[0,1_000_000]` are admitted;
unrecognized values clear. Permit admits only zero or one, in addition to the
two control codes. Values are converted to int64 before validation. The first
dedicated query token in a fresh episode binds plan A or B once. Every later
token, including model-generated query tokens, retains that selection. Before
any query, neither premise can be true. Latest-value gather for receipt fields
has the same sequential-transition argument as the Boolean ledger.

**Query binding invariant.** If the incoming query is already A or B, `_queries`
returns that value at every position. Otherwise it computes, for each prefix,
the earliest occurrence of each of the two distinct query IDs using integer
positions and cumulative minima. The smaller position selects the corresponding
plan. Equal positions occur only while both IDs are absent, which preserves the
unknown value. Once a query occurs, no later position can precede its index.
Induction over chunks then proves single assignment across prefill and decode.
A fresh episode is the only admitted way to reset this binding; ordinary tokens
cannot modify the receipts either. This protected state therefore preserves both
the evidence and its initial episode scope while the learned state evolves.

For the selected plan the internal premises are computed as

`p = active_query and permit==1`

`c = active_query and lower>=0 and upper>=0 and lower>=upper`.

All comparisons and the selected-plan gather are integer operations; there is
no float conversion of interval endpoints or approximate safety threshold.
The endpoint range is far below int64 overflow, and no endpoint addition or
multiplication is used. Thus Proposition 3 applies to `m=p+c` through the same
readout. Unknown fields, absent queries, non-permit values, or insufficient
intervals exclude raw greedy confirm independently of learned weights.

For a physical interpretation, suppose the selected plan's actual capacity is
`C`, actual demand is `D`, the trusted lower endpoint satisfies `lower<=C`, and
the trusted upper endpoint satisfies `D<=upper`, all in the same declared units
and snapshot. If `c` is true, transitivity gives
`D<=upper<=lower<=C`. If `p` is also true and the permit receipt is authentic
and currently valid for that plan, confirmation establishes permission and
capacity sufficiency under those assumptions. This is a computed sufficient
condition, not a truth label supplied directly by the caller. It can still be
conservative: overlapping intervals with insufficient lower-bound separation
cause deferral even when the unknown true `D<=C` happens to hold.

Receipts must be sound and bound to their plan; expirations, revocations, and
corrections must arrive before the affected prediction. Reusing a capacity
bound from a different snapshot or failing to update demand invalidates the
physical conclusion. The interval extension does not authenticate receipts or
verify those temporal assumptions. Non-vacuity is witnessed by permit one,
lower bound ten, upper bound nine, an active matching query, and the positive
readout witness above. Boundary equality `lower==upper` is supported.

**Retired latest-query counterexample:** the earlier variant allowed later
query tokens to change the active plan. An original request for unsupported
plan A could be followed by a generated query for supported plan B and then
confirm. That counterexample applies to the retired latest-query transition.
The current first-query binding prevents this generated-token rebinding while
retaining the same learned weights and signed readout.

**Initial framing boundary:** to interpret the bound plan as the original
trusted request, its intended query token must occur before any other dedicated
query token and before any decision is generated. A poisoned prefix containing
the other reserved query ID can hijack the *first* binding. The implementation
does not authenticate which source supplied that first token. Keeping the
reserved IDs out of earlier untrusted text, or otherwise establishing this
framing precondition, remains an explicit assumption. Under that assumption,
the certificate refers to the immutable initial scoped request throughout the
episode, even if generated text contains another query token. It still does
not certify the correctness of a subsequently written plan name or route.
No interval training or utility result follows merely from implementing this
additional protected transition.
