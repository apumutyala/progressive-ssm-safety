# Bounded alignment theory: executable proofs and synthetic falsifiers

This source-only reference separates three questions: learned recurrent computation, exact byte-to-fact correspondence, and authority to release a result. Its claims are restricted to named typed models. It is not a proof about a general pretrained language model, arbitrary prose, truth of real-world attestations, external actions, or a GPU runtime.

## Reproduce

Tested with Lean 4.8.0 (revision `df668f00e6c0`), Python 3.12, and NumPy 2.5.2 on CPU. No Mathlib or downloaded benchmark data are required for the Lean files. With an existing Lean executable and the Python requirements available:

```text
python scripts/reproduce.py --lean lean
python scripts/reproduce.py --only lean --summary-only
python scripts/reproduce.py --only cpu --seed 910231
```

The driver verifies all standalone Lean files, audits the printed top-level axioms for `sorryAx`, and appends the demo tail to the strict framing proof in a temporary directory. It fits and tests the CPU exercises in a temporary copy. It does not download dependencies or persist checkpoints or reports in this source tree. Use a different `--seed` to generate new synthetic evaluation data. This release regenerates fresh instances; it does not reproduce old held-out result numbers. Resource guards and wall-clock times are host-dependent.

For an individual proof, run `lean lean/<file>.lean`. The Lean sources are complete standalone proof snapshots and deliberately share earlier definitions; do not import them all into one namespace. `demo/FramedDemoTail.lean` is an append-only executable tail for the strict snapshot, not a standalone module. A successful demonstration prints two distinct controlled factual records for the same request under changed admitted evidence, and defers on the adverse controls.

## Proof index

| Source | What it establishes | What it does not establish |
|---|---|---|
| `lean/FittedObserverConformance.lean` | Checks the four synthetic fitted affine rows in a finite signed recurrence; relates all local state/event/error rows to a CPU reference table and proves arbitrary finite joint trace correctness under stipulated events. | Floating-point LM conformance, learned authenticity, general language semantics. |
| `lean/StructuralSSMUnifiedRouted.lean` | Embeds that fitted recurrence in one model with projected parameter updates, routed bounded feedback, width and depth growth. Stable old observations keep their required answer; opposite new evidence changes composed replies and can change old numeric wires without changing their required output. | Unrestricted optimizer writes, runtime response deadlines, arbitrary architecture wiring or real durability. |
| `lean/ExactFramedFittedStrict.lean` | Only eight declared raw canonical request frames may get a factual reply. Whole-token parsing, current scoped directory and revision selection, the fitted recurrent state, and factual serialization form one runnable reference. The emitted grammar round-trips the exact approved record and is not accepted as a request. | Completeness/truth of supplied authoritative records, transport authentication, delivery of external effects, or unrestricted-prose truth. |
| `lean/IngestionSparseCommit.lean` | A current, complete grounded index checks a positive evidence tree and a backward cut for every rival, including unresolved; the checked effect and current grant are bound in one modeled transition. | A production indexed/parser speedup, trusted real-world ingestion, or effect execution. |
| `lean/FaultTolerantDrift.lean` | With stated honest calibration and cumulative fault budget, sufficiently many distinct current reports certify a threshold; bounded drift can safely weaken a reading, and identical reports can leave threshold ambiguity. | Proof that IDs are independent failure domains, real calibration, authenticity or freshness protocol. |
| `lean/DualWitnessUpdate.lean` | A structurally nonnegative weighted witness cannot exclude a feasible world; a protected raw-output projection preserves a declared positive class. Shows why old cached evidence can fail after domain widening. | That learned search improves solver latency, or that arbitrary nonlinear world membership is efficiently decidable. |

Printed generic proofs depend only on the standard Lean axioms reported by the kernel, or none. Some inference properties are positive-class or stable-path statements, not guarantees against arbitrary evidence changes. Explicit attack controls cover malformed framing, zero-width/NUL tails, wrong token roles, unrelated prose, distinct addresses with equal values, tied current revisions, stale frontiers, and duplicate/reordered evidence cases in the relevant fixtures.

The only stored numerical model input is `cpu/minimal_observer_coefficients.json`: four tiny synthetic affine rows and their typed token meanings. No generated checkpoints, evaluation instances, reports, logs, credentials, tools, or non-fixture weights are included.

## CPU falsifiers

- **Fitted observer:** reruns fitting on small synthetic trajectories and checks projected interventions, invalidation and composition. It compares the frozen old rail and an exact state-machine control. A fit test is not a proof of unbounded history; the finite row certificate plus induction is.
- **Affine witnesses:** compares a learned sparse ranker, equal-search-budget geometric heuristics, exhaustive quantized sparse search, and cold/warm exact rational feasibility in two dimensions. The independent vertex-enumeration check validates the clipping oracle on regenerated synthetic cases. A bound on sparse quantized representation is not a bound on all rational witnesses.
- **Nested rival proofs:** times exact indexed propagation, exact backward search and a deterministic greedy witness cache on a generated monotone AND/OR graph. Checks scoped duplicate evidence, stale epochs, and retraction before delivery. Asynchronous controls are separate from timed steady runs. The greedy cache is not a fitted model.
- **Language-to-address:** a small float bidirectional RNN proposes token pointers while the factual value comes from the fitted numeric bank. The frozen audit examines training-disjoint synthetic surfaces, unsupported forms, and counterfactuals. The declared exact grammar is a reference, not an equal-data learner. Known failure mode: a plausible model draft is not evidence of intended meaning. The numeric fitted register and the float query parser are distinct artifacts; the float parser is not Lean-certified.

`research/BoundedLanguageModelProtocol.md` specifies the next pretraining experiment. It has not been executed as a full pretrained-model result in this release. The successful bounded path retains sentence parsing as an explicit typed authority boundary, and asks learned computation to contribute evidence and raw candidates without repairing them at the sink.
