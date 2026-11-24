# Imperium formal research: portable synthetic sources

Four isolated research packages, staged for a coordinator to review and place in an appropriate branch. No Git command, push, installation or external publication is performed by this bundle. Repository visibility and licensing remain coordinator decisions; staging itself grants no license.

## Reproduce safely

Use CPython 3.12 with the standard library; no dependencies need installation. From this directory:

```text
python -I -S -B verify.py hashes
python -I -S -B verify.py quantized_binder
python -I -S -B verify.py byte_query
python -I -S -B verify.py correspondence
python -I -S -B verify.py atomic_reference
# Or, one sequential run of all four:
python -I -S -B verify.py all
```

The entry point checks the exact MANIFEST.json file allowlist/hashes, starts a fresh interpreter per suite and rejects missing/unexpected files or test-count mismatches. Keep output redirection outside this directory. `-B` avoids bytecode caches. `.git` bookkeeping is ignored by the verifier if the coordinator later commits this folder. Do not invoke the reference neural snapshot files: they are source-inspection inputs only, have incomplete numerical dependencies here, and may import Torch if executed directly. The verified entry points never import them or Torch.

Each suite has a ten-second subprocess wait timeout; the four suites are sequential. This is not a hard OS CPU/memory/disk/descendant bound. Trusted small tests include one two-thread race, exact rational arithmetic, a tiny least-squares coefficient fit on declared synthetic examples, and finite truth/region grids. There is no broad-text training, neural forward/backward, checkpoint load, GPU, external effect, network, holdout or operational lease code.

## Packages and assurance boundaries

| Folder | Checks | What is supplied |
| --- | ---: | --- |
| quantized_binder | 20 | Learned affine coefficients fitted from supplied numeric preactivation examples; exact quantized recurrent execution; independent signed interval certificate for declared regions and update/growth checks. Numeric latent supervision and semantics are supplied, not discovered language meaning. |
| byte_query | 19 | Literal controlled-language byte events feed that learned recurrence. Separate trusted typed scope/source/epoch snapshots and fresh pending-output checks. Both referents have reachable licensed positives; revocation, UNKNOWN, cache/update/growth and spoofing negatives are retained. |
| correspondence | 7 | Pinned base/routed source syntax plus exact synthetic routed observer execution expose raw-candidate, reject-only AST and sink boundaries. No actual neural selector is run or certified. |
| atomic_reference | 13 | Separately reviewed later reference for current-state validation and atomic single consumption of exact issued intent bytes, in one protected in-memory host lineage. No actuator, crash durability or distributed exactly-once. |

The atomic_reference package is separately labeled for a later commit after validating the existing proof packages. It does not alter their source or retroactively strengthen their claims. Its conditional induction and material limits are in atomic_reference/CONTRACT.md; this is a source argument plus tests, not a kernel-checked universal theorem. Independent source review found and prompted a sticky-halt fix for event-cap exhaustion before qualification. The cap regression and useful commits for two distinct referents pass.

## Evidence and open obligations

MANIFEST.json enumerates every shipped file. PROVENANCE.json maps unchanged source/fixture bytes to source hashes and declares the one portable test adaptation and sanitized metadata/documents. VALIDATION.json reports this export's checks only; original machine-specific logs and raw execution receipts are deliberately absent.

The base qualification source file hash is `a1b41c310451617b613ff8dd606db8900f4e4ffb5f4d703f2fb2f0958185fe6f`; source-owner attestation associates it with commit `f1ccb1f5aec8a2199b28e9dcc29889826287f3c7` and tree `15505a99571df7fbe7d6fbd0a7fe34494ffb57bc`. The association was not independently Git-verified. Its typed A/B query latch is caller-token driven. The routed successor adds a learned request-to-route scope head. No proof connects that FP32 selector's reachable caches/readout to the tiny quantized binder. Source identity is resolved; semantic/native correspondence remains open.

The grammar never repairs an incorrect learned guess: both referents may be authorized while a wrong-route raw candidate is ALLOW; the pre-sink grammar rejects it as SCOPE_MISMATCH. The original sink returns a Boolean and can accept an unchanged intent repeatedly; restoring an old schema-valid snapshot can resurrect it. The later atomic reference addresses those omissions within one protected host, but its restart and split check/use counterexamples remain. Route epoch labels are distinct from evidence/grant revision counters.

Authority/evidence truth, authentication, timely delivery, common units, protected host state and serialized commit are explicit premises. Every parameter path that can affect selection needs a valid semantic certificate after update/growth; bounded states alone do not supply one. Ordinary generated text may be false or unsafe. No general alignment, safe English generation, intrinsic learned reasoning guarantee, practical scaling advantage or production execution guarantee is established. Newly accepted mechanized proofs: zero.
