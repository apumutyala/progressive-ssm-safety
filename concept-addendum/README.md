# Scoped concept extension: standalone source addendum

This is a separate addendum to the bounded proof release. It does not change that release or its manifest. The single Lean source includes its prerequisites and the tiny four-row synthetic recurrent fixture. It requires Lean 4.8.0, no Mathlib, model checkpoint, data download, or GPU.

```text
lean --version
lean ScopedConceptBound.lean
```

The file contains executable definitions with kernel-checked assertions. Look for these final declarations:

- `calculation_never_invents`: a non-deferred Boolean result has the declared formula denotation when primitive fitted reads agree with the stipulated world.
- `install_never_writes_ledger`: installing a concept cannot modify the grant ledger.
- `evaluated_under_current_footprint`: a result uses the selected current formula/revision, actual fitted reads, and grants for **all** canonical leaves, not only branches that determine the value.
- `old_protocol_unchanged_by_real_install`: the exact legacy branch does not depend on concept registration.
- `strict_new_semantic_coverage` and `derived_value_not_either_atomic_projection`: a newly admitted concept request returns a multi-fact result on two modeled histories, and its truth function cannot be reproduced by selecting either old single fact.
- `unauthorized_substitution_does_not_install`, `bound_exact_proposal_mutation_rejected`, `conflicts_and_version_deny`, and `cannot_install_unaddressable`: mismatch, wrong version, collisions, and invalid symbols fail their controls.
- `every_symbol_has_exact_wire`: admitted concept symbols have a bounded exact wire encoding.

The theorem assumes that a trusted host supplies the **exact separately authorized declaration**, complete current definitions, and trustworthy evidence semantics. It does not implement authentication, prove arbitrary prose intent, provide timing noninterference, enact external effects, or show that a learned proposer produces desirable concepts. A multi-fact query is more expressive than either old one-fact query in one call, not new hidden information: an authorized client could have combined several old reads. No new fact permission follows from installing a concept.

`MANIFEST.SHA256` identifies source bytes; line endings are fixed to LF. Do not include generated build artifacts in this addendum.
