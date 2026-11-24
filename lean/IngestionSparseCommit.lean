namespace Ingestion
inductive Rule where
  | fact (out : Nat)
  | unary (p out : Nat)
  | both (p q out : Nat)
deriving DecidableEq

def Oblig (chart : Nat → Bool) : Rule → Prop
  | .fact o => chart o = true
  | .unary p o => chart p = true → chart o = true
  | .both p q o => chart p = true → chart q = true → chart o = true

instance (chart : Nat → Bool) (r : Rule) : Decidable (Oblig chart r) := by
  cases r with
  | fact o => exact inferInstanceAs (Decidable (chart o = true))
  | unary p o => exact inferInstanceAs (Decidable (chart p = true → chart o = true))
  | both p q o => exact inferInstanceAs (Decidable (chart p = true → chart q = true → chart o = true))

def check (rules : List Rule) (chart : Nat → Bool) : Bool :=
  rules.all (fun r => decide (Oblig chart r))

inductive Derives (rs : List Rule) : Nat → Prop where
  | fact {o} : Rule.fact o ∈ rs → Derives rs o
  | unary {p o} : Rule.unary p o ∈ rs → Derives rs p → Derives rs o
  | both {p q o} : Rule.both p q o ∈ rs → Derives rs p → Derives rs q → Derives rs o

theorem checked_covers_derivation (rs : List Rule) (chart : Nat → Bool)
    (accepted : check rs chart = true) (out : Nat) (hd : Derives rs out) : chart out = true := by
  have hh := List.all_eq_true.mp accepted
  induction hd with
  | fact mem => exact of_decide_eq_true (hh _ mem)
  | unary mem _ ih => exact (of_decide_eq_true (hh _ mem)) ih
  | both mem _ _ ihp ihq => exact (of_decide_eq_true (hh _ mem)) ihp ihq

-- 0 = unresolved/non-effect, 1 and 2 = two distinct exact effect identities.
-- These are domain values, not actual external documents/contacts.
def unanimous (chart : Nat → Bool) : Option Nat :=
  if chart 0 then none else
  if chart 1 && !chart 2 then some 1 else
  if chart 2 && !chart 1 then some 2 else none

theorem unanimity_bits : ∀ a b c : Bool,
    let chart := fun n : Nat => if n = 0 then a else if n = 1 then b else c
    (∀ o : Fin 3, ∀ r : Fin 3, unanimous chart = some r.val →
        chart o.val = true → o.val = r.val) := by decide

theorem selected_sound (rs : List Rule) (chart : Nat → Bool) (out r : Fin 3)
    (accepted : check rs chart = true) (dr : Derives rs out.val)
    (result : unanimous chart = some r.val) : out.val = r.val := by
  have h := checked_covers_derivation rs chart accepted out.val dr
  have hb := unanimity_bits (chart 0) (chart 1) (chart 2) out r
  have ox : out.val = 0 ∨ out.val = 1 ∨ out.val = 2 := by
    have ho : ∀ x : Fin 3, x.val = 0 ∨ x.val = 1 ∨ x.val = 2 := by decide
    exact ho out
  have h2 : (if out.val = 0 then chart 0 else if out.val = 1 then chart 1 else chart 2) = chart out.val := by
    cases ox with
    | inl zz => rw [zz]; rfl
    | inr hh => cases hh with
      | inl oo => rw [oo]; rfl
      | inr tt => rw [tt]; rfl
  exact hb result (h2.trans h)

-- Tiny controlled byte request: ASCII fwd:L:L:P:Q (forward/latest/large/picked/QA).
-- The parser tests all bytes and EOF and provides no scope, identity or grant.
def request : List Nat := [102,119,100,58,76,58,76,58,80,58,81]
def parses (bytes : List Nat) : Bool := decide (bytes = request)
def allCodes : Bool → List Nat
  | false => [4,5,6,7,8]
  | true => [11,12,13,14,15]
structure Context where
  candidates : List Bool
  facts : List Nat
  snap : Nat
  grantA : Bool
  grantB : Bool

def ready (c : Context) (candidate : Bool) : Bool :=
  (allCodes candidate).all (fun f => c.facts.contains f)
def proofPath (b : Bool) : List Rule :=
  if b then
    [.fact 11, .fact 12, .fact 13, .fact 14, .fact 15,
     .both 11 12 21, .both 13 14 22, .both 21 22 23, .both 23 15 2]
  else
    [.fact 4, .fact 5, .fact 6, .fact 7, .fact 8,
     .both 4 5 16, .both 6 7 17, .both 16 17 18, .both 18 8 1]
def groundingRules (bytes : List Nat) (c : Context) : List Rule :=
  if !parses bytes || c.candidates.isEmpty then [.fact 0] else
    List.join (c.candidates.map (fun b => if !ready c b then [.fact 0] else proofPath b))

-- Protected sink only accepts the exact current snapshot and the typed effect ID.
def redeem (now : Context) (stamp effect : Nat) : Bool :=
  if stamp != now.snap then false else
  if effect == 1 then now.grantA else if effect == 2 then now.grantB else false

theorem stale_never_redeems (now : Context) (stamp effect : Nat) (h : stamp ≠ now.snap) :
    redeem now stamp effect = false := by
  simp [redeem, h]

def A : Context := ⟨[false], [4,5,6,7,8], 11, true, false⟩
def B : Context := ⟨[true], [11,12,13,14,15], 12, false, true⟩
def conflict : Context := ⟨[false,true], [4,5,6,7,8,11,12,13,14,15], 13, true, true⟩
def missing : Context := ⟨[false,true], [4,5,6,7,8], 14, true, true⟩
def chartA (n : Nat) : Bool := ([1,4,5,6,7,8,16,17,18] : List Nat).contains n
def chartB (n : Nat) : Bool := ([2,11,12,13,14,15,21,22,23] : List Nat).contains n
def chartUnresolved (n : Nat) : Bool := n == 0 || chartA n

theorem same_bytes_distinct_current_effects :
    check (groundingRules request A) chartA = true ∧ unanimous chartA = some 1 ∧
    redeem A 11 1 = true ∧
    check (groundingRules request B) chartB = true ∧ unanimous chartB = some 2 ∧
    redeem B 12 2 = true := by decide

theorem omitted_ambiguous_parse_fails :
    check (groundingRules request conflict) chartA = false := by decide
theorem omitting_unresolved_candidate_fails :
    check (groundingRules request missing) chartA = false := by decide
theorem keeping_unresolved_candidate_prevents_effect :
    check (groundingRules request missing) chartUnresolved = true ∧
    unanimous chartUnresolved = none := by decide
theorem injected_trailing_byte_fails :
    check (groundingRules (request ++ [0]) A) chartA = false := by decide
theorem forged_effect_stale_epoch_fails :
    redeem B 11 1 = false ∧ redeem B 12 1 = false := by decide

#print axioms checked_covers_derivation
#print axioms selected_sound
#print axioms stale_never_redeems
#print axioms same_bytes_distinct_current_effects
#print axioms omitted_ambiguous_parse_fails
#print axioms omitting_unresolved_candidate_fails
#print axioms keeping_unresolved_candidate_prevents_effect
#print axioms injected_trailing_byte_fails
#print axioms forged_effect_stale_epoch_fails

-- One modeled effect transition. The bytes and chart used to select the effect are
-- the same values checked here; only the trusted current context supplies grants.
-- The real sink still needs an implementation that linearizes on that context.
def commit (bytes : List Nat) (now : Context) (stamp : Nat) (chart : Nat → Bool) : Option Nat :=
  if check (groundingRules bytes now) chart then
    match unanimous chart with
    | none => none
    | some a => if redeem now stamp a then some a else none
  else none

theorem committed_exact_checked_effect (bytes : List Nat) (now : Context)
    (stamp : Nat) (chart : Nat → Bool) (a : Nat)
    (hc : commit bytes now stamp chart = some a) :
    check (groundingRules bytes now) chart = true ∧
    unanimous chart = some a ∧ redeem now stamp a = true := by
  unfold commit at hc
  cases he : check (groundingRules bytes now) chart with
  | false => simp [he] at hc
  | true =>
    simp only [he, if_true] at hc
    cases hu : unanimous chart with
    | none => simp [hu] at hc
    | some b =>
      simp only [hu] at hc
      cases hr : redeem now stamp b with
      | false => simp [hr] at hc
      | true =>
        have hb : b = a := Option.some.inj (by simpa [hr] using hc)
        cases hb
        exact ⟨rfl,rfl,hr⟩

theorem joint_ingress_commit_sound (bytes : List Nat) (now : Context)
    (stamp : Nat) (chart : Nat → Bool) (may actual : Fin 3)
    (hc : commit bytes now stamp chart = some may.val)
    (actualDerives : Derives (groundingRules bytes now) actual.val) :
    actual.val = may.val ∧ redeem now stamp may.val = true := by
  have h := committed_exact_checked_effect bytes now stamp chart may.val hc
  exact ⟨selected_sound _ _ actual may h.1 actualDerives h.2.1, h.2.2⟩

theorem matching_grant_wrong_request_cannot_commit :
    redeem A 11 1 = true ∧ commit (request ++ [0]) A 11 chartA = none := by decide

theorem all_current_effect_controls :
    commit request A 11 chartA = some 1 ∧ commit request B 12 chartB = some 2 ∧
    commit request conflict 13 chartA = none ∧ commit request missing 14 chartUnresolved = none ∧
    commit request A 10 chartA = none := by decide

#print axioms committed_exact_checked_effect
#print axioms joint_ingress_commit_sound
#print axioms matching_grant_wrong_request_cannot_commit
#print axioms all_current_effect_controls


def head : Rule → Nat
  | .fact o => o
  | .unary _ o => o
  | .both _ _ o => o

def blocked (excluded : List Nat) : Rule → Prop
  | .fact _ => False
  | .unary p _ => p ∈ excluded
  | .both p q _ => p ∈ excluded ∨ q ∈ excluded

instance (excluded : List Nat) (r : Rule) : Decidable (blocked excluded r) := by
  cases r with
  | fact _ => exact inferInstanceAs (Decidable False)
  | unary p _ => exact inferInstanceAs (Decidable (p ∈ excluded))
  | both p q _ => exact inferInstanceAs (Decidable (p ∈ excluded ∨ q ∈ excluded))

def CompleteIndex (rules : List Rule) (idx : Nat → List Rule) : Prop :=
  ∀ r, r ∈ rules → r ∈ idx (head r)

def checkCut (idx : Nat → List Rule) (ex : List Nat) : Bool :=
  ex.all (fun q => (idx q).all (fun r => decide (blocked ex r)))

theorem verified_backward_cut (rules : List Rule) (idx : Nat → List Rule)
    (ex : List Nat) (hi : CompleteIndex rules idx) (hc : checkCut idx ex = true)
    (o : Nat) (hd : Derives rules o) : o ∉ ex := by
  have hh := List.all_eq_true.mp hc
  induction hd with
  | @fact o hm =>
    intro he
    have hr := List.all_eq_true.mp (hh _ he) _ (hi (.fact o) hm)
    exact of_decide_eq_true hr
  | @unary p o hm _ ih =>
    intro he
    have hr := List.all_eq_true.mp (hh _ he) _ (hi (.unary p o) hm)
    exact ih (of_decide_eq_true hr)
  | @both p q o hm _ _ ihp ihq =>
    intro he
    have hr := List.all_eq_true.mp (hh _ he) _ (hi (.both p q o) hm)
    have h : p ∈ ex ∨ q ∈ ex := of_decide_eq_true hr
    cases h with
    | inl hp => exact ihp hp
    | inr hq => exact ihq hq

-- A learnable proposal may return any cut. In addition to excluding every other
-- reference outcome (including unresolved), prove a positive derivation exists.
theorem positive_and_negative_witnesses (rules : List Rule) (idx : Nat → List Rule)
    (ex : List Nat) (hi : CompleteIndex rules idx) (hc : checkCut idx ex = true)
    (actual selected : Fin 3)
    (_positive : Derives rules selected.val)
    (allCompetitorsCut : ∀ r : Fin 3, r.val ≠ selected.val → r.val ∈ ex)
    (actualDerives : Derives rules actual.val) : actual.val = selected.val := by
  by_cases same : actual.val = selected.val
  · exact same
  · exact False.elim (verified_backward_cut rules idx ex hi hc _ actualDerives
      (allCompetitorsCut actual same))

theorem lower_bounds_sound (x y l r t : Nat)
    (hx : l ≤ x) (hy : r ≤ y) (ht : t ≤ l+r) : t ≤ x+y :=
  Nat.le_trans ht (Nat.add_le_add hx hy)

#print axioms verified_backward_cut
#print axioms positive_and_negative_witnesses
#print axioms lower_bounds_sound


inductive EvidenceTree where
  | leaf (o : Nat)
  | unary (child : EvidenceTree) (o : Nat)
  | both (left right : EvidenceTree) (o : Nat)

def verifyTree (idx : Nat → List Rule) : EvidenceTree → Option Nat
  | .leaf o => if decide (Rule.fact o ∈ idx o) then some o else none
  | .unary t o =>
    match verifyTree idx t with
    | none => none
    | some p => if decide (Rule.unary p o ∈ idx o) then some o else none
  | .both t u o =>
    match verifyTree idx t, verifyTree idx u with
    | some p, some q => if decide (Rule.both p q o ∈ idx o) then some o else none
    | _, _ => none

structure TrustedIndex (rs : List Rule) where
  idx : Nat → List Rule
  complete : CompleteIndex rs idx
  valid : ∀ (h : Nat) (r : Rule), r ∈ idx h → r ∈ rs

def prepareReferenceIndex (rs : List Rule) : TrustedIndex rs where
  idx := fun h => rs.filter (fun r => head r == h)
  complete := by
    intro r hm
    apply List.mem_filter.mpr
    exact ⟨hm, by simp⟩
  valid := by
    intro h r hm
    exact (List.mem_filter.mp hm).1

theorem accepted_tree_derives (rs : List Rule) (idx : TrustedIndex rs) (t : EvidenceTree) (a : Nat)
    (hc : verifyTree idx.idx t = some a) : Derives rs a := by
  induction t generalizing a with
  | leaf o =>
    unfold verifyTree at hc
    split at hc
    · next hh =>
      have eq : o = a := Option.some.inj hc
      cases eq
      exact Derives.fact (idx.valid o _ (of_decide_eq_true hh))
    · next hh => cases hc
  | unary t o ih =>
    unfold verifyTree at hc
    cases h : verifyTree idx.idx t with
    | none => simp [h] at hc
    | some p =>
      simp only [h] at hc
      split at hc
      · next hh =>
        have eq : o = a := Option.some.inj hc
        cases eq
        exact Derives.unary (idx.valid o _ (of_decide_eq_true hh)) (ih p h)
      · next hh => cases hc
  | both t u o it iu =>
    unfold verifyTree at hc
    cases ht : verifyTree idx.idx t with
    | none => simp [ht] at hc
    | some p =>
      cases hu : verifyTree idx.idx u with
      | none => simp [ht, hu] at hc
      | some q =>
        simp only [ht, hu] at hc
        split at hc
        · next hh =>
          have eq : o = a := Option.some.inj hc
          cases eq
          exact Derives.both (idx.valid o _ (of_decide_eq_true hh)) (it p ht) (iu q hu)
        · next hh => cases hc

def allRivals (a : Nat) (cut : List Nat) : Bool :=
  ([0,1,2] : List Nat).all (fun r => r == a || cut.contains r)

theorem rivals_verified : ∀ (a : Fin 3) (b : Fin 3) (c0 c1 c2 : Bool),
  let membership := fun n => if n = 0 then c0 else if n = 1 then c1 else c2
  (([0,1,2] : List Nat).all (fun r => r == a.val || membership r)) = true →
  b.val ≠ a.val → membership b.val = true := by decide

def commitSparse (bytes : List Nat) (now : Context) (stamp : Nat)
    (idx : TrustedIndex (groundingRules bytes now)) (tree : EvidenceTree) (cut : List Nat) : Option Nat :=
  match verifyTree idx.idx tree with
  | none => none
  | some a =>
    if checkCut idx.idx cut && allRivals a cut && redeem now stamp a then some a else none

def certA : EvidenceTree :=
  .both (.both (.both (.leaf 4) (.leaf 5) 16) (.both (.leaf 6) (.leaf 7) 17) 18) (.leaf 8) 1
def certB : EvidenceTree :=
  .both (.both (.both (.leaf 11) (.leaf 12) 21) (.both (.leaf 13) (.leaf 14) 22) 23) (.leaf 15) 2

def exactSparse (bytes : List Nat) (now : Context) (stamp : Nat)
    (tree : EvidenceTree) (cut : List Nat) : Option Nat :=
  commitSparse bytes now stamp (prepareReferenceIndex (groundingRules bytes now)) tree cut

theorem sparse_current_effect_controls :
    exactSparse request A 11 certA [0,2] = some 1 ∧
    exactSparse request B 12 certB [0,1] = some 2 ∧
    exactSparse request conflict 13 certA [0,2] = none ∧
    exactSparse request missing 14 certA [0,2] = none ∧
    exactSparse (request ++ [0]) A 11 certA [0,2] = none ∧
    exactSparse request A 10 certA [0,2] = none := by decide

-- Same candidate cut: valid at old snapshot, invalid when rival is enabled.
theorem stale_head_index_counterexample :
    checkCut (prepareReferenceIndex (groundingRules request A)).idx [0,2] = true ∧
    checkCut (prepareReferenceIndex (groundingRules request conflict)).idx [0,2] = false := by decide

#print axioms accepted_tree_derives
#print axioms rivals_verified
#print axioms sparse_current_effect_controls
#print axioms stale_head_index_counterexample


theorem every_rival_mem (a : Nat) (cut : List Nat) (h : allRivals a cut = true)
    (b : Fin 3) (neq : b.val ≠ a) : b.val ∈ cut := by
  have ib : ∀ b : Fin 3, b.val ∈ ([0,1,2] : List Nat) := by decide
  have h0 := List.all_eq_true.mp h b.val (ib b)
  cases (Bool.or_eq_true_iff _ _).mp h0 with
  | inl bad => exact False.elim (neq ((beq_iff_eq _ _).mp bad))
  | inr ok => exact List.mem_of_elem_eq_true ok

theorem sparse_commit_sound (bytes : List Nat) (now : Context) (stamp : Nat)
    (idx : TrustedIndex (groundingRules bytes now)) (tree : EvidenceTree) (cut : List Nat)
    (chosen : Nat) (actual : Fin 3)
    (h : commitSparse bytes now stamp idx tree cut = some chosen)
    (actualDerives : Derives (groundingRules bytes now) actual.val) :
    actual.val = chosen ∧ redeem now stamp chosen = true ∧
      Derives (groundingRules bytes now) chosen := by
  unfold commitSparse at h
  cases ht : verifyTree idx.idx tree with
  | none => simp [ht] at h
  | some got =>
    simp only [ht] at h
    split at h
    · next tests =>
      have same : got = chosen := Option.some.inj h
      cases same
      have h2 := (Bool.and_eq_true_iff _ _).mp tests
      have h1 := (Bool.and_eq_true_iff _ _).mp h2.1
      have unique : actual.val = chosen := by
        by_cases eq : actual.val = chosen
        · exact eq
        · have cutmem := every_rival_mem chosen cut h1.2 actual eq
          exact False.elim (verified_backward_cut _ _ _ idx.complete h1.1 _ actualDerives cutmem)
      exact ⟨unique,h2.2,accepted_tree_derives _ idx tree chosen ht⟩
    · next tests => cases h

#print axioms every_rival_mem
#print axioms sparse_commit_sound

end Ingestion
