namespace FaultTolerantGrounding
structure Report where
  source : Nat
  epoch : Nat
  lower : Nat
  upper : Nat

def uniqueSources : List Report → Bool
  | [] => true
  | r :: rs => !(rs.any (fun s => s.source == r.source)) && uniqueSources rs

def faults (honest : Nat → Bool) : List Report → Nat
  | [] => 0
  | r :: rs => (if honest r.source then 0 else 1) + faults honest rs

theorem all_faults_length (honest : Nat → Bool) (rs : List Report)
    (ha : ∀ r, r ∈ rs → honest r.source = false) : faults honest rs = rs.length := by
  induction rs with
  | nil => rfl
  | cons r rs ih =>
    have hr := ha r (List.mem_cons.mpr (Or.inl rfl))
    have hrest := ih (fun s hs => ha s (List.mem_cons.mpr (Or.inr hs)))
    simp [faults,hr,hrest,Nat.add_comm]

def admissible (verify : Report → Bool) (epoch threshold : Nat) (r : Report) : Bool :=
  verify r && decide (r.epoch = epoch) && decide (threshold ≤ r.lower)

def accepted (verify : Report → Bool) (epoch threshold f : Nat) (rs : List Report) : Bool :=
  uniqueSources rs && rs.all (admissible verify epoch threshold) && decide (f < rs.length)

-- The signature verifier and world/calibration assumptions are supplied by the
-- protocol, not learned. A global bound over all distinct live reporting keys
-- should discharge faultBudget for every accepted proposal.
theorem accepted_threshold_sound (verify : Report → Bool) (honest : Nat → Bool)
    (epoch threshold f actual : Nat) (rs : List Report)
    (hv : accepted verify epoch threshold f rs = true)
    (faultBudget : faults honest rs ≤ f)
    (calibrated : ∀ r, verify r = true → r.epoch = epoch →
      honest r.source = true → r.lower ≤ actual ∧ actual ≤ r.upper) :
    threshold ≤ actual := by
  have htop := (Bool.and_eq_true_iff _ _).mp hv
  have hc : f < rs.length := of_decide_eq_true htop.2
  have hab := (Bool.and_eq_true_iff _ _).mp htop.1
  by_cases strong : threshold ≤ actual
  · exact strong
  · have ha : ∀ r, r ∈ rs → honest r.source = false := by
      intro r hr
      have hp := List.all_eq_true.mp hab.2 r hr
      have hp2 := (Bool.and_eq_true_iff _ _).mp hp
      have hp1 := (Bool.and_eq_true_iff _ _).mp hp2.1
      cases hh : honest r.source with
      | false => rfl
      | true =>
        have ht := Nat.le_trans (of_decide_eq_true hp2.2)
          (calibrated r hp1.1 (of_decide_eq_true hp1.2) hh).1
        exact False.elim (strong ht)
    have he := all_faults_length honest rs ha
    exact False.elim ((Nat.not_lt_of_le (he ▸ faultBudget)) hc)

def allSigned (_ : Report) := true
def packet : List Report := [⟨0,11,7,12⟩,⟨1,11,8,10⟩]
def duplicates : List Report := [⟨0,11,7,12⟩,⟨0,11,8,10⟩]

theorem cannot_amplify_one_source :
  accepted allSigned 11 6 1 packet = true ∧
  accepted allSigned 11 6 1 duplicates = false := by decide

theorem stale_cannot_amplify :
  accepted allSigned 12 6 1 packet = false := by decide

theorem one_fault_two_reports_indistinguishable :
    let reports : List Report := [⟨0,11,4,4⟩,⟨1,11,6,6⟩]
    let left := fun id : Nat => id == 0
    let right := fun id : Nat => id == 1
    faults left reports = 1 ∧ faults right reports = 1 ∧
    accepted allSigned 11 5 1 reports = false := by decide

#print axioms all_faults_length
#print axioms accepted_threshold_sound
#print axioms cannot_amplify_one_source
#print axioms stale_cannot_amplify
#print axioms one_fault_two_reports_indistinguishable

-- Current lower endpoint for a nonnegative quantity with a certified maximum
-- fall over the interval. Sign measurement time and the physical schema.
def adjustedLower (lo issued now rate : Nat) : Nat := lo - rate * (now-issued)
theorem lower_after_drift (lo issued now rate thenValue nowValue : Nat)
    (honestAtIssue : lo ≤ thenValue)
    (boundedLoss : thenValue ≤ nowValue + rate * (now-issued)) :
    adjustedLower lo issued now rate ≤ nowValue :=
  Nat.sub_le_of_le_add (Nat.le_trans honestAtIssue boundedLoss)

def conflicting : List Report := [⟨0,11,4,4⟩,⟨1,11,6,6⟩]
def honestLeft (id : Nat) := id == 0
def honestRight (id : Nat) := id == 1
def modelFits (actual : Nat) (honest : Nat → Bool) (rs : List Report) : Prop :=
  ∀ r, r ∈ rs → honest r.source = true → r.lower ≤ actual ∧ actual ≤ r.upper

theorem same_attestations_two_possible_actuals :
    faults honestLeft conflicting = 1 ∧ faults honestRight conflicting = 1 ∧
    modelFits 4 honestLeft conflicting ∧ modelFits 6 honestRight conflicting ∧
    ¬ 5 ≤ 4 ∧ 5 ≤ 6 := by
  refine ⟨rfl,rfl,?_,?_,by decide,by decide⟩
  · intro r hr hh
    have h := List.mem_cons.mp hr
    cases h with
    | inl eq => cases eq; decide
    | inr rest =>
      cases List.mem_cons.mp rest with
      | inl eq => cases eq; cases hh
      | inr e => cases e
  · intro r hr hh
    have h := List.mem_cons.mp hr
    cases h with
    | inl eq => cases eq; cases hh
    | inr rest =>
      cases List.mem_cons.mp rest with
      | inl eq => cases eq; decide
      | inr e => cases e

#print axioms lower_after_drift
#print axioms same_attestations_two_possible_actuals

end FaultTolerantGrounding
