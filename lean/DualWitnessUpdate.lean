namespace DualWitness
def score {W : Type} {m : Nat} (rows : Fin m → W → Int) (w : W) :
    List (Nat × Fin m) → Int
  | [] => 0
  | (weight,i) :: tail => Int.ofNat weight * rows i w + score rows w tail

def feasible {W : Type} {m : Nat} (rows : Fin m → W → Int) (w : W) :=
  ∀ i, rows i w ≤ 0

theorem weighted_nonpos {W : Type} {m : Nat} (rows : Fin m → W → Int) (w : W)
    (weights : List (Nat × Fin m)) (h : feasible rows w) : score rows w weights ≤ 0 := by
  induction weights with
  | nil => exact Int.le_refl 0
  | cons p rest ih =>
    change Int.ofNat p.1 * rows p.2 w + score rows w rest ≤ 0
    exact Int.add_le_add (Int.mul_nonpos_of_nonneg_of_nonpos (Int.ofNat_nonneg _) (h p.2)) ih

def lowerBound {W : Type} {m : Nat} (rows : Fin m → W → Int)
    (candidate : W → Prop) (weights : List (Nat × Fin m)) (lower : Int) :=
  ∀ w, candidate w → lower ≤ score rows w weights

theorem exclude_sound {W : Type} {m : Nat} (rows : Fin m → W → Int)
    (candidate : W → Prop) (weights : List (Nat × Fin m)) (lower : Int)
    (proof : lowerBound rows candidate weights lower) (strict : 0 < lower) :
    ∀ w, feasible rows w → ¬ candidate w := by
  intro w h hw
  have nd := Int.le_trans (proof w hw) (weighted_nonpos rows w weights h)
  exact Int.lt_irrefl 0 (Int.lt_of_lt_of_le strict nd)

-- Both hypotheses permit x in {-10,...,10}; y is the lane.
abbrev World := Fin 21 × Bool
def rows (desired : Bool) (i : Fin 2) (w : World) : Int :=
  let x := Int.ofNat w.1.val - 10
  let ly : Int := if w.2 = desired then 0 else 5
  (if i.val = 0 then x else -x) + ly - 2

def mix : List (Nat × Fin 2) := [(1,0),(1,1)]

theorem whole_box_opposite_lower : ∀ (d : Bool) (x : Fin 21),
    6 ≤ score (rows d) (x,!d) mix := by decide

theorem actual_witness : ∀ (d : Bool) (i : Fin 2), rows d i (10,d) ≤ 0 := by decide

theorem singleton_rows_not_enough : ∀ d : Bool,
    rows d 0 (7,!d) ≤ 0 ∧ rows d 1 (13,!d) ≤ 0 := by decide

theorem learned_mix_decides_correct_lane (d : Bool) (w : World)
    (h : feasible (rows d) w) : w.2 = d := by
  have bad := exclude_sound (rows d) (fun a => a.2 = !d) mix 6
    (fun a ha => by cases a with
                   | mk x y =>
                     change y = !d at ha
                     cases ha
                     exact whole_box_opposite_lower d x)
    (by decide) w h
  cases d <;> cases w with
  | mk x y => cases y <;> first | rfl | exact False.elim (bad rfl)

-- Any label proposal from outside this proof path is not a certified conclusion.
-- Removing either row leaves an opposite-lane witness, so cannot certify
-- the same required label under the singleton-row evidence.
theorem supersede_or_remove_left (d : Bool) :
    ∃ w : World, rows d 1 w ≤ 0 ∧ w.2 = !d :=
  ⟨(13,!d), (singleton_rows_not_enough d).2, rfl⟩
theorem supersede_or_remove_right (d : Bool) :
    ∃ w : World, rows d 0 w ≤ 0 ∧ w.2 = !d :=
  ⟨(7,!d), (singleton_rows_not_enough d).1, rfl⟩

#print axioms weighted_nonpos
#print axioms exclude_sound
#print axioms whole_box_opposite_lower
#print axioms actual_witness
#print axioms singleton_rows_not_enough
#print axioms learned_mix_decides_correct_lane
#print axioms supersede_or_remove_left
#print axioms supersede_or_remove_right

-- Raw optimizer output is arbitrary; only the end of the forward graph is constrained.
def protectedMixture (raw : Nat) : List (Nat × Fin 2) :=
  let a := 4 + raw % 3
  [(a,0),(10-a,1)]

theorem normalized_port_robust : ∀ (g : Fin 3) (d : Bool) (x : Fin 21),
    10 ≤ score (rows d) (x,!d) (protectedMixture g.val) := by decide

theorem protected_arbitrary_raw_update (raw : Nat) (d : Bool) (x : Fin 21) :
    10 ≤ score (rows d) (x,!d) (protectedMixture raw) := by
  have h := normalized_port_robust ⟨raw % 3, Nat.mod_lt _ (by decide)⟩ d x
  unfold protectedMixture at h ⊢
  rw [Nat.mod_mod] at h
  exact h

theorem protected_qualified_progress (raw : Nat) (d : Bool) (w : World)
    (h : feasible (rows d) w) : w.2 = d := by
  have bad := exclude_sound (rows d) (fun a => a.2 = !d)
    (protectedMixture raw) 10
    (fun a ha => by cases a with
                   | mk x y =>
                     change y = !d at ha
                     cases ha
                     exact protected_arbitrary_raw_update raw d x)
    (by decide) w h
  cases d <;> cases w with
  | mk x y => cases y <;> first | rfl | exact False.elim (bad rfl)

theorem raw_update_changes_witness_strength :
    score (rows false) (20,true) (protectedMixture 0) ≠
    score (rows false) (20,true) (protectedMixture 1) := by decide

-- Same evidence row, expanded hidden-domain/model. An exact old witness is no
-- evidence that a hypothesis remains impossible after migration.
theorem domain_widening_invalidates_old_witness :
    (∀ z : Int, z = 0 → 0 < 1-z) ∧ (∃ z : Int, 0 ≤ z ∧ z ≤ 1 ∧ 1-z ≤ 0) := by
  constructor
  · intro z hz
    rw [hz]
    decide
  · exact ⟨1, by decide, by decide, by decide⟩

#print axioms protected_arbitrary_raw_update
#print axioms protected_qualified_progress
#print axioms raw_update_changes_witness_strength
#print axioms domain_widening_invalidates_old_witness

end DualWitness
