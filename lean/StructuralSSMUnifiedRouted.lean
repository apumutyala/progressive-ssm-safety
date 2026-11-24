namespace StrictGrowth
abbrev Cell := Fin 7
abbrev Mode := Fin 3
abbrev Op := Fin 4
abbrev Gain := Fin 2
abbrev Noise := Fin 3

def inside (q : Mode) (x : Cell) : Prop :=
  if q.val = 0 then 2 ≤ x.val ∧ x.val ≤ 4
  else if q.val = 1 then x.val ≤ 1 else 5 ≤ x.val
instance (q : Mode) (x : Cell) : Decidable (inside q x) := inferInstanceAs
  (Decidable (if q.val = 0 then 2 ≤ x.val ∧ x.val ≤ 4 else if q.val = 1 then x.val ≤ 1 else 5 ≤ x.val))

def center (q : Mode) : Cell := if q.val = 0 then 3 else if q.val = 1 then 0 else 6
def modeStep (q : Mode) (e : Op) : Mode :=
  if e.val = 0 then q else if e.val = 1 then
    if q.val = 0 then 0 else if q.val = 1 then 2 else 1
  else if e.val = 2 then 1 else 2

def sat (x : Int) : Int := max (-3) (min 3 x)
def shrink (x : Int) : Int := if x ≥ 0 then max 0 (x - 5) else min 0 (x + 5)
def signed (x : Cell) : Int := Int.ofNat x.val - 3
def enc (x : Int) : Cell := ⟨(sat x + 3).toNat % 7, Nat.mod_lt _ (by decide)⟩
def transition (g : Gain) (d : Noise) (x : Cell) (e : Op) : Cell :=
  let a := 4 + Int.ofNat g.val
  let u := if e.val = 0 then sat (shrink (a * signed x))
    else if e.val = 1 then sat (shrink (-(a * signed x)))
    else if e.val = 2 then -3 else 3
  let v := if u > 0 then u + Int.ofNat g.val else if u < 0 then u - Int.ofNat g.val else u
  enc (v + Int.ofNat d.val - 1)

def decode (x : Cell) : Option Bool :=
  if x.val ≤ 1 then some false else if 5 ≤ x.val then some true else none
def meaning (q : Mode) : Option Bool :=
  if q.val = 0 then none else if q.val = 1 then some false else some true

theorem init_inside : inside 0 3 := by decide
theorem rows : ∀ (g : Gain) (d : Noise) (q : Mode) (x : Cell) (e : Op),
    inside q x → inside (modeStep q e) (transition g d x e) := by decide
theorem readout : ∀ (q : Mode) (x : Cell), inside q x → decode x = meaning q := by decide

structure Tick where
  g : Gain
  d : Noise
  e : Op

def run (x : Cell) : List Tick → Cell
  | [] => x
  | t :: es => run (transition t.g t.d x t.e) es
def shadow (q : Mode) : List Tick → Mode
  | [] => q
  | t :: es => shadow (modeStep q t.e) es

theorem run_refines (es : List Tick) (q : Mode) (x : Cell) (h : inside q x) :
    inside (shadow q es) (run x es) := by
  induction es generalizing q x with
  | nil => exact h
  | cons t _ ih => exact ih _ _ (rows t.g t.d q x t.e h)

abbrev World := Bool × Bool × Bool
def lhs (w : World) := Bool.xor w.1 w.2.1
def rhs (w : World) := Bool.xor w.2.1 w.2.2
def lane (w : World) := Bool.xor w.1 w.2.2

theorem join_semantics : ∀ a b c : Bool,
    Bool.xor (lhs (a,b,c)) (rhs (a,b,c)) = lane (a,b,c) := by decide

def compatible (q : Option Bool) (b : Bool) : Prop :=
  q = none ∨ q = some b

def oldView (l : Option Bool) (w : World) : Prop := compatible l (lhs w)
def newView (l r : Option Bool) (w : World) : Prop :=
  oldView l w ∧ compatible r (rhs w)

def commits (view : World → Prop) (b : Bool) : Prop :=
  (∃ w, view w) ∧ ∀ w, view w → lane w = b

def both (x y : Cell) : Option Bool :=
  match decode x, decode y with
  | some a, some b => some (Bool.xor a b)
  | _, _ => none

theorem old_never_commits (a b : Bool) : ¬ commits (oldView (some a)) b := by
  intro h
  have hf : oldView (some a) (false,a,false) := Or.inr (by cases a <;> rfl)
  have ht : oldView (some a) (false,a,true) := Or.inr (by cases a <;> rfl)
  have f : false = b := h.2 _ hf
  have t : true = b := h.2 _ ht
  cases (f.trans t.symm)

theorem new_necessarily_commits (a b : Bool) :
    commits (newView (some a) (some b)) (Bool.xor a b) := by
  constructor
  · refine ⟨(false,a,Bool.xor a b), Or.inr ?_, Or.inr ?_⟩
    · cases a <;> rfl
    · cases a <;> cases b <;> rfl
  · intro w hw
    have la : a = lhs w := by
      cases hw.1 with
      | inl bad => cases bad
      | inr good => exact Option.some.inj good
    have ra : b = rhs w := by
      cases hw.2 with
      | inl bad => cases bad
      | inr good => exact Option.some.inj good
    rw [la, ra]
    exact (join_semantics w.1 w.2.1 w.2.2).symm

theorem decoded_correct_after_arbitrary_traces (le re : List Tick) (a b : Bool)
    (hl : meaning (shadow 0 le) = some a) (hr : meaning (shadow 0 re) = some b) :
    both (run 3 le) (run 3 re) = some (Bool.xor a b) ∧
    commits (newView (some a) (some b)) (Bool.xor a b) := by
  constructor
  · unfold both
    rw [readout _ _ (run_refines le 0 3 init_inside), readout _ _ (run_refines re 0 3 init_inside), hl, hr]
  · exact new_necessarily_commits a b

def tick (e : Op) : Tick := ⟨0,1,e⟩
def leftHistory : List Tick := [tick 2, tick 0]
def rightHistory0 : List Tick := [tick 2, tick 0]
def rightHistory1 : List Tick := [tick 2, tick 1]

theorem strict_growth_same_old_state :
    meaning (shadow 0 leftHistory) = some false ∧
    (∀ b : Bool, ¬ commits (oldView (some false)) b) ∧
    both (run 3 leftHistory) (run 3 rightHistory0) = some false ∧
    both (run 3 leftHistory) (run 3 rightHistory1) = some true ∧
    commits (newView (some false) (some false)) false ∧
    commits (newView (some false) (some true)) true := by
  exact ⟨rfl, old_never_commits false, rfl, rfl,
    new_necessarily_commits false false, new_necessarily_commits false true⟩

theorem reject_low_gain_mutation :
    ¬ inside (modeStep 2 0) (enc (sat (shrink (2 * signed 5)) - 1)) := by decide
theorem reject_dropped_flip_mutation :
    ¬ inside (modeStep 1 1) (transition 0 1 0 0) := by decide
theorem dropped_new_view_fails_positive_case :
    both (run 3 leftHistory) 3 ≠ some true := by decide

#print axioms rows
#print axioms run_refines
#print axioms join_semantics
#print axioms old_never_commits
#print axioms new_necessarily_commits
#print axioms decoded_correct_after_arbitrary_traces
#print axioms strict_growth_same_old_state
#print axioms reject_low_gain_mutation
#print axioms reject_dropped_flip_mutation
#print axioms dropped_new_view_fails_positive_case

abbrev Event := Fin 9
def actualStep (w : World) (e : Event) : World :=
  if e.val = 1 then (!w.1,w.2.1,w.2.2)
  else if e.val = 2 then (w.1,!w.2.1,w.2.2)
  else if e.val = 3 then (w.1,w.2.1,!w.2.2)
  else w
def attested (w : World) (e : Event) : Prop :=
  if e.val = 4 then lhs w = false else if e.val = 5 then lhs w = true
  else if e.val = 6 then rhs w = false else if e.val = 7 then rhs w = true else True
instance (w : World) (e : Event) : Decidable (attested w e) := inferInstanceAs
  (Decidable (if e.val = 4 then lhs w = false else if e.val = 5 then lhs w = true
  else if e.val = 6 then rhs w = false else if e.val = 7 then rhs w = true else True))
def leftOp (e : Event) : Op :=
  if e.val = 1 ∨ e.val = 2 then 1 else if e.val = 4 then 2 else if e.val = 5 then 3 else 0
def rightOp (e : Event) : Op :=
  if e.val = 2 ∨ e.val = 3 then 1 else if e.val = 6 then 2 else if e.val = 7 then 3 else 0
def rightMode (q : Mode) (e : Event) : Mode := if e.val = 8 then 0 else modeStep q (rightOp e)
def rightStep (g : Gain) (d : Noise) (x : Cell) (e : Event) : Cell :=
  if e.val = 8 then 3 else transition g d x (rightOp e)
instance (q : Option Bool) (b : Bool) : Decidable (compatible q b) := inferInstanceAs
  (Decidable (q = none ∨ q = some b))

theorem left_event_preserves : ∀ (a b c : Bool) (q : Mode) (e : Event),
    attested (a,b,c) e → compatible (meaning q) (lhs (a,b,c)) →
    compatible (meaning (modeStep q (leftOp e))) (lhs (actualStep (a,b,c) e)) := by decide
theorem right_event_preserves : ∀ (a b c : Bool) (q : Mode) (e : Event),
    attested (a,b,c) e → compatible (meaning q) (rhs (a,b,c)) →
    compatible (meaning (rightMode q e)) (rhs (actualStep (a,b,c) e)) := by decide
theorem right_rows : ∀ (g : Gain) (d : Noise) (q : Mode) (x : Cell) (e : Event),
    inside q x → inside (rightMode q e) (rightStep g d x e) := by decide

structure JointTick where
  e : Event
  gl : Gain
  gr : Gain
  dl : Noise
  dr : Noise

def jointRun (xy : Cell × Cell) : List JointTick → Cell × Cell
  | [] => xy
  | t :: ts => jointRun (transition t.gl t.dl xy.1 (leftOp t.e), rightStep t.gr t.dr xy.2 t.e) ts
def jointShadow (qq : Mode × Mode) : List JointTick → Mode × Mode
  | [] => qq
  | t :: ts => jointShadow (modeStep qq.1 (leftOp t.e), rightMode qq.2 t.e) ts
def worldRun (w : World) : List JointTick → World
  | [] => w
  | t :: ts => worldRun (actualStep w t.e) ts
def legalTrace (w : World) : List JointTick → Prop
  | [] => True
  | t :: ts => attested w t.e ∧ legalTrace (actualStep w t.e) ts

def jointInv (w : World) (xy : Cell × Cell) (qq : Mode × Mode) : Prop :=
  inside qq.1 xy.1 ∧ inside qq.2 xy.2 ∧ newView (meaning qq.1) (meaning qq.2) w

theorem jointInit (w : World) : jointInv w (3,3) (0,0) :=
  ⟨init_inside, init_inside, Or.inl rfl, Or.inl rfl⟩

theorem joint_closed (tr : List JointTick) (w : World) (xy : Cell × Cell) (qq : Mode × Mode)
    (hl : legalTrace w tr) (hi : jointInv w xy qq) :
    jointInv (worldRun w tr) (jointRun xy tr) (jointShadow qq tr) := by
  induction tr generalizing w xy qq with
  | nil => exact hi
  | cons t _ ih =>
    exact ih _ _ _ hl.2
      ⟨rows t.gl t.dl qq.1 xy.1 (leftOp t.e) hi.1,
       right_rows t.gr t.dr qq.2 xy.2 t.e hi.2.1,
       left_event_preserves w.1 w.2.1 w.2.2 qq.1 t.e hl.1 hi.2.2.1,
       right_event_preserves w.1 w.2.1 w.2.2 qq.2 t.e hl.1 hi.2.2.2⟩

def answer (tr : List JointTick) : Option Bool := both (jointRun (3,3) tr).1 (jointRun (3,3) tr).2
def expected (tr : List JointTick) (a b : Bool) : Prop :=
  meaning (jointShadow (0,0) tr).1 = some a ∧ meaning (jointShadow (0,0) tr).2 = some b

theorem joint_correct_commit (tr : List JointTick) (w : World) (a b : Bool)
    (hl : legalTrace w tr) (hh : expected tr a b) :
    answer tr = some (Bool.xor a b) ∧ lane (worldRun w tr) = Bool.xor a b := by
  have hi := joint_closed tr w (3,3) (0,0) hl (jointInit w)
  have hw := hi.2.2
  unfold expected at hh
  rw [hh.1, hh.2] at hw
  constructor
  · unfold answer both
    rw [readout _ _ hi.1, readout _ _ hi.2.1, hh.1, hh.2]
  · exact (new_necessarily_commits a b).2 _ hw

def jt (e : Event) : JointTick := ⟨e,0,0,1,1⟩
def hist0 : List JointTick := [jt 4, jt 6, jt 0]
def hist1 : List JointTick := [jt 4, jt 6, jt 3]
def oldSummary (tr : List JointTick) : Cell := (jointRun (3,3) tr).1

theorem joint_strict : legalTrace (false,false,false) hist0 ∧
    legalTrace (false,false,false) hist1 ∧ oldSummary hist0 = oldSummary hist1 ∧
    answer hist0 = some false ∧ answer hist1 = some true ∧
    lane (worldRun (false,false,false) hist0) = false ∧
    lane (worldRun (false,false,false) hist1) = true := by
  have h0 := joint_correct_commit hist0 (false,false,false) false false
    ⟨rfl, rfl, True.intro, True.intro⟩ ⟨rfl,rfl⟩
  have h1 := joint_correct_commit hist1 (false,false,false) false true
    ⟨rfl, rfl, True.intro, True.intro⟩ ⟨rfl,rfl⟩
  exact ⟨⟨rfl, rfl, True.intro, True.intro⟩, ⟨rfl, rfl, True.intro, True.intro⟩,
    rfl, h0.1, h1.1, h0.2, h1.2⟩

-- Event 8 invalidates the right observation without revealing a replacement.
theorem invalidated_right_abstains (tr : List JointTick) (xy : Cell × Cell) :
    both (jointRun xy (tr ++ [jt 8])).1 (jointRun xy (tr ++ [jt 8])).2 = none := by
  induction tr generalizing xy with
  | nil =>
    change both (transition 0 1 xy.1 (leftOp 8)) 3 = none
    unfold both
    cases decode (transition 0 1 xy.1 (leftOp 8)) <;> rfl
  | cons t ts ih => exact ih _

theorem migration_without_retained_evidence_cannot_distinguish
    (recover : Cell → Option Bool) :
    ¬ (recover (oldSummary hist0) = some false ∧ recover (oldSummary hist1) = some true) := by
  intro h
  have same : oldSummary hist0 = oldSummary hist1 := rfl
  rw [same] at h
  have bad : some false = some true := h.1.symm.trans h.2
  cases bad

#print axioms left_event_preserves
#print axioms right_event_preserves
#print axioms right_rows
#print axioms joint_closed
#print axioms joint_correct_commit
#print axioms joint_strict
#print axioms invalidated_right_abstains
#print axioms migration_without_retained_evidence_cannot_distinguish


theorem right_only_never_commits (b c : Bool) :
    ¬ commits (fun w => compatible (some b) (rhs w)) c := by
  intro h
  have hf : compatible (some b) (rhs (false,b,false)) := Or.inr (by cases b <;> rfl)
  have ht : compatible (some b) (rhs (true,b,false)) := Or.inr (by cases b <;> rfl)
  have f : false = c := h.2 _ hf
  have t : true = c := h.2 _ ht
  cases (f.trans t.symm)

theorem gains_actually_change_forward : transition 0 0 6 0 ≠ transition 1 0 6 0 := by decide

def oldRun (x : Cell) : List JointTick → Cell
  | [] => x
  | t :: ts => oldRun (transition t.gl t.dl x (leftOp t.e)) ts

theorem old_projection (tr : List JointTick) (x y : Cell) :
    oldRun x tr = (jointRun (x,y) tr).1 := by
  induction tr generalizing x y with
  | nil => rfl
  | cons t _ ih => exact ih _ _

def grow (x : Cell) : Cell × Cell := (x,3)

theorem live_install_refines (w : World) (x : Cell) (q : Mode)
    (h : inside q x) (hw : oldView (meaning q) w) :
    jointInv w (grow x) (q,0) :=
  ⟨h, init_inside, hw, Or.inl rfl⟩

def advanceAfterInstall (tr : List JointTick) : Option Bool :=
  both (jointRun (grow (oldRun 3 [jt 4])) tr).1 (jointRun (grow (oldRun 3 [jt 4])) tr).2

theorem live_growth_new_evidence_strict :
    both (grow (oldRun 3 [jt 4])).1 (grow (oldRun 3 [jt 4])).2 = none ∧
    oldRun 3 hist0 = oldRun 3 hist1 ∧
    advanceAfterInstall [jt 6,jt 0] = some false ∧
    advanceAfterInstall [jt 6,jt 3] = some true ∧
    lane (worldRun (false,false,false) hist0) = false ∧
    lane (worldRun (false,false,false) hist1) = true :=
  ⟨rfl,rfl,rfl,rfl,joint_strict.2.2.2.2.2.1,joint_strict.2.2.2.2.2.2⟩

theorem false_attestation_not_admitted : ¬ attested (false,false,false) 7 := by decide

theorem no_vacuous_commit {v : World → Prop} (h : ¬ ∃ w, v w) (b : Bool) :
    ¬ commits v b := fun good => h good.1

theorem fixed_snapshot_extension_preserves (v f : World → Prop) (w : World) (b : Bool)
    (old : commits v b) (hw : v w ∧ f w) :
    commits (fun candidate => v candidate ∧ f candidate) b :=
  ⟨⟨w, hw⟩, fun candidate hc => old.2 candidate hc.1⟩

#print axioms right_only_never_commits
#print axioms gains_actually_change_forward
#print axioms old_projection
#print axioms live_install_refines
#print axioms live_growth_new_evidence_strict
#print axioms false_attestation_not_admitted
#print axioms no_vacuous_commit
#print axioms fixed_snapshot_extension_preserves

end StrictGrowth


namespace Fitted
open StrictGrowth

-- Four small synthetic fitted coefficients, also listed in cpu/minimal_observer_coefficients.json.
def coeff (e : Fin 5) : Int × Int :=
  if e.val = 0 then (-4,0)
  else if e.val = 1 then (0,8)
  else if e.val = 2 then (4,0)
  else if e.val = 3 then (0,-8)
  else (0,0)

def step (x : Cell) (e : Fin 5) (d : Noise) : Cell :=
  if e.val = 4 then enc (Int.ofNat d.val - 1)
  else
    let ab := coeff e
    enc (sat (shrink (ab.1 * signed x + ab.2)) + Int.ofNat d.val - 1)

def encodeEvent (e : Event) : Fin 5 :=
  if e.val = 8 then 4 else
    if (rightOp e).val = 0 then 2 else
    if (rightOp e).val = 1 then 0 else
    if (rightOp e).val = 2 then 3 else 1

def specToken (q : Mode) (e : Fin 5) : Mode :=
  if e.val = 4 then 0 else
    if e.val = 0 then modeStep q 1 else
    if e.val = 1 then modeStep q 3 else
    if e.val = 2 then modeStep q 0 else modeStep q 2

theorem fitted_rows : ∀ (q : Mode) (x : Cell) (e : Fin 5) (d : Noise),
    inside q x → inside (specToken q e) (step x e d) := by decide
theorem token_projection_correct : ∀ (q : Mode) (e : Event),
    specToken q (encodeEvent e) = rightMode q e := by decide

def run (xy : Cell × Cell) : List JointTick → Cell × Cell
  | [] => xy
  | t :: ts => run
    (transition t.gl t.dl xy.1 (leftOp t.e), step xy.2 (encodeEvent t.e) t.dr) ts

theorem fitted_closed (tr : List JointTick) (w : World) (xy : Cell × Cell) (qq : Mode × Mode)
    (hl : legalTrace w tr) (hi : jointInv w xy qq) :
    jointInv (worldRun w tr) (run xy tr) (jointShadow qq tr) := by
  induction tr generalizing w xy qq with
  | nil => exact hi
  | cons t _ ih =>
    have hr := fitted_rows qq.2 xy.2 (encodeEvent t.e) t.dr hi.2.1
    rw [token_projection_correct] at hr
    exact ih _ _ _ hl.2
      ⟨rows t.gl t.dl qq.1 xy.1 (leftOp t.e) hi.1, hr,
       left_event_preserves w.1 w.2.1 w.2.2 qq.1 t.e hl.1 hi.2.2.1,
       right_event_preserves w.1 w.2.1 w.2.2 qq.2 t.e hl.1 hi.2.2.2⟩

def answer (xy : Cell × Cell) (tr : List JointTick) : Option Bool :=
  both (run xy tr).1 (run xy tr).2

theorem fitted_commits_correct (tr : List JointTick) (w : World) (xy : Cell × Cell)
    (qq : Mode × Mode) (a b : Bool)
    (hl : legalTrace w tr) (hi : jointInv w xy qq)
    (hlq : meaning (jointShadow qq tr).1 = some a)
    (hrq : meaning (jointShadow qq tr).2 = some b) :
    answer xy tr = some (Bool.xor a b) ∧ lane (worldRun w tr) = Bool.xor a b := by
  have inv := fitted_closed tr w xy qq hl hi
  have hw := inv.2.2
  rw [hlq, hrq] at hw
  constructor
  · unfold answer both
    rw [readout _ _ inv.1, readout _ _ inv.2.1, hlq, hrq]
  · exact (new_necessarily_commits a b).2 _ hw

theorem preserves_old_computation (tr : List JointTick) (x y : Cell) :
    (run (x,y) tr).1 = oldRun x tr := by
  induction tr generalizing x y with
  | nil => rfl
  | cons t _ ih => exact ih _ _

theorem both_fitted_strict (b : Bool) :
    answer (grow (oldRun 3 [jt 4])) [jt 6,jt 0] = some false ∧
    answer (grow (oldRun 3 [jt 4])) [jt 6,jt 3] = some true ∧
    oldRun 3 hist0 = oldRun 3 hist1 ∧
    ¬ commits (oldView (some false)) b :=
  ⟨rfl,rfl,rfl,old_never_commits false b⟩

theorem broken_flip_mutation_rejected :
    ¬ inside (specToken 2 0) (step 6 2 1) := by decide

theorem wrong_attest_token_mutation_rejected :
    ¬ inside (specToken 0 1) (step 3 3 1) := by decide

#print axioms fitted_rows
#print axioms token_projection_correct
#print axioms fitted_closed
#print axioms fitted_commits_correct
#print axioms preserves_old_computation
#print axioms both_fitted_strict
#print axioms broken_flip_mutation_rejected
#print axioms wrong_attest_token_mutation_rejected
end Fitted


namespace StructuralSSM
open StrictGrowth

abbrev Wide := Fin 13
abbrev Dist := Fin 5
def num (x : Wide) : Int := Int.ofNat x.val - 6
def encW (x : Int) : Wide :=
  ⟨(max (-6) (min 6 x) + 6).toNat % 13, Nat.mod_lt _ (by decide)⟩
def mag (p : Int) := if p ≥ 0 then max 0 (p-10) else min 0 (p+10)
def satW (p : Int) := max (-6) (min 6 p)

-- These are learned effective parameter selectors. The deployment maps every
-- raw optimizer output to Fin 2 ports with bounded, immutable operators.
structure Raw where
  gain : Nat
  excursion : Nat
  writeGain : Nat

def bit (raw : Nat) : Fin 2 := ⟨raw % 2, Nat.mod_lt _ (by decide)⟩

def kernel (g j w : Fin 2) (x : Wide) (e : Fin 5) (eta : Dist) : Wide :=
  let gain := 4 + Int.ofNat g.val
  let wx := Int.ofNat w.val
  let inside :=
    if e.val = 0 then -(gain * num x)
    else if e.val = 1 then wx * num x + 16 + 6 * wx
    else if e.val = 2 then gain * num x
    else if e.val = 3 then wx * num x - 16 - 6 * wx
    else 0
  let u := satW (mag inside)
  let v := if u > 0 then u + Int.ofNat j.val else if u < 0 then u - Int.ofNat j.val else 0
  encW (v + Int.ofNat eta.val - 2)

def deployed (raw : Raw) (x : Wide) (e : Fin 5) (eta : Dist) : Wide :=
  kernel (bit raw.gain) (bit raw.excursion) (bit raw.writeGain) x e eta

def cell (q : Mode) (x : Wide) : Prop :=
  if q.val = 0 then 4 ≤ x.val ∧ x.val ≤ 8
  else if q.val = 1 then x.val ≤ 2 else 10 ≤ x.val
instance (q : Mode) (x : Wide) : Decidable (cell q x) := inferInstanceAs
  (Decidable (if q.val = 0 then 4 ≤ x.val ∧ x.val ≤ 8 else if q.val = 1 then x.val ≤ 2 else 10 ≤ x.val))

def projected (x : Wide) : Option Bool :=
  if x.val ≤ 2 then some false else if 10 ≤ x.val then some true else none

theorem inductive_rows :
    ∀ (g j w : Fin 2) (x : Wide) (q : Mode) (e : Fin 5) (eta : Dist),
      cell q x → cell (Fitted.specToken q e) (kernel g j w x e eta) := by decide
theorem projected_cell : ∀ (q : Mode) (x : Wide), cell q x → projected x = meaning q := by decide
theorem unknown_at_center : cell 0 6 := by decide

theorem deployed_rows (raw : Raw) (x : Wide) (q : Mode) (e : Fin 5) (eta : Dist)
    (h : cell q x) : cell (Fitted.specToken q e) (deployed raw x e eta) :=
  inductive_rows _ _ _ _ _ _ _ h

def lift (x : Cell) : Wide := encW (2 * signed x)
def liftDist (d : Noise) : Dist := ⟨(2*d.val) % 5,Nat.mod_lt _ (by decide)⟩
theorem exactly_contains_fitted_baseline :
    ∀ (x : Cell) (e : Fin 5) (d : Noise),
      deployed ⟨0,0,0⟩ (lift x) e (liftDist d) = lift (Fitted.step x e d) := by decide
theorem obs_agrees_fitted : ∀ x : Cell, projected (lift x) = decode x := by decide

def combinedDist (physical structural : Noise) : Dist :=
  ⟨(physical.val + structural.val) % 5, Nat.mod_lt _ (by decide)⟩
theorem two_disturbances_represented (p a : Noise) :
    (combinedDist p a).val = p.val + a.val := by
  have h : ∀ (p a : Noise), p.val + a.val < 5 := by decide
  exact Nat.mod_eq_of_lt (h p a)

-- Decoder/modelling stages observe old/current states; there is no reference
-- semantic-Mode input to kernel/deployed/projected.
structure Bank where
  n : Nat
  x : Fin n → Wide

structure Ghost (bank : Bank) where
  semantic : Fin bank.n → Mode

def closed (bank : Bank) (ref : Ghost bank) : Prop :=
  ∀ i, cell (ref.semantic i) (bank.x i)

def infer (bank : Bank) (raw : Fin bank.n → Raw)
    (evt : Fin bank.n → Fin 5) (noise : Fin bank.n → Dist) : Bank :=
  ⟨bank.n, fun i => deployed (raw i) (bank.x i) (evt i) (noise i)⟩
def refInfer (bank : Bank) (ref : Ghost bank) (raw : Fin bank.n → Raw)
    (evt : Fin bank.n → Fin 5) (noise : Fin bank.n → Dist) :
      Ghost (infer bank raw evt noise) :=
  ⟨fun i => Fitted.specToken (ref.semantic i) (evt i)⟩

theorem forward_and_update_closed (bank : Bank) (ref : Ghost bank)
    (raw : Fin bank.n → Raw) (evt : Fin bank.n → Fin 5) (noise : Fin bank.n → Dist)
    (good : closed bank ref) :
    closed (infer bank raw evt noise) (refInfer bank ref raw evt noise) :=
  fun i => deployed_rows _ _ _ _ _ (good i)

-- Width growth preserves all old coordinate state exactly at install. Further
-- inference may let new layers influence old coordinates through bounded Dist.
def widen (b : Bank) : Bank := ⟨b.n+1, Fin.lastCases 6 b.x⟩
def widenRef (b : Bank) (g : Ghost b) : Ghost (widen b) :=
  ⟨Fin.lastCases 0 g.semantic⟩
theorem widen_closed (b : Bank) (g : Ghost b) (h : closed b g) :
    closed (widen b) (widenRef b g) := by
  intro i
  refine Fin.lastCases ?_ (fun k => ?_) i
  · change cell (Fin.lastCases (motive := fun _ => Mode) 0 g.semantic (Fin.last b.n))
      (Fin.lastCases (motive := fun _ => Wide) 6 b.x (Fin.last b.n))
    rw [Fin.lastCases_last, Fin.lastCases_last]
    exact unknown_at_center
  · change cell (Fin.lastCases (motive := fun _ => Mode) 0 g.semantic k.castSucc)
      (Fin.lastCases (motive := fun _ => Wide) 6 b.x k.castSucc)
    rw [Fin.lastCases_castSucc, Fin.lastCases_castSucc]
    exact h k

theorem widen_old_wire (b : Bank) (i : Fin b.n) :
    (widen b).x i.castSucc = b.x i := Fin.lastCases_castSucc i

structure Config where
  bank : Bank
  ghost : Ghost bank

def good (c : Config) : Prop := closed c.bank c.ghost
def more (c : Config) : Config := ⟨widen c.bank, widenRef c.bank c.ghost⟩
def step (c : Config) (raw : Fin c.bank.n → Raw)
    (evt : Fin c.bank.n → Fin 5) (eta : Fin c.bank.n → Dist) : Config :=
  ⟨infer c.bank raw evt eta,refInfer c.bank c.ghost raw evt eta⟩

inductive Trace (initial : Config) : Config → Prop where
  | start : Trace initial initial
  | updateAndInfer {c : Config} : Trace initial c →
      (raw : Fin c.bank.n → Raw) → (evt : Fin c.bank.n → Fin 5) →
      (noise : Fin c.bank.n → Dist) → Trace initial (step c raw evt noise)
  | addWidth {c : Config} : Trace initial c → Trace initial (more c)

theorem arbitrary_updates_and_widths {c f : Config} (tr : Trace c f) (h : good c) : good f := by
  induction tr with
  | start => exact h
  | updateAndInfer _ raw evt noise ih => exact forward_and_update_closed _ _ raw evt noise ih
  | addWidth _ ih => exact widen_closed _ _ ih

theorem mandatory_progress_after_mixed_trace {initial final : Config} (tr : Trace initial final)
    (hg : good initial) (i : Fin final.bank.n) (b : Bool)
    (winning : meaning (final.ghost.semantic i) = some b) :
    projected (final.bank.x i) = some b :=
  (projected_cell _ _ (arbitrary_updates_and_widths tr hg i)).trans winning

-- One explicit repeated optimizer/feedback suffix can be arbitrary. Revoke
-- is a lawful intervention, not off-manifold zeroing.
structure Hold where
  raw : Raw
  eta : Dist
def runHold (x : Wide) : List Hold → Wide
  | [] => x
  | t :: ts => runHold (deployed t.raw x 2 t.eta) ts
theorem hold_modes : ∀ q : Mode, Fitted.specToken q 2 = q := by decide
theorem hold_preserves_any_optimizer (ts : List Hold) (q : Mode) (x : Wide) (h : cell q x) :
    cell q (runHold x ts) := by
  induction ts generalizing x with
  | nil => exact h
  | cons t _ ih =>
    have h1 := deployed_rows t.raw x q 2 t.eta h
    rw [hold_modes] at h1
    exact ih _ h1

theorem paired_intervention_requires_opposite_outputs (ts : List Hold)
    (raw : Raw) (eta : Dist) (x : Wide) (q : Mode) (h : cell q x) :
    projected (runHold (deployed raw x 3 eta) ts) = some false ∧
    projected (runHold (deployed raw x 1 eta) ts) = some true ∧
    projected (runHold (deployed raw x 4 eta) ts) = none := by
  have hl := deployed_rows raw x q 3 eta h
  have hr := deployed_rows raw x q 1 eta h
  have hz := deployed_rows raw x q 4 eta h
  have ha : ∀ q : Mode, Fitted.specToken q 3 = 1 ∧ Fitted.specToken q 1 = 2 ∧
       Fitted.specToken q 4 = 0 := by decide
  rw [(ha q).1] at hl
  rw [(ha q).2.1] at hr
  rw [(ha q).2.2] at hz
  exact ⟨projected_cell _ _ (hold_preserves_any_optimizer ts _ _ hl),
         projected_cell _ _ (hold_preserves_any_optimizer ts _ _ hr),
         projected_cell _ _ (hold_preserves_any_optimizer ts _ _ hz)⟩

theorem nondecorative_protected_wires :
    deployed ⟨0,0,0⟩ 12 2 (combinedDist 0 0) ≠
      deployed ⟨0,1,0⟩ 12 2 (combinedDist 0 0) ∧
    deployed ⟨0,0,0⟩ 12 2 (combinedDist 0 0) ≠
      deployed ⟨0,0,0⟩ 12 2 (combinedDist 0 2) := by decide

#print axioms inductive_rows
#print axioms exactly_contains_fitted_baseline
#print axioms obs_agrees_fitted
#print axioms two_disturbances_represented
#print axioms forward_and_update_closed
#print axioms widen_closed
#print axioms widen_old_wire
#print axioms arbitrary_updates_and_widths
#print axioms mandatory_progress_after_mixed_trace
#print axioms paired_intervention_requires_opposite_outputs
#print axioms nondecorative_protected_wires
end StructuralSSM


namespace StructuralSSM
open StrictGrowth

-- Each added internal depth block is a typed hold on the protected wire.
-- Intermediate readouts are not published; the deepStep outs are emitted
-- only when the scheduled transition finishes.
def deepStep (before after : List Hold) (raw : Raw) (e : Fin 5) (eta : Dist) (x : Wide) : Wide :=
  runHold (deployed raw (runHold x before) e eta) after

theorem arbitrary_depth_and_raw_closed (before after : List Hold) (raw : Raw)
    (e : Fin 5) (eta : Dist) (q : Mode) (x : Wide) (h : cell q x) :
    cell (Fitted.specToken q e) (deepStep before after raw e eta x) :=
  hold_preserves_any_optimizer after _ _
    (deployed_rows raw _ q e eta (hold_preserves_any_optimizer before q x h))

theorem macro_positive (before after : List Hold) (raw : Raw)
    (e : Fin 5) (eta : Dist) (q : Mode) (x : Wide) (b : Bool)
    (h : cell q x) (wins : meaning (Fitted.specToken q e) = some b) :
    projected (deepStep before after raw e eta x) = some b :=
  (projected_cell _ _ (arbitrary_depth_and_raw_closed before after raw e eta q x h)).trans wins

-- An old coordinate is tracked through insertions and arbitrarily many
-- updates/observed events on other coordinates. Only its own event must hold.
inductive InertPath (start : Config) (old : Fin start.bank.n) :
    (c : Config) → Fin c.bank.n → Prop where
  | nil : InertPath start old start old
  | tick {c : Config} {i : Fin c.bank.n} : InertPath start old c i →
      (raw : Fin c.bank.n → Raw) → (evt : Fin c.bank.n → Fin 5) →
      (eta : Fin c.bank.n → Dist) → evt i = 2 →
      InertPath start old (step c raw evt eta) i
  | newRail {c : Config} {i : Fin c.bank.n} : InertPath start old c i →
      InertPath start old (more c) i.castSucc

theorem old_meaning_preserved_on_inert_path {c f : Config} {old : Fin c.bank.n} {now : Fin f.bank.n}
    (p : InertPath c old f now) : f.ghost.semantic now = c.ghost.semantic old := by
  induction p with
  | nil => rfl
  | @tick c i _ _ evt _ held ih =>
    change Fitted.specToken (c.ghost.semantic i) (evt i) = _
    rw [held,hold_modes]
    exact ih
  | @newRail c i _ ih =>
    change (Fin.lastCases (motive := fun _ => Mode) 0 c.ghost.semantic i.castSucc) = _
    rw [Fin.lastCases_castSucc]
    exact ih

theorem inert_path_trace {c f : Config} {old : Fin c.bank.n} {now : Fin f.bank.n}
    (p : InertPath c old f now) : Trace c f := by
  induction p with
  | nil => exact .start
  | tick _ raw evt eta _ ih => exact .updateAndInfer ih raw evt eta
  | newRail _ ih => exact .addWidth ih

theorem old_positive_coverage_through_repeated_width_and_updates {c f : Config}
    {old : Fin c.bank.n} {now : Fin f.bank.n} (p : InertPath c old f now)
    (h : good c) (b : Bool) (oldWinning : meaning (c.ghost.semantic old) = some b) :
    projected (f.bank.x now) = some b := by
  apply mandatory_progress_after_mixed_trace (inert_path_trace p) h now b
  rw [old_meaning_preserved_on_inert_path p]
  exact oldWinning

theorem active_depth_not_erased :
    let r : Raw := ⟨0,0,0⟩
    let eta : Dist := 0
    deepStep [] [] r 2 4 12 ≠ deepStep [] [⟨r,eta⟩] r 2 4 12 ∧
    projected (deepStep [] [] r 2 4 12) = some true ∧
    projected (deepStep [] [⟨r,eta⟩] r 2 4 12) = some true := by decide

#print axioms arbitrary_depth_and_raw_closed
#print axioms macro_positive
#print axioms old_meaning_preserved_on_inert_path
#print axioms inert_path_trace
#print axioms old_positive_coverage_through_repeated_width_and_updates
#print axioms active_depth_not_erased
end StructuralSSM


namespace StructuralSSM
open StrictGrowth

def feedback (x : Wide) : Noise :=
  match projected x with
  | none => 1
  | some false => 0
  | some true => 2

structure JointInput where
  e : Event
  paramL : Raw
  paramR : Raw
  physicsL : Noise
  physicsR : Noise

def encodeOp (e : Op) : Fin 5 :=
  if e.val = 0 then 2 else if e.val = 1 then 0 else if e.val = 2 then 3 else 1
theorem old_route_matches : ∀ (q : Mode) (o : Op),
    Fitted.specToken q (encodeOp o) = modeStep q o := by decide

-- Simultaneous, real two-way feedback inside the deployed recurrence,
-- not a proof-only parameter. Each path has a structural magnitude bound.
def wiredStep (xy : Wide × Wide) (t : JointInput) : Wide × Wide :=
  (deployed t.paramL xy.1 (encodeOp (leftOp t.e)) (combinedDist t.physicsL (feedback xy.2)),
   deployed t.paramR xy.2 (Fitted.encodeEvent t.e) (combinedDist t.physicsR (feedback xy.1)))

def wiredRun (xy : Wide × Wide) : List JointInput → Wide × Wide
  | [] => xy
  | t :: ts => wiredRun (wiredStep xy t) ts

def wideRef (q : Mode × Mode) : List JointInput → Mode × Mode
  | [] => q
  | t :: ts => wideRef (modeStep q.1 (leftOp t.e),rightMode q.2 t.e) ts

def live (w : World) : List JointInput → Prop
  | [] => True
  | t :: ts => attested w t.e ∧ live (actualStep w t.e) ts

def actualWorld (w : World) : List JointInput → World
  | [] => w
  | t :: ts => actualWorld (actualStep w t.e) ts

def wideRel (w : World) (xy : Wide × Wide) (q : Mode × Mode) : Prop :=
  cell q.1 xy.1 ∧ cell q.2 xy.2 ∧ newView (meaning q.1) (meaning q.2) w

theorem real_cross_coupling_wide_refinement (tr : List JointInput) (w : World)
    (xy : Wide × Wide) (q : Mode × Mode) (l : live w tr) (h : wideRel w xy q) :
    wideRel (actualWorld w tr) (wiredRun xy tr) (wideRef q tr) := by
  induction tr generalizing w xy q with
  | nil => exact h
  | cons t _ ih =>
    have a := deployed_rows t.paramL xy.1 q.1 (encodeOp (leftOp t.e))
       (combinedDist t.physicsL (feedback xy.2)) h.1
    have b := deployed_rows t.paramR xy.2 q.2 (Fitted.encodeEvent t.e)
       (combinedDist t.physicsR (feedback xy.1)) h.2.1
    rw [old_route_matches] at a
    rw [Fitted.token_projection_correct] at b
    exact ih _ _ _ l.2
      ⟨a,b,left_event_preserves w.1 w.2.1 w.2.2 q.1 t.e l.1 h.2.2.1,
           right_event_preserves w.1 w.2.1 w.2.2 q.2 t.e l.1 h.2.2.2⟩

def bothWide (xy : Wide × Wide) : Option Bool :=
  match projected xy.1,projected xy.2 with
  | some a,some b => some (Bool.xor a b)
  | _,_ => none

theorem wide_actual_lane (tr : List JointInput) (w : World) (xy : Wide × Wide)
    (q : Mode × Mode) (a b : Bool) (l : live w tr) (h : wideRel w xy q)
    (la : meaning (wideRef q tr).1 = some a) (rb : meaning (wideRef q tr).2 = some b) :
    bothWide (wiredRun xy tr) = some (Bool.xor a b) ∧
    lane (actualWorld w tr) = Bool.xor a b := by
  have inv := real_cross_coupling_wide_refinement tr w xy q l h
  have world := inv.2.2
  rw [la,rb] at world
  constructor
  · unfold bothWide
    rw [projected_cell _ _ inv.1,projected_cell _ _ inv.2.1,la,rb]
  · exact (new_necessarily_commits a b).2 _ world

def jtW (e : Event) : JointInput := ⟨e,⟨0,0,0⟩,⟨0,0,0⟩,1,1⟩

-- Same targeted query after adding the other rail. Physical old coordinate
-- differs causally with the other rail's state, while old observations agree.
theorem nonzero_installed_feedback_and_new_coverage :
    (wiredStep (0,0) (jtW 0)).1 ≠ (wiredStep (0,12) (jtW 0)).1 ∧
    projected (wiredStep (0,0) (jtW 0)).1 = some false ∧
    projected (wiredStep (0,12) (jtW 0)).1 = some false ∧
    bothWide (wiredRun (0,6) [jtW 6,jtW 0]) = some false ∧
    bothWide (wiredRun (0,6) [jtW 6,jtW 3]) = some true := by decide

-- A numeric reset only discards the certified right view; it is not an
-- authorization grant or external revocation proof.
theorem installed_observer_reset_can_remove_effect :
    bothWide (wiredRun (0,6) [jtW 6,jtW 0]) = some false ∧
    bothWide (wiredRun (0,6) [jtW 6,jtW 0,jtW 8]) = none := by decide

#print axioms old_route_matches
#print axioms real_cross_coupling_wide_refinement
#print axioms wide_actual_lane
#print axioms nonzero_installed_feedback_and_new_coverage
#print axioms installed_observer_reset_can_remove_effect
end StructuralSSM


namespace Unified
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000
open StrictGrowth StructuralSSM

def eventAgrees (e : Fin 5) (value : Bool) : Prop :=
  if e.val = 1 then value = true else if e.val = 3 then value = false else True
instance (e : Fin 5) (value : Bool) : Decidable (eventAgrees e value) := inferInstanceAs
  (Decidable (if e.val = 1 then value = true else if e.val = 3 then value = false else True))
def advanceWorld (e : Fin 5) (value : Bool) : Bool :=
  if e.val = 0 then !value else value

theorem one_world_step : ∀ (e : Fin 5) (value : Bool) (q : Mode),
    eventAgrees e value → compatible (meaning q) value →
    compatible (meaning (Fitted.specToken q e)) (advanceWorld e value) := by decide

structure Config where
  bank : Bank
  spec : Ghost bank
  world : Fin bank.n → Bool
  depth : Nat

def invariant (c : Config) : Prop :=
  closed c.bank c.spec ∧ ∀ i, compatible (meaning (c.spec.semantic i)) (c.world i)

structure Delivered (c : Config) where
  param : Fin c.bank.n → Raw
  event : Fin c.bank.n → Fin 5
  physical : Fin c.bank.n → Noise
  route : Fin c.bank.n → Fin c.bank.n
  front : Fin c.bank.n → Nat → Hold
  back : Fin c.bank.n → Nat → Hold

def observedLaw (c : Config) (t : Delivered c) : Prop :=
  ∀ i, eventAgrees (t.event i) (c.world i)

def routedWire (c : Config) (t : Delivered c) (i : Fin c.bank.n) : Wide :=
  let n := combinedDist (t.physical i) (feedback (c.bank.x (t.route i)))
  if c.depth = 0 then deployed (t.param i) (c.bank.x i) (t.event i) n else
    let body := deepStep ((List.range c.depth).map (t.front i))
      ((List.range c.depth).map (t.back i)) (t.param i) (t.event i)
      (combinedDist (t.physical i) 1) (c.bank.x i)
    deployed (t.param i) body 2 n

theorem routedWire_closed (c : Config) (t : Delivered c) (i : Fin c.bank.n)
    (h : cell (c.spec.semantic i) (c.bank.x i)) :
    cell (Fitted.specToken (c.spec.semantic i) (t.event i)) (routedWire c t i) := by
  unfold routedWire
  split
  · exact deployed_rows _ _ _ _ _ h
  · have hb := arbitrary_depth_and_raw_closed ((List.range c.depth).map (t.front i))
       ((List.range c.depth).map (t.back i)) (t.param i) (t.event i)
       (combinedDist (t.physical i) 1) (c.spec.semantic i) (c.bank.x i) h
    have hf := deployed_rows (t.param i) _ _ 2
      (combinedDist (t.physical i) (feedback (c.bank.x (t.route i)))) hb
    rw [hold_modes] at hf
    exact hf

def forward (c : Config) (t : Delivered c) : Config :=
  ⟨⟨c.bank.n,routedWire c t⟩,
   ⟨fun i => Fitted.specToken (c.spec.semantic i) (t.event i)⟩,
   fun i => advanceWorld (t.event i) (c.world i), c.depth⟩

def addRail (c : Config) (actual : Bool) : Config :=
  ⟨widen c.bank, widenRef c.bank c.spec, Fin.lastCases actual c.world, c.depth⟩

def growDepth (c : Config) : Config := {c with depth := c.depth + 1}

theorem forward_invariant (c : Config) (t : Delivered c) (legal : observedLaw c t)
    (h : invariant c) : invariant (forward c t) :=
  ⟨fun i => routedWire_closed c t i (h.1 i),
    fun i => one_world_step _ _ _ (legal i) (h.2 i)⟩

theorem add_rail_invariant (c : Config) (actual : Bool) (h : invariant c) :
    invariant (addRail c actual) := by
  constructor
  · exact widen_closed _ _ h.1
  · intro i
    refine Fin.lastCases ?_ (fun k => ?_) i
    · change compatible (meaning (Fin.lastCases (motive := fun _ => Mode) 0 c.spec.semantic
        (Fin.last c.bank.n))) (Fin.lastCases (motive := fun _ => Bool) actual c.world (Fin.last c.bank.n))
      rw [Fin.lastCases_last, Fin.lastCases_last]
      exact Or.inl rfl
    · change compatible (meaning (Fin.lastCases (motive := fun _ => Mode) 0 c.spec.semantic
        k.castSucc)) (Fin.lastCases (motive := fun _ => Bool) actual c.world k.castSucc)
      rw [Fin.lastCases_castSucc, Fin.lastCases_castSucc]
      exact h.2 k

-- The same executable trace includes bounded live feedback, inference using
-- every optimizer parameter value, insertion of new live coordinates, and depth.
inductive Run (init : Config) : Config → Prop where
  | start : Run init init
  | execute {c : Config} : Run init c → (t : Delivered c) → observedLaw c t → Run init (forward c t)
  | width {c : Config} : Run init c → (value : Bool) → Run init (addRail c value)
  | depth {c : Config} : Run init c → Run init (growDepth c)

theorem all_mixed_transitions {init fin : Config} (r : Run init fin) (h : invariant init) :
    invariant fin := by
  induction r with
  | start => exact h
  | execute _ t live ih => exact forward_invariant _ t live ih
  | width _ value ih => exact add_rail_invariant _ value ih
  | depth _ ih => exact ih

def read (c : Config) (i : Fin c.bank.n) : Option Bool := projected (c.bank.x i)

theorem mandatory_read_sound {init fin : Config} (r : Run init fin) (h : invariant init)
    (i : Fin fin.bank.n) (b : Bool) (eligible : meaning (fin.spec.semantic i) = some b) :
    read fin i = some b ∧ fin.world i = b := by
  have hh := all_mixed_transitions r h
  constructor
  · exact (projected_cell _ _ (hh.1 i)).trans eligible
  · have hr := hh.2 i
    rw [eligible] at hr
    cases hr with
    | inl bad => cases bad
    | inr succ => exact (Option.some.inj succ).symm

-- Track the old query's referent explicitly across index embeddings. Other
-- coordinates may be updated arbitrarily; only the queried one must receive HOLD.
inductive StablePath (start : Config) (i : Fin start.bank.n) :
    (c : Config) → Fin c.bank.n → Prop where
  | nil : StablePath start i start i
  | exec {c : Config} {j : Fin c.bank.n} : StablePath start i c j →
      (t : Delivered c) → observedLaw c t → t.event j = 2 →
      StablePath start i (forward c t) j
  | wid {c : Config} {j : Fin c.bank.n} : StablePath start i c j →
      (v : Bool) → StablePath start i (addRail c v) j.castSucc
  | dep {c : Config} {j : Fin c.bank.n} : StablePath start i c j →
      StablePath start i (growDepth c) j

theorem stable_modes {start final : Config} {i : Fin start.bank.n} {j : Fin final.bank.n}
    (p : StablePath start i final j) : final.spec.semantic j = start.spec.semantic i := by
  induction p with
  | nil => rfl
  | @exec c j _ t _ hold ih =>
    change Fitted.specToken (c.spec.semantic j) (t.event j) = _
    rw [hold,hold_modes]
    exact ih
  | @wid c j _ _ ih =>
    change (Fin.lastCases (motive := fun _ => Mode) 0 c.spec.semantic j.castSucc) = _
    rw [Fin.lastCases_castSucc]
    exact ih
  | dep _ ih => exact ih

theorem stable_is_run {start final : Config} {i : Fin start.bank.n} {j : Fin final.bank.n}
    (p : StablePath start i final j) : Run start final := by
  induction p with
  | nil => exact .start
  | exec _ t live _ ih => exact .execute ih t live
  | wid _ v ih => exact .width ih v
  | dep _ ih => exact .depth ih

theorem old_coverage_preserved_on_unified_trace {start final : Config}
    {i : Fin start.bank.n} {j : Fin final.bank.n} (p : StablePath start i final j)
    (h : invariant start) (b : Bool) (oldEligible : meaning (start.spec.semantic i) = some b) :
    read final j = some b ∧ final.world j = b := by
  apply mandatory_read_sound (stable_is_run p) h j b
  rw [stable_modes p]
  exact oldEligible

def composed (c : Config) (i j : Fin c.bank.n) := bothWide (c.bank.x i,c.bank.x j)

theorem composed_answers_are_correct {start final : Config} (r : Run start final)
    (h : invariant start) (i j : Fin final.bank.n) (a b : Bool)
    (left : meaning (final.spec.semantic i) = some a)
    (right : meaning (final.spec.semantic j) = some b) :
    composed final i j = some (Bool.xor a b) ∧
    Bool.xor (final.world i) (final.world j) = Bool.xor a b := by
  have hl := mandatory_read_sound r h i a left
  have hr := mandatory_read_sound r h j b right
  constructor
  · change (match read final i, read final j with
             | some a, some b => some (Bool.xor a b)
             | _, _ => none) = _
    rw [hl.1,hr.1]
  · rw [hl.2,hr.2]

#print axioms one_world_step
#print axioms forward_invariant
#print axioms add_rail_invariant
#print axioms all_mixed_transitions
#print axioms mandatory_read_sound
#print axioms old_coverage_preserved_on_unified_trace
#print axioms composed_answers_are_correct

def oldOnly : Config := ⟨⟨1,fun _ => 0⟩,⟨fun _ => 1⟩,fun _ => false,0⟩
def liveInstall (value : Bool) : Config := growDepth (addRail oldOnly value)

def afterNewEvidence (value : Bool) : Delivered (liveInstall value) :=
  ⟨fun i => ⟨1,i.val,1⟩,
   fun i => if i.val = 0 then 2 else if value then 1 else 3,
   fun _ => 1,
   fun i => if i.val = 0 then (1 : Fin 2) else (0 : Fin 2),
   fun _ _ => ⟨⟨1,0,0⟩,0⟩,
   fun _ _ => ⟨⟨0,1,0⟩,4⟩⟩

def installedForward (value : Bool) : Config :=
  forward (liveInstall value) (afterNewEvidence value)

theorem live_growth_scheduled_legal : ∀ value : Bool,
    observedLaw (liveInstall value) (afterNewEvidence value) := by
  intro value
  have h : ∀ i : Fin 2, eventAgrees ((afterNewEvidence value).event i) ((liveInstall value).world i) := by
    cases value <;> decide
  exact h

def installedTrace (value : Bool) : Run oldOnly (installedForward value) :=
  .execute (.depth (.width .start value)) (afterNewEvidence value)
    (live_growth_scheduled_legal value)

theorem installed_joint_strict_positive :
    read (liveInstall false) (1 : Fin 2) = none ∧ read (liveInstall true) (1 : Fin 2) = none ∧
    read (installedForward false) (0 : Fin 2) = some false ∧ read (installedForward true) (0 : Fin 2) = some false ∧
    composed (installedForward false) (0 : Fin 2) (1 : Fin 2) = some false ∧
    composed (installedForward true) (0 : Fin 2) (1 : Fin 2) = some true := by decide

#print axioms live_growth_scheduled_legal
#print axioms installedTrace
#print axioms installed_joint_strict_positive


def afterInertTick (value : Bool) : Delivered (installedForward value) :=
  ⟨fun _ => ⟨0,0,0⟩,
   fun _ => 2,
   fun _ => 1,
   fun i => if i.val = 0 then (1 : Fin 2) else (0 : Fin 2),
   fun _ _ => ⟨⟨0,0,0⟩,2⟩,
   fun _ _ => ⟨⟨0,0,0⟩,2⟩⟩
def afterInertState (value : Bool) : Config := forward (installedForward value) (afterInertTick value)

theorem second_tick_legal (value : Bool) :
    observedLaw (installedForward value) (afterInertTick value) := by
  intro _
  exact True.intro

def liveMixedRun (value : Bool) : Run oldOnly (afterInertState value) :=
  .execute (installedTrace value) (afterInertTick value) (second_tick_legal value)

theorem live_same_old_meaning_with_surviving_cross_feedback :
    (afterInertState false).depth = 1 ∧
    (afterInertState true).depth = 1 ∧
    (afterInertState false).bank.x (0 : Fin 2) ≠
       (afterInertState true).bank.x (0 : Fin 2) ∧
    read (afterInertState false) (0 : Fin 2) = some false ∧
    read (afterInertState true) (0 : Fin 2) = some false ∧
    composed (afterInertState false) (0 : Fin 2) (1 : Fin 2) = some false ∧
    composed (afterInertState true) (0 : Fin 2) (1 : Fin 2) = some true := by decide

#print axioms routedWire_closed
#print axioms second_tick_legal
#print axioms liveMixedRun
#print axioms live_same_old_meaning_with_surviving_cross_feedback

end Unified
