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


namespace Fitted
open StrictGrowth
def pythonTable : List Nat := [5, 6, 6, 5, 6, 6, 0, 0, 1, 0, 0, 1, 2, 3, 4, 5, 6, 6, 5, 6, 6, 0, 0, 1, 0, 0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 6, 2, 3, 4, 0, 0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 6, 2, 3, 4, 0, 0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 6, 2, 3, 4, 0, 0, 1, 2, 3, 4, 0, 0, 1, 5, 6, 6, 5, 6, 6, 0, 0, 1, 2, 3, 4, 0, 0, 1, 5, 6, 6, 5, 6, 6, 0, 0, 1, 2, 3, 4]
def finOf (n k : Nat) (h : 0 < n) : Fin n := ⟨k % n, Nat.mod_lt _ h⟩
def referenceTable : List Nat :=
    List.join ((List.range 7).map (fun x =>
      List.join ((List.range 5).map (fun e =>
        (List.range 3).map (fun d =>
          (step (finOf 7 x (by decide)) (finOf 5 e (by decide)) (finOf 3 d (by decide))).val)))))
theorem exact_cpu_local_fixture_agrees : referenceTable = pythonTable := by decide
#print axioms exact_cpu_local_fixture_agrees
end Fitted
