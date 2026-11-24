namespace ControlledTokenGrammar
inductive Piece where
  | exactToken (id : Nat)
  | capture (role : Nat)
deriving DecidableEq

abbrev Capture := Nat × Nat

inductive Recognizes : List Piece → List Nat → List Capture → Prop
  | nil : Recognizes [] [] []
  | lit {p : List Piece} {w : List Nat} {c : List Capture} (tok : Nat) :
      Recognizes p w c → Recognizes (.exactToken tok :: p) (tok :: w) c
  | bind {p : List Piece} {w : List Nat} {c : List Capture} (r tok : Nat) :
      Recognizes p w c → Recognizes (.capture r :: p) (tok :: w) ((r,tok)::c)

def scan : List Piece → List Nat → Option (List Capture)
  | [], [] => some []
  | .exactToken tok :: p, y :: ys => if tok = y then scan p ys else none
  | .capture r :: p, y :: ys => (scan p ys).map (fun h => (r,y)::h)
  | _, _ => none

theorem scan_complete {p : List Piece} {w : List Nat} {c : List Capture}
    (h : Recognizes p w c) : scan p w = some c := by
  induction h with
  | nil => rfl
  | lit tok _ ih => simp [scan,ih]
  | bind r tok _ ih => simp [scan,ih]

theorem scan_sound (p : List Piece) (w : List Nat) (c : List Capture)
    (h : scan p w = some c) : Recognizes p w c := by
  induction p generalizing w c with
  | nil =>
    cases w with
    | nil => cases h; exact .nil
    | cons x xs => cases h
  | cons first rest ih =>
    cases w with
    | nil => cases first <;> cases h
    | cons x xs =>
      cases first with
      | exactToken tok =>
        by_cases e : tok = x
        · cases e
          have hh : scan rest xs = some c := by simpa [scan] using h
          exact .lit x (ih xs c hh)
        · simp [scan,e] at h
      | capture r =>
        cases hb : scan rest xs with
        | none => simp [scan,hb] at h
        | some cs =>
          have hh : (r,x) :: cs = c := Option.some.inj (by simpa [scan,hb] using h)
          cases hh
          exact .bind r x (ih xs cs hb)

def allCaptures (grammar : List (List Piece)) (w : List Nat) : List (List Capture) :=
  match grammar with
  | [] => []
  | p::ps => match scan p w with
    | none => allCaptures ps w
    | some cs => cs :: allCaptures ps w

theorem every_formal_parse_retained (grammar : List (List Piece)) (w : List Nat)
    (p : List Piece) (c : List Capture) (member : p ∈ grammar)
    (parses : Recognizes p w c) : c ∈ allCaptures grammar w := by
  induction grammar with
  | nil => cases member
  | cons a tail ih =>
    cases List.mem_cons.mp member with
    | inl eq =>
      cases eq
      have hm := scan_complete parses
      simp [allCaptures,hm,List.mem_cons]
    | inr tailmem =>
      have q := ih tailmem
      cases hs : scan a w with
      | none => simpa [allCaptures,hs] using q
      | some other =>
        have : c ∈ other :: allCaptures tail w := List.mem_cons.mpr (Or.inr q)
        simpa [allCaptures,hs] using this

-- Two controlled token forms; placeholders capture symbols, not final facts.
-- Actual byte-to-token correspondence, scoped name resolution and certified
-- evidence stay separate. Unknown text cannot silently generate authority.
def form : List Piece :=
  [.exactToken 10,.capture 1,.exactToken 11,.capture 2,.exactToken 12,.capture 3,.exactToken 13]
def rival : List Piece :=
  [.exactToken 10,.capture 2,.exactToken 11,.capture 1,.exactToken 12,.capture 3,.exactToken 13]
def goodWords : List Nat := [10,101,11,201,12,301,13]

theorem no_ignored_suffix :
    allCaptures [form] (goodWords ++ [77]) = [] := by decide
theorem no_ignored_quote_prefix :
    allCaptures [form] (88 :: goodWords) = [] := by decide
theorem overlapping_forms_retain_distinct_roles :
    allCaptures [form,rival] goodWords =
      [[(1,101),(2,201),(3,301)],[(2,101),(1,201),(3,301)]] := by decide

#print axioms scan_complete
#print axioms scan_sound
#print axioms every_formal_parse_retained
#print axioms no_ignored_suffix
#print axioms no_ignored_quote_prefix
#print axioms overlapping_forms_retain_distinct_roles
end ControlledTokenGrammar

namespace ProtectedBinding
structure Key where
  account : Nat
  scope : Nat
  entity : Nat
  field : Nat
deriving DecidableEq
structure FactValue where
  value : Bool
  frontier : Nat
deriving DecidableEq
structure Draft where
  request : Nat
  addressed : Key
  value : Bool
  frontier : Nat
deriving DecidableEq

structure Current where
  request : Nat
  frontier : Nat
  liveRead : Key → Option FactValue
  grant : Key → Bool

-- This function sees exact actual recurrent outputs, not reference outcomes.
-- A canonical pointer from the learned text head is just an untrusted address.
def propose (curr : Current) (pointer : Key) : Option Draft :=
  match curr.liveRead pointer with
  | none => none
  | some x => some ⟨curr.request,pointer,x.value,x.frontier⟩

-- none is an explicit ambiguous/unsupported interpretation, not 'no rival'.
def unanimous (possible : List (Option Key)) (k : Key) : Bool :=
  !possible.isEmpty && possible.all (fun x => decide (x = some k))

def liveContext (curr : Current) (ast : Draft) : Bool :=
  decide (ast.request = curr.request) && decide (ast.frontier = curr.frontier) &&
    curr.grant ast.addressed &&
    decide (curr.liveRead ast.addressed = some ⟨ast.value,curr.frontier⟩)

-- Exact candidate is returned unchanged or rejected, never repaired.
-- Cover soundness is separate: the runtime caller may not select its own list.
def admit (curr : Current) (possible : List (Option Key)) (raw : Option Draft) : Option Draft :=
  match raw with
  | none => none
  | some a => if unanimous possible a.addressed && liveContext curr a then some a else none

theorem unchanged_or_reject (curr : Current) (possible : List (Option Key)) (raw : Option Draft) (a : Draft)
    (h : admit curr possible raw = some a) :
    raw = some a ∧ unanimous possible a.addressed = true ∧ liveContext curr a = true := by
  cases raw with
  | none => cases h
  | some p =>
    change (if unanimous possible p.addressed && liveContext curr p then some p else none) = some a at h
    cases he : unanimous possible p.addressed && liveContext curr p with
    | false => simp [he] at h
    | true =>
      have hp : p = a := Option.some.inj (by simpa [he] using h)
      cases hp
      exact ⟨rfl,(Bool.and_eq_true_iff _ _).mp he⟩

theorem proves_intended_full_address (curr : Current) (possible : List (Option Key)) (p : Key)
    (a : Draft) (actual : Option Key)
    (coverContainsActual : actual ∈ possible)
    (h : admit curr possible (propose curr p) = some a) :
    actual = some a.addressed ∧ curr.grant a.addressed = true ∧
    curr.liveRead p = some ⟨a.value,curr.frontier⟩ ∧ a.addressed = p := by
  have hh := unchanged_or_reject curr possible (propose curr p) a h
  have valid := (Bool.and_eq_true_iff _ _).mp hh.2.2
  have live := (Bool.and_eq_true_iff _ _).mp valid.1
  have un := (Bool.and_eq_true_iff _ _).mp hh.2.1
  have actualEq := of_decide_eq_true (List.all_eq_true.mp un.2 _ coverContainsActual)
  cases hread : curr.liveRead p with
  | none => simp [propose,hread] at hh
  | some d =>
    have he : (⟨curr.request,p,d.value,d.frontier⟩ : Draft) = a :=
      Option.some.inj (by simpa [propose,hread] using hh.1)
    cases he
    have hb : curr.liveRead p = some ⟨d.value,curr.frontier⟩ := of_decide_eq_true valid.2
    rw [hread] at hb
    exact ⟨actualEq,live.2,hb,rfl⟩

def namedA : Key := ⟨1,40,7,1⟩
def namedB : Key := ⟨1,40,8,1⟩
def current (a b permit : Bool) : Current :=
  ⟨6,20,fun p => if p = namedA then some ⟨a,20⟩ else if p = namedB then some ⟨b,20⟩ else none,
    fun p => permit && decide (p = namedA ∨ p = namedB)⟩

theorem positive_paired_binder_sensitive :
    admit (current true false true) [some namedA]
       (propose (current true false true) namedA) = some ⟨6,namedA,true,20⟩ ∧
    admit (current true false true) [some namedB]
       (propose (current true false true) namedB) = some ⟨6,namedB,false,20⟩ := by decide

theorem same_bit_cannot_hide_ambiguous_address :
    admit (current true true true) [some namedA,some namedB]
      (propose (current true true true) namedA) = none := by decide

theorem broad_grant_cannot_repair_wrong_pointer :
    (current true false true).grant namedB = true ∧
    admit (current true false true) [some namedA]
      (propose (current true false true) namedB) = none := by decide

theorem unsupported_and_revoke_controls :
    admit (current true false true) [none,some namedA]
      (propose (current true false true) namedA) = none ∧
    propose (current true false false) namedA = propose (current true false true) namedA ∧
    admit (current true false false) [some namedA]
      (propose (current true false false) namedA) = none := by decide

#print axioms unchanged_or_reject
#print axioms proves_intended_full_address
#print axioms positive_paired_binder_sensitive
#print axioms same_bit_cannot_hide_ambiguous_address
#print axioms broad_grant_cannot_repair_wrong_pointer
#print axioms unsupported_and_revoke_controls

-- Unlike the model, this check re-reads the exact protected address in the
-- same modeled transition. It cannot substitute a different scope or payload.
theorem authorized_current_candidate (curr : Current) (possible : List (Option Key))
    (raw : Option Draft) (a : Draft) (actual : Option Key) (cover : actual ∈ possible)
    (accepted : admit curr possible raw = some a) :
    raw = some a ∧ actual = some a.addressed ∧ curr.grant a.addressed = true ∧
    curr.liveRead a.addressed = some ⟨a.value,curr.frontier⟩ := by
  have hr := unchanged_or_reject curr possible raw a accepted
  have hm := (Bool.and_eq_true_iff _ _).mp hr.2.1
  have hc := (Bool.and_eq_true_iff _ _).mp hr.2.2
  have hg := (Bool.and_eq_true_iff _ _).mp hc.1
  exact ⟨hr.1, of_decide_eq_true (List.all_eq_true.mp hm.2 _ cover), hg.2,
    of_decide_eq_true hc.2⟩

theorem same_stamp_wrong_value_cannot_redeem :
    admit (current true false true) [some namedA] (some ⟨6,namedA,false,20⟩) = none := by decide

#print axioms authorized_current_candidate
#print axioms same_stamp_wrong_value_cannot_redeem

end ProtectedBinding


namespace ControlledBinding
open ControlledTokenGrammar ProtectedBinding

structure Entry where
  spellingScope : Nat
  spellingEntity : Nat
  spellingField : Nat
  canonical : Key
  frontier : Nat

def onlyRole (caps : List Capture) (role word : Nat) : Bool :=
  ((caps.filter (fun p => p.1 == role)).map (fun p => p.2)) == [word]

def matchesEntry (now : Current) (caps : List Capture) (en : Entry) : Bool :=
  en.frontier == now.frontier &&
  onlyRole caps 1 en.spellingScope && onlyRole caps 2 en.spellingEntity &&
  onlyRole caps 3 en.spellingField

def compatibleEntries (now : Current) (caps : List Capture) (d : List Entry) : List Entry :=
  d.filter (matchesEntry now caps)

def resolve (now : Current) (caps : List Capture) (d : List Entry) : List (Option Key) :=
  match compatibleEntries now caps d with
  | [] => [none]
  | x :: xs => (x :: xs).map (some ∘ Entry.canonical)

def flatten (f : List Capture → List (Option Key)) : List (List Capture) → List (Option Key)
  | [] => []
  | a :: as => f a ++ flatten f as

theorem flatten_keeps (f : List Capture → List (Option Key)) (xs : List (List Capture))
    (c : List Capture) (a : Option Key) (inCaptures : c ∈ xs) (inResolve : a ∈ f c) :
    a ∈ flatten f xs := by
  induction xs with
  | nil => cases inCaptures
  | cons y ys ih =>
    cases List.mem_cons.mp inCaptures with
    | inl eq =>
      cases eq
      exact List.mem_append_of_mem_left _ inResolve
    | inr h => exact List.mem_append_of_mem_right _ (ih h)

def scopedCover (now : Current) (grammar : List (List Piece)) (d : List Entry) (w : List Nat) :
    List (Option Key) :=
  match allCaptures grammar w with
  | [] => [none]
  | c :: cs => flatten (fun k => resolve now k d) (c :: cs)

inductive Meaning (now : Current) (grammar : List (List Piece))
    (d : List Entry) (w : List Nat) : Option Key → Prop where
  | noParse : allCaptures grammar w = [] → Meaning now grammar d w none
  | known {p caps e} : p ∈ grammar → Recognizes p w caps → e ∈ d →
      matchesEntry now caps e = true → Meaning now grammar d w (some e.canonical)
  | ambiguousNoEntry {p caps} : p ∈ grammar → Recognizes p w caps →
      compatibleEntries now caps d = [] → Meaning now grammar d w none

theorem entry_resolution_complete (now : Current) (caps : List Capture) (d : List Entry)
    (e : Entry) (inD : e ∈ d) (fits : matchesEntry now caps e = true) :
    some e.canonical ∈ resolve now caps d := by
  have hm : e ∈ compatibleEntries now caps d := List.mem_filter.mpr ⟨inD,fits⟩
  cases hg : compatibleEntries now caps d with
  | nil => rw [hg] at hm; cases hm
  | cons a xs =>
    rw [hg] at hm
    unfold resolve
    rw [hg]
    exact List.mem_map.mpr ⟨e,hm,rfl⟩

theorem captures_transfer (now : Current) (grammar : List (List Piece)) (d : List Entry)
    (w : List Nat) (c : List Capture) (a : Option Key)
    (member : c ∈ allCaptures grammar w) (supported : a ∈ resolve now c d) :
    a ∈ scopedCover now grammar d w := by
  cases hg : allCaptures grammar w with
  | nil => rw [hg] at member; cases member
  | cons p rest =>
    rw [hg] at member
    unfold scopedCover
    rw [hg]
    exact flatten_keeps _ _ _ _ member supported

theorem actual_meaning_in_cover {now grammar d w a}
    (law : Meaning now grammar d w a) : a ∈ scopedCover now grammar d w := by
  cases law with
  | noParse np =>
    unfold scopedCover; rw [np]
    exact List.mem_cons.mpr (Or.inl rfl)
  | known mp parse source fits =>
    exact captures_transfer _ _ _ _ _ _
      (every_formal_parse_retained _ _ _ _ mp parse)
      (entry_resolution_complete _ _ _ _ source fits)
  | ambiguousNoEntry mp parse u =>
    apply captures_transfer _ _ _ _ _ _
      (every_formal_parse_retained _ _ _ _ mp parse)
    unfold resolve; rw [u]
    exact List.mem_cons.mpr (Or.inl rfl)

def guarded (now : Current) (grammar : List (List Piece)) (d : List Entry)
    (w : List Nat) (raw : Option Draft) : Option Draft :=
  admit now (scopedCover now grammar d w) raw

theorem audited_binding (now : Current) (grammar : List (List Piece)) (d : List Entry)
    (w : List Nat) (actual : Option Key) (raw : Option Draft) (a : Draft)
    (meaning : Meaning now grammar d w actual) (approved : guarded now grammar d w raw = some a) :
    raw = some a ∧ actual = some a.addressed ∧ now.grant a.addressed = true ∧
      now.liveRead a.addressed = some ⟨a.value,now.frontier⟩ :=
  authorized_current_candidate _ _ _ _ _ (actual_meaning_in_cover meaning) approved

def anEntry : Entry := ⟨101,201,301,namedA,20⟩
def confusing : Entry := ⟨201,101,301,namedB,20⟩

theorem integrated_positive_and_negative :
    guarded (current true true true) [form] [anEntry,confusing] goodWords
      (propose (current true true true) namedA) = some ⟨6,namedA,true,20⟩ ∧
    guarded (current true true true) [form,rival] [anEntry,confusing] goodWords
      (propose (current true true true) namedA) = none ∧
    guarded (current true true true) [form] [anEntry,confusing] (goodWords++[77])
      (propose (current true true true) namedA) = none ∧
    guarded (current true true true) [form] [{anEntry with frontier := 19},confusing] goodWords
      (propose (current true true true) namedA) = none := by decide

#print axioms flatten_keeps
#print axioms entry_resolution_complete
#print axioms captures_transfer
#print axioms actual_meaning_in_cover
#print axioms audited_binding
#print axioms integrated_positive_and_negative
end ControlledBinding


namespace ByteProtocol
open ControlledTokenGrammar ProtectedBinding ControlledBinding

-- Nats represent byte values at this interface. Exact successful equality
-- to the encoder below forces every payload atom to be one ASCII byte.
abbrev Bytes := List Nat
def REQUEST : Nat := 209
def CLAIM : Nat := 210
def space : Nat := 32

def seal (tag : Nat) (payload : Bytes) : Bytes := tag :: payload.length :: payload

def openFrame (tag : Nat) : Bytes → Option Bytes
  | got :: len :: body =>
    if got = tag ∧ len = body.length ∧ body.length ≤ 96 then some body else none
  | _ => none

theorem unseal_exact (tag : Nat) (input out : Bytes) (h : openFrame tag input = some out) :
    input = seal tag out ∧ out.length ≤ 96 := by
  cases input with
  | nil => cases h
  | cons got rest =>
    cases rest with
    | nil => cases h
    | cons len body =>
      by_cases ok : got = tag ∧ len = body.length ∧ body.length ≤ 96
      · have hb : body = out := Option.some.inj (by simpa [openFrame,ok] using h)
        cases hb
        rw [ok.1,ok.2.1]
        exact ⟨rfl,ok.2.2⟩
      · simp [openFrame,ok] at h

theorem sealed_opens (tag : Nat) (body : Bytes) (bound : body.length ≤ 96) :
    openFrame tag (seal tag body) = some body := by simp [openFrame,seal,bound]

-- No event is emitted by append; reading occurs only at close.
structure Buffered where
  input : Bytes

def feed (s : Buffered) (chunk : Bytes) : Buffered := ⟨s.input ++ chunk⟩
theorem feed_chunking (s : Buffered) (x y : Bytes) :
    feed (feed s x) y = feed s (x ++ y) := by simp [feed,List.append_assoc]

def splitAux (current : Bytes) : Bytes → List Bytes
  | [] => [current.reverse]
  | x :: xs =>
      if x = space then current.reverse :: splitAux [] xs
      else splitAux (x::current) xs

def vocabulary : List (Nat × Bytes) :=
  [(10,[114,101,97,100]),         -- read
   (11,[111,102]),                -- of
   (12,[105,110]),                -- in
   (301,[114,101,97,100,121]),    -- ready
   (302,[99,108,101,97,114,101,100]), -- cleared
   (201,[97]),(202,[98]),         -- a,b
   (101,[120]),(102,[121])]       -- x,y

def wordCode (body : Bytes) : List (Nat × Bytes) → Option Nat
  | [] => none
  | (tok,w)::xs => if body = w then some tok else wordCode body xs
def codeWord (tok : Nat) : List (Nat × Bytes) → Option Bytes
  | [] => none
  | (id,w)::xs => if tok = id then some w else codeWord tok xs

def scanWords : List Bytes → Option (List Nat)
  | [] => some []
  | w::ws => match wordCode w vocabulary, scanWords ws with
    | some id, some ids => some (id::ids)
    | _,_ => none

def encodeTokens : List Nat → Option Bytes
  | [] => some []
  | [tok] => codeWord tok vocabulary
  | tok::rest =>
    match codeWord tok vocabulary, encodeTokens rest with
    | some w, some w2 => some (w ++ [space] ++ w2)
    | _,_ => none

-- Executable lexer. No display normalization, no ignored bytes and no partial
-- tokens. The encoded check excludes alternative whitespace or spelling.
def canonicalTokens (body : Bytes) : Option (List Nat) :=
  match scanWords (splitAux [] body) with
  | none => none
  | some ts => if encodeTokens ts = some body then some ts else none
def parseRequest (input : Bytes) : Option (List Nat) :=
  (openFrame REQUEST input).bind canonicalTokens

theorem canonicalTokens_exact (body : Bytes) (tokens : List Nat)
    (h : canonicalTokens body = some tokens) :
    encodeTokens tokens = some body := by
  unfold canonicalTokens at h
  cases hs : scanWords (splitAux [] body) with
  | none => simp [hs] at h
  | some ts =>
    simp only [hs] at h
    by_cases he : encodeTokens ts = some body
    · have ht : ts = tokens := Option.some.inj (by simpa [he] using h)
      cases ht
      exact he
    · simp [he] at h

theorem accepted_frame_is_exactly_encoder (input : Bytes) (tokens : List Nat)
    (ok : parseRequest input = some tokens) :
    ∃ payload, input = seal REQUEST payload ∧ encodeTokens tokens = some payload ∧ payload.length ≤ 96 := by
  cases hw : openFrame REQUEST input with
  | none => simp [parseRequest,hw] at ok
  | some payload =>
    have h := unseal_exact REQUEST input payload hw
    have lex : canonicalTokens payload = some tokens := by simpa [parseRequest,hw] using ok
    exact ⟨payload,h.1,canonicalTokens_exact payload tokens lex,h.2⟩

def requestTokens (scope entity field : Bool) : List Nat :=
  [10,if field then 302 else 301,11,if entity then 202 else 201,12,if scope then 102 else 101]
def requestBytes (scope entity field : Bool) : Bytes :=
  seal REQUEST ((encodeTokens (requestTokens scope entity field)).getD [])

theorem all_declared_queries_parse : ∀ s e f : Bool,
    parseRequest (requestBytes s e f) = some (requestTokens s e f) := by decide

-- Exact role capture. No grammar token creates evidence or authority.
def requestForm : List Piece :=
  [.exactToken 10,.capture 3,.exactToken 11,.capture 2,.exactToken 12,.capture 1]
def realQueryGrammar := [requestForm]

def addTail (b : Bytes) (tail : Bytes) : Bytes :=
  seal REQUEST ((openFrame REQUEST b).getD [] ++ tail)

theorem byte_attack_controls :
    parseRequest (addTail (requestBytes false false false) [0]) = none ∧
    parseRequest (addTail (requestBytes false false false) [32,226,128,139]) = none ∧
    parseRequest (addTail (requestBytes false false false) [32,255]) = none ∧
    parseRequest ((requestBytes false false false)++[0]) = none ∧
    parseRequest (seal REQUEST [34,114,101,97,100,34]) = none ∧
    parseRequest (seal CLAIM ((openFrame REQUEST (requestBytes false false false)).getD [])) = none := by decide

#print axioms unseal_exact
#print axioms sealed_opens
#print axioms feed_chunking
#print axioms accepted_frame_is_exactly_encoder
#print axioms all_declared_queries_parse
#print axioms byte_attack_controls
end ByteProtocol


namespace FactRenderer
set_option maxRecDepth 16000
set_option maxHeartbeats 1500000
open ByteProtocol ProtectedBinding

def digit (n : Nat) : Nat := if n < 10 then 48+n else 87+n
def undigit (b : Nat) : Option Nat :=
  if 48 ≤ b ∧ b ≤ 57 then some (b-48) else
  if 97 ≤ b ∧ b ≤ 102 then some (b-87) else none
def hi (n : Nat) := digit (n / 16)
def lo (n : Nat) := digit (n % 16)
def decodePair (h l : Nat) : Option Nat := do
  let a ← undigit h
  let b ← undigit l
  pure (16*a+b)

theorem every_byte_rendered : ∀ b : Fin 256, decodePair (hi b.val) (lo b.val) = some b.val := by decide
theorem pair_roundtrip (b : Nat) (short : b < 256) :
    decodePair (hi b) (lo b) = some b := every_byte_rendered ⟨b,short⟩

def valueByte (b : Bool) := if b then 49 else 48
def decodeBool (b : Nat) : Option Bool := if b = 48 then some false else if b = 49 then some true else none
theorem bool_roundtrip : ∀ b : Bool, decodeBool (valueByte b) = some b := by decide

-- Human-readable controlled factual assertion with exact two-hex-digit opaque
-- identifiers. Meaning is this exact record, not an unconstrained audience.
def body (d : Draft) : Bytes := [102,97,99,116,32,114,101,113,117,101,115,116,32,hi d.request,lo d.request,32,97,99,99,111,117,110,116,32,hi d.addressed.account,lo d.addressed.account,32,115,99,111,112,101,32,hi d.addressed.scope,lo d.addressed.scope,32,101,110,116,105,116,121,32,hi d.addressed.entity,lo d.addressed.entity,32,102,105,101,108,100,32,hi d.addressed.field,lo d.addressed.field,32,118,97,108,117,101,32,valueByte d.value,32,102,114,111,110,116,105,101,114,32,hi d.frontier,lo d.frontier]

def getPair (bs : Bytes) (i j : Nat) : Option Nat := do
  let x ← bs.get? i
  let y ← bs.get? j
  decodePair x y

def decodeBody (bs : Bytes) : Option Draft := do
  let req ← getPair bs 13 14
  let a ← getPair bs 24 25
  let s ← getPair bs 33 34
  let e ← getPair bs 43 44
  let f ← getPair bs 52 53
  let g ← getPair bs 72 73
  let v ← (bs.get? 61).bind decodeBool
  let d : Draft := ⟨req,⟨a,s,e,f⟩,v,g⟩
  if body d = bs then some d else none

def fits (d : Draft) : Prop :=
  d.request < 256 ∧ d.addressed.account < 256 ∧ d.addressed.scope < 256 ∧
  d.addressed.entity < 256 ∧ d.addressed.field < 256 ∧ d.frontier < 256
instance (d : Draft) : Decidable (fits d) := inferInstanceAs
  (Decidable (d.request < 256 ∧ d.addressed.account < 256 ∧ d.addressed.scope < 256 ∧
  d.addressed.entity < 256 ∧ d.addressed.field < 256 ∧ d.frontier < 256))
def render (d : Draft) : Option Bytes :=
  if fits d then some (seal CLAIM (body d)) else none
def readFact (b : Bytes) : Option Draft := (openFrame CLAIM b).bind decodeBody

theorem body_length (d : Draft) : (body d).length = 74 := rfl
theorem body_roundtrip (d : Draft) (h : fits d) : decodeBody (body d) = some d := by
  cases d with
  | mk r k v g =>
    cases k with
    | mk a s e f =>
      unfold fits at h
      change (do
        let req ← decodePair (hi r) (lo r)
        let ca ← decodePair (hi a) (lo a)
        let cs ← decodePair (hi s) (lo s)
        let ce ← decodePair (hi e) (lo e)
        let cf ← decodePair (hi f) (lo f)
        let cg ← decodePair (hi g) (lo g)
        let cv ← decodeBool (valueByte v)
        let d : Draft := ⟨req,⟨ca,cs,ce,cf⟩,cv,cg⟩
        if body d = body ⟨r,⟨a,s,e,f⟩,v,g⟩ then some d else none) = _
      simp [pair_roundtrip r h.1,pair_roundtrip a h.2.1,
        pair_roundtrip s h.2.2.1,pair_roundtrip e h.2.2.2.1,
        pair_roundtrip f h.2.2.2.2.1,pair_roundtrip g h.2.2.2.2.2,bool_roundtrip]

theorem encoded_fact_exact (d : Draft) (b : Bytes) (h : render d = some b) :
    readFact b = some d ∧ parseRequest b = none := by
  by_cases good : fits d
  · have hb : seal CLAIM (body d) = b := Option.some.inj (by simpa [render,good] using h)
    cases hb
    have limit : (body d).length ≤ 96 := by rw [body_length]; decide
    constructor
    · unfold readFact; rw [sealed_opens CLAIM (body d) limit]
      exact body_roundtrip d good
    · simp [parseRequest,openFrame,seal,CLAIM,REQUEST]
  · simp [render,good] at h

theorem exactly_one_factual_record (a b : Draft) (wire : Bytes)
    (ha : render a = some wire) (hb : render b = some wire) : a = b :=
  Option.some.inj ((encoded_fact_exact a wire ha).1.symm.trans (encoded_fact_exact b wire hb).1)

#print axioms every_byte_rendered
#print axioms pair_roundtrip
#print axioms body_roundtrip
#print axioms encoded_fact_exact
#print axioms exactly_one_factual_record
end FactRenderer

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

-- Four small affine rows fitted to a synthetic CPU fixture.
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


namespace Composed
set_option maxRecDepth 16000
set_option maxHeartbeats 2500000
open ProtectedBinding ControlledBinding ByteProtocol FactRenderer StrictGrowth

structure EvidenceTick where
  token : Fin 5
  disturbance : Noise
def fitRun (x : Cell) : List EvidenceTick → Cell
  | [] => x
  | t::ts => fitRun (Fitted.step x t.token t.disturbance) ts
def refRun (q : Mode) : List EvidenceTick → Mode
  | [] => q
  | t::ts => refRun (Fitted.specToken q t.token) ts

theorem fit_run_preserves (ts : List EvidenceTick) (q : Mode) (x : Cell) (ok : inside q x) :
    inside (refRun q ts) (fitRun x ts) := by
  induction ts generalizing q x with
  | nil => exact ok
  | cons t _ ih => exact ih _ _ (Fitted.fitted_rows _ _ _ _ ok)

structure Record where
  addressed : Key
  frontier : Nat
  revision : Nat
  -- Authenticated typed evidence sequence, never derived from request prose.
  history : List EvidenceTick

structure Envelope where
  request : Nat
  frontier : Nat
  directory : List Entry
  ledger : List Key
  bank : List Record

def currentRevisions (env : Envelope) (key : Key) : List Record :=
  env.bank.filter (fun r => decide (r.addressed = key ∧ r.frontier = env.frontier))

def maximalRevisions (env : Envelope) (key : Key) : List Record :=
  let current := currentRevisions env key
  current.filter (fun r => current.all (fun s => decide (s.revision ≤ r.revision)))

def uniqueRecord (env : Envelope) (key : Key) : Option Record :=
  match maximalRevisions env key with
  | [x] => some x
  | _ => none

def actualLiveRead (env : Envelope) (key : Key) : Option FactValue := do
  let stored ← uniqueRecord env key
  let truth ← decode (fitRun 3 stored.history)
  pure ⟨truth,env.frontier⟩

def currentView (env : Envelope) : Current :=
  ⟨env.request,env.frontier,actualLiveRead env,fun k => decide (k ∈ env.ledger)⟩

def queryCover (env : Envelope) (ts : List Nat) : List (Option Key) :=
  scopedCover (currentView env) realQueryGrammar env.directory ts

-- Reflects the actual fitted state from the selected numeric register. No
-- reference decoder answers the factual query on behalf of the model.
def rawProposal (env : Envelope) (ts : List Nat) : Option Draft :=
  match queryCover env ts with
  | some k :: _ => propose (currentView env) k
  | _ => none


def legalForms : List (Bool × Bool × Bool) :=
  [(false,false,false),(false,false,true),(false,true,false),(false,true,true),
   (true,false,false),(true,false,true),(true,true,false),(true,true,true)]
def typedSurface (ts : List Nat) : Bool :=
  legalForms.any (fun f => decide (ts = requestTokens f.1 f.2.1 f.2.2))
def strictRequest (req : Bytes) : Option (List Nat) := do
  let ts ← parseRequest req
  if typedSurface ts then some ts else none
theorem strict_only_declared (req : Bytes) (ts : List Nat)
    (h : strictRequest req = some ts) :
    ∃ (s e f : Bool), req = requestBytes s e f ∧ ts = requestTokens s e f := by
  cases p : parseRequest req with
  | none => simp [strictRequest,p] at h
  | some t =>
    by_cases ok : typedSurface t = true
    · have same : t = ts := Option.some.inj (by simpa [strictRequest,p,ok] using h)
      cases same
      have ⟨form,_,hf⟩ := List.any_eq_true.mp ok
      have hf2 : ts = requestTokens form.1 form.2.1 form.2.2 := of_decide_eq_true hf
      have ⟨payload,exactly,encoding,_⟩ := accepted_frame_is_exactly_encoder _ _ p
      rw [hf2] at encoding
      refine ⟨form.1,form.2.1,form.2.2,?_,hf2⟩
      simp [requestBytes,encoding,exactly]
    · simp [strictRequest,p,ok] at h
theorem strict_all_declared : ∀ s e f : Bool,
    strictRequest (requestBytes s e f) = some (requestTokens s e f) := by decide

def protectedDecision (env : Envelope) (requestBytes : Bytes) : Option Draft := do
  let ts ← strictRequest requestBytes
  guarded (currentView env) realQueryGrammar env.directory ts (rawProposal env ts)

def reply (env : Envelope) (req : Bytes) : Option Bytes := do
  let approved ← protectedDecision env req
  render approved

theorem live_read_uses_fitted_run (env : Envelope) (key : Key) (val : FactValue)
    (h : actualLiveRead env key = some val) :
    ∃ row, uniqueRecord env key = some row ∧
      decode (fitRun 3 row.history) = some val.value ∧
      meaning (refRun 0 row.history) = some val.value ∧ val.frontier = env.frontier := by
  cases hs : uniqueRecord env key with
  | none => simp [actualLiveRead,hs] at h
  | some row =>
    cases hd : decode (fitRun 3 row.history) with
    | none => simp [actualLiveRead,hs,hd] at h
    | some b =>
      have hv : (⟨b,env.frontier⟩ : FactValue) = val :=
        Option.some.inj (by simpa [actualLiveRead,hs,hd] using h)
      cases hv
      have hh := readout _ _ (fit_run_preserves row.history 0 3 init_inside)
      rw [hd] at hh
      exact ⟨row,rfl,hd,hh.symm,rfl⟩

theorem decision_decodes_request (env : Envelope) (req : Bytes) (d : Draft)
    (t : protectedDecision env req = some d) :
    ∃ ts, strictRequest req = some ts ∧
       guarded (currentView env) realQueryGrammar env.directory ts (rawProposal env ts) = some d := by
  cases hs : strictRequest req with
  | none => simp [protectedDecision,hs] at t
  | some ts => exact ⟨ts,rfl,by simpa [protectedDecision,hs] using t⟩

theorem composed_truth_and_authority (env : Envelope) (req wire : Bytes)
    (h : reply env req = some wire) :
    ∃ ts d row, strictRequest req = some ts ∧
      readFact wire = some d ∧ parseRequest wire = none ∧
      rawProposal env ts = some d ∧ d.addressed ∈ env.ledger ∧
      uniqueRecord env d.addressed = some row ∧
      decode (fitRun 3 row.history) = some d.value ∧
      meaning (refRun 0 row.history) = some d.value ∧
      ∀ a, Meaning (currentView env) realQueryGrammar env.directory ts a → a = some d.addressed := by
  cases hd : protectedDecision env req with
  | none => simp [reply,hd] at h
  | some d =>
    have hr : render d = some wire := by simpa [reply,hd] using h
    have ⟨ts,hlex,hg⟩ := decision_decodes_request env req d hd
    have hu := unchanged_or_reject (currentView env) (queryCover env ts) (rawProposal env ts) d hg
    have l := (Bool.and_eq_true_iff _ _).mp hu.2.2
    have lg := (Bool.and_eq_true_iff _ _).mp l.1
    have permit : d.addressed ∈ env.ledger := of_decide_eq_true lg.2
    have hv : actualLiveRead env d.addressed = some ⟨d.value,env.frontier⟩ := of_decide_eq_true l.2
    have ⟨row,choose,numeric,ref,_⟩ := live_read_uses_fitted_run env d.addressed _ hv
    exact ⟨ts,d,row,hlex,(encoded_fact_exact d wire hr).1,(encoded_fact_exact d wire hr).2,
      hu.1,permit,choose,numeric,ref,fun a ha => (audited_binding _ _ _ _ _ _ _ ha hg).2.1⟩

-- Streams are separated before content is inspected. An untrusted prose
-- frame is inert; only delivering request chunks can change its buffer.
inductive StreamInput where
  | requestChunk (bs : Bytes)
  | prose (bs : Bytes)

def push (s : Buffered) : StreamInput → Buffered
  | .requestChunk bs => feed s bs
  | .prose _ => s

def publish (env : Envelope) (s : Buffered) : Option Bytes := reply env s.input

theorem arbitrary_prose_is_inert (s : Buffered) (p q : Bytes) :
    push s (.prose p) = push s (.prose q) := rfl
theorem same_ordered_request_chunking (env : Envelope) (s : Buffered) (a b text : Bytes) :
    publish env (push (push (push s (.requestChunk a)) (.prose text)) (.requestChunk b)) =
    publish env (push s (.requestChunk (a ++ b))) := by
  change publish env (feed (feed s a) b) = publish env (feed s (a ++ b))
  rw [feed_chunking]

def authA : Key := ⟨1,40,7,1⟩
def sameValueUnauthorized : Key := ⟨1,41,8,1⟩

def entryA : Entry := ⟨101,201,301,authA,20⟩
def entryB : Entry := ⟨102,201,301,sameValueUnauthorized,20⟩
def duplicateAlias : Entry := ⟨101,201,301,sameValueUnauthorized,20⟩

def hold : EvidenceTick := ⟨2,1⟩
def flip : EvidenceTick := ⟨0,0⟩
def reset : EvidenceTick := ⟨4,1⟩
def authored (b : Bool) : EvidenceTick := ⟨if b then 1 else 3,1⟩
def track (b : Bool) : List EvidenceTick := [authored b,hold,flip,hold,flip,hold]

def fixture (b : Bool) : Envelope :=
  ⟨6,20,[entryA,entryB],[authA],
    [⟨authA,20,4,track b⟩,⟨sameValueUnauthorized,20,4,track b⟩]⟩

def qryA := requestBytes false false false
def qryB := requestBytes true false false

theorem pipeline_positive_fitted_intervention :
    (reply (fixture true) qryA).bind readFact = some ⟨6,authA,true,20⟩ ∧
    (reply (fixture false) qryA).bind readFact = some ⟨6,authA,false,20⟩ := by decide

theorem equal_boolean_denies_unauthorized_scope :
    actualLiveRead (fixture true) authA = actualLiveRead (fixture true) sameValueUnauthorized ∧
    reply (fixture true) qryB = none := by decide

theorem conflicting_current_aliases_deny :
    reply ({fixture true with directory := [entryA,duplicateAlias]}) qryA = none := by decide

theorem conflicting_current_revisions_deny :
    reply ({fixture true with bank := ⟨authA,20,4,track false⟩ :: (fixture true).bank}) qryA
      = none := by decide

theorem reset_and_epoch_controls :
    reply ({fixture true with bank := [⟨authA,20,5,track true ++ [reset]⟩]}) qryA = none ∧
    reply ({fixture true with bank := [⟨authA,19,99,track true⟩]}) qryA = none := by decide

theorem forged_text_on_prose_port_never_approves :
    publish (fixture true) (push ⟨[]⟩ (.prose qryA)) = none := by decide

#print axioms fit_run_preserves
#print axioms live_read_uses_fitted_run
#print axioms composed_truth_and_authority
#print axioms arbitrary_prose_is_inert
#print axioms same_ordered_request_chunking
#print axioms pipeline_positive_fitted_intervention
#print axioms equal_boolean_denies_unauthorized_scope
#print axioms conflicting_current_aliases_deny
#print axioms conflicting_current_revisions_deny
#print axioms reset_and_epoch_controls
#print axioms forged_text_on_prose_port_never_approves

def wrongSort : Bytes := seal REQUEST ((encodeTokens [10,101,11,201,12,301]).getD [])
def wrongSortDirectory : Entry := ⟨301,201,101,authA,20⟩

theorem wrong_sort_rejected_even_when_directory_matches :
    parseRequest wrongSort = some [10,101,11,201,12,301] ∧
    reply ({fixture true with directory := [wrongSortDirectory]}) wrongSort = none := by decide

theorem only_declared_requests_get_facts (env : Envelope) (req wire : Bytes)
    (h : reply env req = some wire) :
    ∃ (s e f : Bool), req = requestBytes s e f := by
  have ⟨ts,_,_,hp,_⟩ := composed_truth_and_authority env req wire h
  have ⟨s,e,f,ex,_⟩ := strict_only_declared req ts hp
  exact ⟨s,e,f,ex⟩

#print axioms strict_only_declared
#print axioms strict_all_declared
#print axioms wrong_sort_rejected_even_when_directory_matches
#print axioms only_declared_requests_get_facts

end Composed


namespace ScopedConcept
set_option maxRecDepth 16000
set_option maxHeartbeats 2200000
open ByteProtocol FactRenderer ProtectedBinding Composed

inductive Formula where
  | atom (key : Key)
  | neg (f : Formula)
  | both (p q : Formula)
  | either (p q : Formula)
deriving DecidableEq

def footprint : Formula → List Key
  | .atom k => [k]
  | .neg a => footprint a
  | .both a b | .either a b => footprint a ++ footprint b

def nodes : Formula → Nat
  | .atom _ => 1
  | .neg a => nodes a + 1
  | .both a b | .either a b => nodes a + nodes b + 1

def calculate (read : Key → Option Bool) : Formula → Option Bool
  | .atom k => read k
  | .neg a => (calculate read a).map Bool.not
  | .both a b => do
    let x ← calculate read a
    let y ← calculate read b
    pure (x && y)
  | .either a b => do
    let x ← calculate read a
    let y ← calculate read b
    pure (x || y)

def denote (interpret : Key → Bool) : Formula → Bool
  | .atom k => interpret k
  | .neg a => !(denote interpret a)
  | .both a b => denote interpret a && denote interpret b
  | .either a b => denote interpret a || denote interpret b

theorem calculation_never_invents (read : Key → Option Bool) (world : Key → Bool)
    (sound : ∀ k v, read k = some v → world k = v) (f : Formula) (value : Bool)
    (h : calculate read f = some value) :
    denote world f = value := by
  induction f generalizing value with
  | atom k => exact sound k value h
  | neg a ih =>
    cases hw : calculate read a with
    | none => simp [calculate,hw] at h
    | some v =>
      have hm := ih v hw
      have hh : (!v) = value := Option.some.inj (by simpa [calculate,hw] using h)
      simp [denote,hm,hh]
  | both a b ih jh =>
    cases h1 : calculate read a with
    | none => simp [calculate,h1] at h
    | some v =>
      cases h2 : calculate read b with
      | none => simp [calculate,h1,h2] at h
      | some w =>
        have hh : (v && w) = value := Option.some.inj (by simpa [calculate,h1,h2] using h)
        simp [denote,ih v h1,jh w h2,hh]
  | either a b ih jh =>
    cases h1 : calculate read a with
    | none => simp [calculate,h1] at h
    | some v =>
      cases h2 : calculate read b with
      | none => simp [calculate,h1,h2] at h
      | some w =>
        have hh : (v || w) = value := Option.some.inj (by simpa [calculate,h1,h2] using h)
        simp [denote,ih v h1,jh w h2,hh]

-- A separately authenticated declaration must supply this exact immutable
-- AST, owner and version. This reference does not implement authentication.
structure Declaration where
  account : Nat
  scope : Nat
  symbol : Nat
  version : Nat
  formula : Formula
deriving DecidableEq

structure MacroContext where
  env : Envelope
  account : Nat
  scope : Nat
  definitionVersion : Nat
  definitions : List Declaration
  -- The host authenticates authority to *register a definition*. It is not
  -- permission to read any fact, and cannot mutate the fact ledger.
  mayDeclare : Bool

def sameAddress (u : MacroContext) (m : Nat) (d : Declaration) : Bool :=
  decide (d.account = u.account ∧ d.scope = u.scope ∧
    d.symbol = m ∧ d.version = u.definitionVersion)
def candidates (u : MacroContext) (m : Nat) : List Declaration :=
  u.definitions.filter (sameAddress u m)

def select (u : MacroContext) (m : Nat) : Option Declaration :=
  match candidates u m with
  | [d] => some d
  | _ => none

-- Boolean equivalence is not authorization equivalence. A static footprint
-- is checked in full before any reads, including under OR or NOT.
def ready (u : MacroContext) (f : Formula) : Bool :=
  decide (nodes f ≤ 16) && (footprint f).all
    (fun k => decide (k.account = u.account ∧ k.scope = u.scope ∧ k ∈ u.env.ledger))

def install (u : MacroContext) (approved tag : Declaration) : MacroContext :=
  if u.mayDeclare && decide (tag = approved) && decide (tag.symbol < 256) && sameAddress u tag.symbol tag &&
      decide (nodes tag.formula ≤ 16) && (candidates u tag.symbol).isEmpty
  then {u with definitions := tag :: u.definitions}
  else u

theorem install_never_writes_ledger (u : MacroContext) (approved d : Declaration) :
    (install u approved d).env.ledger = u.env.ledger := by
  unfold install
  split <;> rfl

def readFit (env : Envelope) (k : Key) : Option Bool :=
  (actualLiveRead env k).map FactValue.value

structure Result where
  account : Nat
  scope : Nat
  symbol : Nat
  version : Nat
  frontier : Nat
  payload : Formula
  value : Bool
deriving DecidableEq

def execute (u : MacroContext) (m : Nat) : Option Result := do
  let d ← select u m
  if !ready u d.formula then none else
    let v ← calculate (readFit u.env) d.formula
    pure ⟨d.account,d.scope,d.symbol,d.version,u.env.frontier,d.formula,v⟩

theorem evaluated_under_current_footprint (u : MacroContext) (m : Nat) (r : Result)
    (h : execute u m = some r) :
    ∃ d, select u m = some d ∧ r.payload = d.formula ∧ r.frontier = u.env.frontier ∧
      calculate (readFit u.env) d.formula = some r.value ∧
      ∀ k, k ∈ footprint d.formula → k.account = u.account ∧ k.scope = u.scope ∧ k ∈ u.env.ledger := by
  cases hs : select u m with
  | none => simp [execute,hs] at h
  | some d =>
    by_cases hr : ready u d.formula = true
    · cases hc : calculate (readFit u.env) d.formula with
      | none => simp [execute,hs,hr,hc] at h
      | some v =>
        have hx : (⟨d.account,d.scope,d.symbol,d.version,u.env.frontier,d.formula,v⟩ : Result) = r :=
          Option.some.inj (by simpa [execute,hs,hr,hc] using h)
        cases hx
        refine ⟨d,rfl,rfl,rfl,hc,?_⟩
        intro k member
        have hf := (Bool.and_eq_true_iff _ _).mp hr
        exact of_decide_eq_true (List.all_eq_true.mp hf.2 k member)
    · have hf : ready u d.formula = false := (Bool.not_eq_true _).mp hr
      simp [execute,hs,hf] at h

-- Two namespaces, with no fallback from malformed legacy payload to a macro.
def CONCEPT : Nat := 211
def encodeConcept (m : Nat) : Bytes := seal CONCEPT [hi m,lo m]
def conceptRequest (p : Bytes) : Option Nat := do
  let bytes ← openFrame CONCEPT p
  match bytes with
  | [h,l] => decodePair h l
  | _ => none

inductive Output where
  | literal (wire : Bytes)
  | derived (r : Result)

def serve (u : MacroContext) (p : Bytes) : Option Output :=
  if p.head? = some CONCEPT then
    (conceptRequest p |>.bind (execute u)).map Output.derived
  else (Composed.reply u.env p).map Output.literal

theorem legacy_wire_unchanged_under_any_registry (u : MacroContext) (new : List Declaration)
    (p : Bytes) (notMacro : p.head? ≠ some CONCEPT) :
    serve {u with definitions := new} p = serve u p := by simp [serve,notMacro]

def aKey : Key := authA
def bKey : Key := ⟨1,40,7,2⟩
def privateKey : Key := ⟨2,40,7,3⟩
def tested : Formula := .both (.atom aKey) (.neg (.atom bKey))
def publicFalse : Formula := .either (.atom aKey) (.atom privateKey)
def declaredExample : Declaration := ⟨1,40,33,9,tested⟩
def priorContext (a b : Bool) : MacroContext :=
  let env := fixture a
  let changed : Envelope := { env with
    ledger := [aKey,bKey]
    bank := ⟨bKey,20,3,track b⟩ :: env.bank }
  ⟨changed,1,40,9,[],true⟩
def newContext (a b : Bool) : MacroContext := install (priorContext a b) declaredExample declaredExample
def surface := encodeConcept 33
def resultValue : Output → Option Bool
  | .derived r => some r.value
  | .literal _ => none
def observed (u : MacroContext) (p : Bytes) : Option Bool := (serve u p).bind resultValue

theorem strict_new_semantic_coverage :
    observed (priorContext true false) surface = none ∧
    observed (newContext true false) surface = some true ∧
    observed (newContext true true) surface = some false ∧
    (newContext true false).env.ledger = (priorContext true false).env.ledger := by decide

theorem short_circuit_cannot_read_private_dependency :
    (priorContext true false).env.ledger.contains aKey = true ∧
    (priorContext true false).env.ledger.contains privateKey = false ∧
    execute {priorContext true false with definitions := [⟨1,40,33,9,publicFalse⟩]} 33 = none := by decide

theorem conflicts_and_version_deny :
    observed {newContext true false with definitionVersion := 10} surface = none ∧
    observed {newContext true false with definitions := [declaredExample,declaredExample]} surface = none ∧
    observed (install {priorContext true false with mayDeclare := false} declaredExample declaredExample) surface = none := by decide

#print axioms calculation_never_invents
#print axioms install_never_writes_ledger
#print axioms evaluated_under_current_footprint
#print axioms legacy_wire_unchanged_under_any_registry
#print axioms strict_new_semantic_coverage
#print axioms short_circuit_cannot_read_private_dependency
#print axioms conflicts_and_version_deny

theorem unauthorized_substitution_does_not_install (u : MacroContext) (a d : Declaration)
    (changed : d ≠ a) : install u a d = u := by simp [install,changed]

theorem old_protocol_unchanged_by_real_install (u : MacroContext) (approved d : Declaration)
    (p : Bytes) (notMacro : p.head? ≠ some CONCEPT) :
    serve (install u approved d) p = serve u p := by
  unfold install
  split
  · exact legacy_wire_unchanged_under_any_registry u (d :: u.definitions) p notMacro
  · rfl

-- Relative to the fixture's old one-atom query language, the new concept
-- cannot be implemented by selecting a different single legacy atom.
theorem derived_value_not_either_atomic_projection :
    ∀ chooseA : Bool, ∃ a b : Bool,
      (if chooseA then a else b) ≠ (a && !b) := by decide

theorem bound_exact_proposal_mutation_rejected :
   observed
      (install (priorContext true false) declaredExample
        {declaredExample with formula := .neg tested}) surface = none := by decide

#print axioms unauthorized_substitution_does_not_install
#print axioms old_protocol_unchanged_by_real_install
#print axioms derived_value_not_either_atomic_projection
#print axioms bound_exact_proposal_mutation_rejected


theorem cannot_install_unaddressable (u : MacroContext) (a d : Declaration)
    (outside : ¬ d.symbol < 256) : install u a d = u := by simp [install,outside]

theorem every_symbol_has_exact_wire : ∀ n : Fin 256,
    conceptRequest (encodeConcept n.val) = some n.val := by decide

#print axioms cannot_install_unaddressable
#print axioms every_symbol_has_exact_wire

end ScopedConcept
