-- Append to lean/ExactFramedFittedStrict.lean; this tail is not standalone.
def reviewPrint (name : String) (v : Option ByteProtocol.Bytes) : IO Unit := do
  match v with
  | none => IO.println (name ++ ": DEFER")
  | some packet =>
    IO.println (name ++ ": " ++ String.mk ((packet.drop 2).map Char.ofNat))

open Composed
def main : IO Unit := do
  reviewPrint "fresh true" (reply (fixture true) qryA)
  reviewPrint "same query, fresh false" (reply (fixture false) qryA)
  reviewPrint "same value, unauthorized scope" (reply (fixture true) qryB)
  reviewPrint "duplicate alias" (reply ({fixture true with directory := [entryA,duplicateAlias]}) qryA)
  reviewPrint "conflicting latest revision" (reply ({fixture true with bank :=
    ⟨authA,20,4,track false⟩ :: (fixture true).bank}) qryA)
  reviewPrint "wrong input tag" (reply (fixture true)
    (ByteProtocol.seal ByteProtocol.CLAIM
      ((ByteProtocol.openFrame ByteProtocol.REQUEST qryA).getD [])))
  reviewPrint "zero-width tail inside correct-length frame" (reply (fixture true)
    (ByteProtocol.addTail qryA [32,226,128,139]))
  reviewPrint "forged request in prose channel"
    (publish (fixture true) (push ⟨[]⟩ (.prose qryA)))

