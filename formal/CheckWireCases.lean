import AssemblyIR.Wire

/-! Execute Lean admission against the cross-language explicit-round corpus. -/
open Lean
open AssemblyIR.Wire

private def checkCase (entry : Json) : Except String (String × Bool) := do
  let name ← (← entry.getObjVal? "name").getStr?
  let expected ← (← entry.getObjVal? "valid").getBool?
  let admitted ←
    match entry.getObjVal? "raw_json" with
    | .ok raw =>
        match raw.getStr? with
        | .ok source => pure (Json.parse source >>= decodeRound)
        | .error message => throw message
    | .error _ => pure (decodeRound (← entry.getObjVal? "document"))
  return (name, admitted.isOk == expected)

def main (arguments : List String) : IO Unit := do
  let path ← match arguments with
    | [path] => pure path
    | _ => throw <| IO.userError "usage: check-wire-cases <explicit-round.cases.json>"
  let source ← IO.FS.readFile path
  let json ← IO.ofExcept (Json.parse source)
  let entries ← IO.ofExcept json.getArr?
  let checks ← IO.ofExcept (entries.toList.mapM checkCase)
  let failed := checks.filterMap fun (name, passed) => if passed then none else some name
  if !failed.isEmpty then
    throw <| IO.userError s!"explicit-round corpus mismatches: {failed}"
  IO.println s!"Lean accepted explicit-round corpus: {checks.length} cases"
