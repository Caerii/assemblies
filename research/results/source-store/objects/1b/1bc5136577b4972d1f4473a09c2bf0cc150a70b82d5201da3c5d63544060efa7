import AssemblyIR.Projection
import Lean.Data.Json
import Std.Tactic

/-!
# JSON wire admission for the explicit round

Specification: neural_assemblies/ir/VERIFICATION.md#contract-formal-round-wire

This decoder owns the Lean interpretation of `explicit-area-round-v1`. JSON
numbers remain exact decimal values until an explicit scale converts them to the
integer semantics in `AssemblyIR.Projection`. No floating-point backend claim is
made here.
-/
namespace AssemblyIR.Wire

open Lean
open AssemblyIR.Projection

structure WireRound where
  target : String
  fromAreas : List String
  plasticity : Bool
  externalDrive : List JsonNumber
  deriving Repr

private def allowedKeys : List String :=
  ["profile", "target", "from_areas", "plasticity", "external_drive"]

private def distinctStrings : List String -> Bool
  | [] => true
  | value :: rest => (!decide (value ∈ rest)) && distinctStrings rest

def decodeRound (json : Json) : Except String WireRound := do
  let object ← json.getObj?
  if object.size != allowedKeys.length ||
      !object.keys.all (fun key => decide (key ∈ allowedKeys)) then
    throw "explicit round has missing or unknown fields"
  let profile ← (← json.getObjVal? "profile").getStr?
  if profile != "explicit-area-round-v1" then
    throw "unsupported explicit round profile"
  let target ← (← json.getObjVal? "target").getStr?
  if target.isEmpty then
    throw "target must be a nonempty area name"
  let sourceValues ← (← json.getObjVal? "from_areas").getArr?
  let sources ← sourceValues.toList.mapM Json.getStr?
  if sources.any String.isEmpty then
    throw "sources must be nonempty area names"
  if !distinctStrings sources then
    throw "duplicate sources would count drive and learning twice"
  let plasticity ← (← json.getObjVal? "plasticity").getBool?
  let driveValues ← (← json.getObjVal? "external_drive").getArr?
  let drive ← driveValues.toList.mapM Json.getNum?
  if sources.isEmpty && drive.isEmpty then
    throw "a round needs area input or explicit external drive"
  return { target := target, fromAreas := sources, plasticity := plasticity,
           externalDrive := drive }

/-- Convert an exact JSON decimal to an integer at `10^scale` units.
    Reject rather than round when the decimal is not representable. -/
def scaleNumber (scale : Nat) (number : JsonNumber) : Except String Int :=
  if number.exponent ≤ scale then
    .ok (number.mantissa * (10 : Int) ^ (scale - number.exponent))
  else
    let divisor := (10 : Int) ^ (number.exponent - scale)
    if number.mantissa % divisor = 0 then
      .ok (number.mantissa / divisor)
    else
      .error s!"decimal {number.toString} is not exact at scale {scale}"

def normalizeRound (scale : Nat) (wire : WireRound) :
    Except String (Round String) :=
  match wire.externalDrive.mapM (scaleNumber scale) with
  | .error message => .error message
  | .ok drive => .ok {
      target := wire.target, fromAreas := wire.fromAreas,
      plasticity := wire.plasticity, externalDrive := drive }

/-- Normalization changes only the numerical representation of external drive. -/
theorem normalizeRound_identity (scale : Nat) (wire : WireRound)
    (round : Round String) (accepted : normalizeRound scale wire = .ok round) :
    round.target = wire.target ∧ round.fromAreas = wire.fromAreas ∧
      round.plasticity = wire.plasticity := by
  cases converted : wire.externalDrive.mapM (scaleNumber scale) with
  | error message => simp [normalizeRound, converted] at accepted
  | ok drive =>
      simp [normalizeRound, converted] at accepted
      subst round
      simp

-- Positive exact-scaling control. Decoder cases live in the shared JSON corpus.
private def driveOnlyWire : WireRound :=
  { target := "T", fromAreas := [], plasticity := false,
    externalDrive := [
      { mantissa := 0, exponent := 0 },
      { mantissa := -15, exponent := 1 },
      { mantissa := 2, exponent := 0 }] }

private def driveOnlyNormalizes : Bool :=
  match normalizeRound 1 driveOnlyWire with
  | .ok round => decide (
      round.target = "T" ∧ round.fromAreas = [] ∧
      round.plasticity = false ∧ round.externalDrive = [0, -15, 20])
  | .error _ => false

example : driveOnlyNormalizes = true := by
  native_decide

-- Constructed lossy-normalization control.
example : (scaleNumber 1 { mantissa := 125, exponent := 2 }).isOk = false := by
  native_decide

#print axioms normalizeRound_identity

end AssemblyIR.Wire
