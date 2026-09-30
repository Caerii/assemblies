import AssemblyIR.Refinement
import Std.Tactic

/-!
# Pure explicit-round semantics and lowering

Specification: neural_assemblies/ir/VERIFICATION.md#contract-formal-explicit-round

This model fixes integer addition and instruction order while keeping winner
selection and the per-fiber learning transform explicit dependencies. It proves
the field mapping into a dense-kernel instruction and observable frame laws. It
does not model JSON parsing, NumPy float32 arithmetic, clipping or exceptions.
-/
namespace AssemblyIR.Projection

/-- The normalized fields shared with `explicit-area-round-v1`. -/
structure Round (Area : Type) where
  target : Area
  fromAreas : List Area
  plasticity : Bool
  externalDrive : List Int
  deriving DecidableEq, Repr

/-- A backend-facing instruction with deliberately different field names. -/
structure DenseKernelRound (Area : Type) where
  destination : Area
  sources : List Area
  learns : Bool
  bias : List Int
  deriving DecidableEq, Repr

/-- Pure state needed by one dense round. Neuron identifiers are natural numbers. -/
structure State (Area : Type) where
  registered : Area -> Bool
  population : Area -> Nat
  capSize : Area -> Nat
  winners : Area -> List Nat
  weights : Area -> Area -> Nat -> Nat -> Int

variable {Area : Type} [DecidableEq Area]

/-- Sum source rows in declared source order, after the external bias. -/
def driveAt (state : State Area) (target : Area) (sources : List Area)
    (externalDrive : List Int) (neuron : Nat) : Int :=
  sources.foldl
    (fun total source =>
      (state.winners source).foldl
        (fun subtotal pre => subtotal + state.weights source target pre neuron)
        total)
    (externalDrive.getD neuron 0)

/-- Scores are presented to the selector in increasing neuron-ID order. -/
def scoreVector (state : State Area) (round : Round Area) : List Int :=
  (List.range (state.population round.target)).map
    (driveAt state round.target round.fromAreas round.externalDrive)

/-- Executable universal quantification over a finite list. -/
private def allList {α : Type} (predicate : α -> Bool) : List α -> Bool
  | [] => true
  | value :: rest => predicate value && allList predicate rest

/-- Executable duplicate rejection using the declared equality decision. -/
private def distinctList {α : Type} [DecidableEq α] : List α -> Bool
  | [] => true
  | value :: rest => (!decide (value ∈ rest)) && distinctList rest

/-- State-dependent admission checks the wire invariants and selector output. -/
def Valid (choose : Nat -> List Int -> List Nat)
    (state : State Area) (round : Round Area) : Bool :=
  let population := state.population round.target
  let selected := choose (state.capSize round.target) (scoreVector state round)
  state.registered round.target &&
    distinctList round.fromAreas &&
    (!round.fromAreas.isEmpty || !round.externalDrive.isEmpty) &&
    (round.externalDrive.isEmpty || decide (round.externalDrive.length = population)) &&
    decide (state.capSize round.target ≤ population) &&
    distinctList selected &&
    decide (selected.length = state.capSize round.target) &&
    allList (fun neuron => decide (neuron < population)) selected &&
    allList (fun source =>
      state.registered source &&
      distinctList (state.winners source) &&
      allList (fun neuron => decide (neuron < state.population source))
        (state.winners source)) round.fromAreas

/-- One semantic definition used by both the normalized and lowered instruction. -/
def applyFields
    (choose : Nat -> List Int -> List Nat)
    (learn : Area -> Area -> Int -> Int)
    (target : Area) (sources : List Area) (plasticity : Bool)
    (externalDrive : List Int) (state : State Area) : State Area :=
  let scores := (List.range (state.population target)).map
    (driveAt state target sources externalDrive)
  let selected := choose (state.capSize target) scores
  { state with
    winners := fun area => if area = target then selected else state.winners area
    weights := fun source destination pre post =>
      if plasticity = true ∧ source ∈ sources ∧ destination = target ∧
          pre ∈ state.winners source ∧ post ∈ selected then
        learn source destination (state.weights source destination pre post)
      else
        state.weights source destination pre post }

def roundStep
    (choose : Nat -> List Int -> List Nat)
    (learn : Area -> Area -> Int -> Int) (round : Round Area) :
    State Area -> State Area :=
  applyFields choose learn round.target round.fromAreas round.plasticity
    round.externalDrive

/-- Reject malformed instructions and state before applying a one-round mutation. -/
def checkedRound
    (choose : Nat -> List Int -> List Nat)
    (learn : Area -> Area -> Int -> Int) (round : Round Area)
    (state : State Area) : Option (State Area) :=
  if Valid choose state round = true then some (roundStep choose learn round state) else none

/-- Checked execution is neither more nor less permissive than `Valid`. -/
theorem checkedRound_iff
    (choose : Nat -> List Int -> List Nat)
    (learn : Area -> Area -> Int -> Int) (round : Round Area)
    (state result : State Area) :
    checkedRound choose learn round state = some result ↔
      Valid choose state round = true ∧ roundStep choose learn round state = result := by
  simp [checkedRound]

def denseKernelStep
    (choose : Nat -> List Int -> List Nat)
    (learn : Area -> Area -> Int -> Int) (round : DenseKernelRound Area) :
    State Area -> State Area :=
  applyFields choose learn round.destination round.sources round.learns round.bias

/-- The compiler bridge maps fields; it does not choose hidden defaults. -/
def lowerRound (round : Round Area) : List (DenseKernelRound Area) :=
  [{ destination := round.target
     sources := round.fromAreas
     learns := round.plasticity
     bias := round.externalDrive }]

/-- The concrete pure lowering discharges the reusable local simulation rule. -/
theorem lowerRound_simulates
    (choose : Nat -> List Int -> List Nat)
    (learn : Area -> Area -> Int -> Int) :
    Simulates (roundStep choose learn) (denseKernelStep choose learn)
      lowerRound Eq := by
  intro round source target related
  subst target
  rfl

/-- Consequently the field lowering preserves every finite sequential program. -/
theorem lowerRound_program
    (choose : Nat -> List Int -> List Nat)
    (learn : Area -> Area -> Int -> Int)
    (program : List (Round Area)) (state : State Area) :
    run (roundStep choose learn) program state =
      run (denseKernelStep choose learn) (lower lowerRound program) state := by
  exact lower_preserves _ _ _ Eq (lowerRound_simulates choose learn)
    program state state rfl

/-- Equal final states give equal winner observations for every named area. -/
theorem lowerRound_winners
    (choose : Nat -> List Int -> List Nat)
    (learn : Area -> Area -> Int -> Int)
    (program : List (Round Area)) (state : State Area) (area : Area) :
    (run (roundStep choose learn) program state).winners area =
      (run (denseKernelStep choose learn) (lower lowerRound program) state).winners area := by
  rw [lowerRound_program]

/-- A frozen instruction leaves every weight unchanged. -/
theorem frozen_weight_frame
    (choose : Nat -> List Int -> List Nat)
    (learn : Area -> Area -> Int -> Int) (round : Round Area)
    (state : State Area) (frozen : round.plasticity = false)
    (source destination : Area) (pre post : Nat) :
    (roundStep choose learn round state).weights source destination pre post =
      state.weights source destination pre post := by
  simp [roundStep, applyFields, frozen]

/-- A round cannot change winners in an area it does not target. -/
theorem other_area_winner_frame
    (choose : Nat -> List Int -> List Nat)
    (learn : Area -> Area -> Int -> Int) (round : Round Area)
    (state : State Area) (area : Area) (other : Not (area = round.target)) :
    (roundStep choose learn round state).winners area = state.winners area := by
  simp [roundStep, applyFields, other]

/-- The target observation is exactly the configured selector's result. -/
theorem target_winners
    (choose : Nat -> List Int -> List Nat)
    (learn : Area -> Area -> Int -> Int) (round : Round Area)
    (state : State Area) :
    (roundStep choose learn round state).winners round.target =
      choose (state.capSize round.target) (scoreVector state round) := by
  simp [roundStep, applyFields, scoreVector]

-- Constructed positive and negative controls on a two-neuron target.
private def exampleState : State Bool :=
  { registered := fun _ => true
    population := fun area => if area then 2 else 1
    capSize := fun _ => 1
    winners := fun area => if area then [] else [0]
    weights := fun source destination pre post =>
      if source = false ∧ destination = true ∧ pre = 0 ∧ post = 0 then 3 else 0 }

private def exampleRound : Round Bool :=
  { target := true
    fromAreas := [false]
    plasticity := true
    externalDrive := [1, 0] }

private def chooseFirst (_ : Nat) (_ : List Int) : List Nat := [0]
private def increment (_ _ : Bool) (weight : Int) : Int := weight + 1

example : scoreVector exampleState exampleRound = [4, 0] := by decide
example : Valid chooseFirst exampleState exampleRound = true := by decide
example : (checkedRound chooseFirst increment exampleRound exampleState).isSome = true := by
  decide
example :
    (roundStep chooseFirst increment exampleRound exampleState).weights false true 0 0 = 4 :=
  by decide

/-- A broken bridge that drops learning is distinguished by the control. -/
private def lowerWithoutLearning (round : Round Bool) : DenseKernelRound Bool :=
  { destination := round.target
    sources := round.fromAreas
    learns := false
    bias := round.externalDrive }

example :
    (denseKernelStep chooseFirst increment (lowerWithoutLearning exampleRound)
      exampleState).weights false true 0 0 = 3 := by decide

private def shortDrive : Round Bool := { exampleRound with externalDrive := [1] }
private def duplicateSource : Round Bool :=
  { exampleRound with fromAreas := [false, false] }
private def invalidSelector (_ : Nat) (_ : List Int) : List Nat := [2]

example : checkedRound chooseFirst increment shortDrive exampleState = none := by decide
example : checkedRound chooseFirst increment duplicateSource exampleState = none := by decide
example : checkedRound invalidSelector increment exampleRound exampleState = none := by decide

#print axioms lowerRound_simulates
#print axioms lowerRound_program
#print axioms lowerRound_winners
#print axioms checkedRound_iff
#print axioms frozen_weight_frame
#print axioms other_area_winner_frame
#print axioms target_winners

end AssemblyIR.Projection
