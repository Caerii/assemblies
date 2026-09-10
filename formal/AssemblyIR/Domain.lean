import AssemblyIR.Refinement

/-!
# Checked domain execution

Specification: neural_assemblies/ir/VERIFICATION.md#contract-checked-domain

A domain supplies its transition, invariant and state-dependent admissibility
rule once. The checked interpreter and its proof use those same definitions.
This pure model does not wrap or certify an effectful backend.
-/
namespace AssemblyIR

universe u v

/-- Domain obligations are data and proofs, not a parallel handwritten program. -/
structure Domain (Op : Type u) (S : Type v) where
  step : Op -> S -> S
  invariant : S -> Prop
  valid : Op -> S -> Prop
  decideValid : (op : Op) -> (state : S) -> Decidable (valid op state)
  preserves : forall op state, invariant state -> valid op state ->
    invariant (step op state)

/-- Check each precondition against the state at that instruction boundary.
`none` exports no final state; this is not rollback of external side effects. -/
def Domain.execute {Op : Type u} {S : Type v} (domain : Domain Op S) :
    List Op -> S -> Option S
  | [], state => some state
  | op :: rest, state =>
    if @decide (domain.valid op state) (domain.decideValid op state) then
      domain.execute rest (domain.step op state)
    else none

/-- Admissibility includes every intermediate state, not only the initial one. -/
def Domain.admissible {Op : Type u} {S : Type v} (domain : Domain Op S) :
    List Op -> S -> Prop
  | [], _ => True
  | op :: rest, state =>
    domain.valid op state ∧ domain.admissible rest (domain.step op state)

/-- Exactly admissible schedules execute, with the ordinary interpreter's result. -/
theorem Domain.execute_iff {Op : Type u} {S : Type v} (domain : Domain Op S)
    (program : List Op) (state result : S) :
    domain.execute program state = some result ↔
      domain.admissible program state ∧ run domain.step program state = result := by
  induction program generalizing state with
  | nil => simp [execute, admissible, run]
  | cons op rest ih =>
    letI := domain.decideValid op state
    by_cases valid : domain.valid op state
    · simpa [execute, admissible, run, valid] using ih (domain.step op state)
    · simp [execute, admissible, valid]

/-- Conditional domain preservation lifts to every admitted schedule. -/
theorem Domain.admissible_preserves {Op : Type u} {S : Type v}
    (domain : Domain Op S) (program : List Op) (state : S)
    (initial : domain.invariant state) (allowed : domain.admissible program state) :
    domain.invariant (run domain.step program state) := by
  induction program generalizing state with
  | nil => exact initial
  | cons op rest ih =>
    exact ih (domain.step op state)
      (domain.preserves op state initial allowed.1) allowed.2

/-- Successful execution preserves the invariant, assuming it holds initially. -/
theorem Domain.execute_preserves {Op : Type u} {S : Type v}
    (domain : Domain Op S) (program : List Op) (state result : S)
    (initial : domain.invariant state)
    (accepted : domain.execute program state = some result) :
    domain.invariant result := by
  obtain ⟨allowed, same⟩ := (domain.execute_iff program state result).mp accepted
  rw [← same]
  exact domain.admissible_preserves program state initial allowed

/-- Validate the initial invariant as well as every operation precondition.
Specification: neural_assemblies/ir/VERIFICATION.md#contract-checked-domain

The decision procedure is an explicit boundary dependency. Keeping it outside
`Domain` preserves domains whose invariant is not computationally decidable. -/
def Domain.checkedExecute {Op : Type u} {S : Type v} (domain : Domain Op S)
    (decideInvariant : (state : S) -> Decidable (domain.invariant state))
    (program : List Op) (state : S) : Option S :=
  if @decide (domain.invariant state) (decideInvariant state) then
    domain.execute program state
  else none

/-- The boundary accepts exactly valid initial states and admissible schedules,
with the same result as the ordinary interpreter. This includes completeness:
a boundary that rejects everything cannot satisfy this contract. -/
theorem Domain.checkedExecute_iff {Op : Type u} {S : Type v} (domain : Domain Op S)
    (decideInvariant : (state : S) -> Decidable (domain.invariant state))
    (program : List Op) (state result : S) :
    domain.checkedExecute decideInvariant program state = some result <->
      domain.invariant state /\ domain.admissible program state /\
        run domain.step program state = result := by
  letI := decideInvariant state
  by_cases initial : domain.invariant state
  case pos => simp [checkedExecute, initial, domain.execute_iff]
  case neg => simp [checkedExecute, initial]

/-- Unlike `execute_preserves`, callers need no separate initial-state premise. -/
theorem Domain.checkedExecute_preserves {Op : Type u} {S : Type v}
    (domain : Domain Op S)
    (decideInvariant : (state : S) -> Decidable (domain.invariant state))
    (program : List Op) (state result : S)
    (accepted : domain.checkedExecute decideInvariant program state = some result) :
    domain.invariant result := by
  obtain ⟨initial, allowed, same⟩ :=
    (domain.checkedExecute_iff decideInvariant program state result).mp accepted
  rw [← same]
  exact domain.admissible_preserves program state initial allowed

/- Constructed controls: bounded allocation, including invalid second steps.
The theorem is about this small domain, not actual assembly recruitment. -/
private def allocation : Domain Nat Nat where
  step amount used := used + amount
  invariant used := used ≤ 3
  valid amount used := used + amount ≤ 3
  decideValid _ _ := inferInstance
  preserves _ _ _ allowed := allowed

example : allocation.execute [1, 2] 0 = some 3 := by decide
example : allocation.execute [2, 2] 0 = none := by decide
example : run allocation.step [2, 2] 0 = 4 := by decide
example : ¬ allocation.invariant (run allocation.step [2, 2] 0) := by
  change ¬ (4 ≤ 3)
  decide

-- Boundary controls include an empty schedule on an invalid initial state:
-- the lower-level interpreter accepts it, the checked boundary rejects it.
private def allocationDecidable (used : Nat) :
    Decidable (allocation.invariant used) := inferInstanceAs (Decidable (used <= 3))

example : allocation.checkedExecute allocationDecidable [1, 2] 0 = some 3 := by decide
example : allocation.checkedExecute allocationDecidable [2, 2] 0 = none := by decide
example : allocation.checkedExecute allocationDecidable [] 3 = some 3 := by decide
example : allocation.checkedExecute allocationDecidable [] 4 = none := by decide
example : allocation.execute [] 4 = some 4 := by decide

#print axioms Domain.checkedExecute_iff
#print axioms Domain.checkedExecute_preserves
#print axioms Domain.execute_iff
#print axioms Domain.admissible_preserves
#print axioms Domain.execute_preserves

end AssemblyIR
