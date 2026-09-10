/-!
# Assembly IR lowering obligations

A lowering may expand one source operation into several target instructions.
Preserving a relation at each instruction boundary suffices for preserving it
for an entire sequential program. Observations require a separate compatibility
obligation: matching storage representations alone is not scientific agreement.

This is a reusable proof rule, NOT an instantiation for Python/Rust/CUDA. In
particular it assumes the single-step simulation that those bridges must prove.
No floating-point, random-distribution, concurrency or exception semantics are
silently inferred. Encode such state/outcomes explicitly when instantiating it.
-/
namespace AssemblyIR

universe u v w x

/-- Sequential semantics; include RNG state and errors in `S` when relevant. -/
def run {Op : Type u} {S : Type v} (step : Op -> S -> S) : List Op -> S -> S
  | [], state => state
  | op :: rest, state => run step rest (step op state)

/-- Lower each instruction, preserving instruction order. -/
def lower {Op : Type u} {TargetOp : Type v}
    (emit : Op -> List TargetOp) (program : List Op) : List TargetOp :=
  program.flatMap emit

theorem run_append {Op : Type u} {S : Type v} (step : Op -> S -> S)
    (left right : List Op) (state : S) :
    run step (left ++ right) state = run step right (run step left state) := by
  induction left generalizing state with
  | nil => rfl
  | cons op rest ih => exact ih (step op state)

/-- A kernel reuses a domain's invariant-preservation obligation. -/
theorem run_preserves {Op : Type u} {S : Type v} (step : Op -> S -> S)
    (invariant : S -> Prop)
    (localProof : forall op state, invariant state -> invariant (step op state))
    (program : List Op) (state : S) (initial : invariant state) :
    invariant (run step program state) := by
  induction program generalizing state with
  | nil => exact initial
  | cons op rest ih => exact ih (step op state) (localProof op state initial)

/-- Successive compiler bridges compose without reconstructing schedules. -/
theorem lower_compose {A : Type u} {B : Type v} {C : Type w}
    (first : A -> List B) (second : B -> List C) (program : List A) :
    lower second (lower first program) =
      lower (fun op => lower second (first op)) program := by
  simp only [lower, List.flatMap_assoc]

/-- The local proof obligation a compiler bridge must discharge. -/
def Simulates {Op : Type u} {TargetOp : Type v} {S : Type w} {T : Type x}
    (sourceStep : Op -> S -> S) (targetStep : TargetOp -> T -> T)
    (emit : Op -> List TargetOp) (related : S -> T -> Prop) : Prop :=
  forall op source target, related source target ->
    related (sourceStep op source) (run targetStep (emit op) target)

/-- A local simulation composes across an arbitrarily long finite schedule. -/
theorem lower_preserves {Op : Type u} {TargetOp : Type v} {S : Type w} {T : Type x}
    (sourceStep : Op -> S -> S) (targetStep : TargetOp -> T -> T)
    (emit : Op -> List TargetOp) (related : S -> T -> Prop)
    (localProof : Simulates sourceStep targetStep emit related)
    (program : List Op) (source : S) (target : T)
    (initial : related source target) :
    related (run sourceStep program source) (run targetStep (lower emit program) target) := by
  induction program generalizing source target with
  | nil => exact initial
  | cons op rest ih =>
    simp only [lower, List.flatMap_cons, run_append, run]
    exact ih (sourceStep op source) (run targetStep (emit op) target)
      (localProof op source target initial)

/-- A common readout needs a proof too, independently of state simulation. -/
theorem observations_agree {Op : Type u} {TargetOp : Type v}
    {S : Type w} {T : Type x} {Observation : Type}
    (sourceStep : Op -> S -> S) (targetStep : TargetOp -> T -> T)
    (emit : Op -> List TargetOp) (related : S -> T -> Prop)
    (localProof : Simulates sourceStep targetStep emit related)
    (sourceRead : S -> Observation) (targetRead : T -> Observation)
    (readProof : forall s t, related s t -> sourceRead s = targetRead t)
    (program : List Op) (source : S) (target : T)
    (initial : related source target) :
    sourceRead (run sourceStep program source) =
      targetRead (run targetStep (lower emit program) target) :=
  readProof _ _ (lower_preserves sourceStep targetStep emit related localProof
    program source target initial)

/-- Equality of a visible count alone cannot certify hidden state restoration. -/
theorem visible_count_is_insufficient :
    Exists fun before : Prod Nat Nat => Exists fun after : Prod Nat Nat =>
      And (before.1 = after.1) (Not (before = after)) := by
  exact Exists.intro (0, 0) (Exists.intro (0, 1) (And.intro rfl (by decide)))

#print axioms run_preserves
#print axioms lower_compose
#print axioms lower_preserves
#print axioms observations_agree
#print axioms visible_count_is_insufficient

end AssemblyIR
