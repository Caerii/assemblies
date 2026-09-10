import AssemblyIR.Refinement

/-!
# Learning-mask frame contract

Specification: neural_assemblies/ir/VERIFICATION.md#contract-learning-frame

This kernel models a fixed set of fibers with a per-fiber learned value and
separate activity. It does not identify learned values with a sampled engine's
entire physical connectome: recruitment can allocate new base connectivity.
Concrete backends must prove that their representation satisfies this relation.
-/
namespace AssemblyIR

structure LearningState (Fiber Weight Activity : Type) where
  weights : Fiber -> Weight
  activity : Activity

variable {Fiber Weight Activity Op : Type}

/-- Apply an arbitrary proposed transition, restricting only its weight writes.
The proposal reads the original state, including every masked fiber's weight.
Its activity result is preserved; this is not a whole-state freeze. -/
def maskedStep
    (blocked : Fiber -> Bool)
    (update : Op -> LearningState Fiber Weight Activity -> LearningState Fiber Weight Activity)
    (op : Op) (state : LearningState Fiber Weight Activity) :
    LearningState Fiber Weight Activity :=
  let proposed := update op state
  { weights := fun fiber => if blocked fiber then state.weights fiber else proposed.weights fiber
    activity := proposed.activity }

/-- Frame property: a blocked fiber retains its value exactly. -/
theorem maskedStep_blocked
    (blocked : Fiber -> Bool)
    (update : Op -> LearningState Fiber Weight Activity -> LearningState Fiber Weight Activity)
    (op : Op) (state : LearningState Fiber Weight Activity) (fiber : Fiber)
    (held : blocked fiber = true) :
    (maskedStep blocked update op state).weights fiber = state.weights fiber := by
  simp [maskedStep, held]

/-- Non-vacuity: unblocked writes are exactly the proposed writes. -/
theorem maskedStep_allowed
    (blocked : Fiber -> Bool)
    (update : Op -> LearningState Fiber Weight Activity -> LearningState Fiber Weight Activity)
    (op : Op) (state : LearningState Fiber Weight Activity) (fiber : Fiber)
    (allowed : blocked fiber = false) :
    (maskedStep blocked update op state).weights fiber = (update op state).weights fiber := by
  simp [maskedStep, allowed]

/-- Activity still advances according to the transition on the original state. -/
theorem maskedStep_activity
    (blocked : Fiber -> Bool)
    (update : Op -> LearningState Fiber Weight Activity -> LearningState Fiber Weight Activity)
    (op : Op) (state : LearningState Fiber Weight Activity) :
    (maskedStep blocked update op state).activity = (update op state).activity := by
  rfl

/-- A fixed suppression scope preserves each blocked value across any schedule. -/
theorem masked_run_frame
    (blocked : Fiber -> Bool)
    (update : Op -> LearningState Fiber Weight Activity -> LearningState Fiber Weight Activity)
    (program : List Op) (state : LearningState Fiber Weight Activity) (fiber : Fiber)
    (held : blocked fiber = true) :
    (run (maskedStep blocked update) program state).weights fiber = state.weights fiber := by
  induction program generalizing state with
  | nil => rfl
  | cons op rest ih =>
    simp only [run]
    rw [ih]
    exact maskedStep_blocked blocked update op state fiber held

/-- Adding a nested suppression cannot release a fiber held by the outer scope. -/
theorem masked_scope_extension
    (outer inner : Fiber -> Bool)
    (update : Op -> LearningState Fiber Weight Activity -> LearningState Fiber Weight Activity)
    (program : List Op) (state : LearningState Fiber Weight Activity) (fiber : Fiber)
    (held : outer fiber = true) :
    (run (maskedStep (fun f => outer f || inner f) update) program state).weights fiber =
      state.weights fiber := by
  apply masked_run_frame
  simp [held]

-- Constructed controls: masking retains a learned value rather than zeroing it,
-- permits another fiber to learn, and does not freeze activity.
private def initial : LearningState Nat Nat Nat :=
  { weights := fun _ => 10, activity := 0 }

private def proposed (_ : Unit) (state : LearningState Nat Nat Nat) :
    LearningState Nat Nat Nat :=
  { weights := fun fiber => state.weights fiber + 1
    activity := state.activity + state.weights 0 }

private def heldZero : Nat -> Bool := fun fiber => fiber == 0

example : (maskedStep heldZero proposed () initial).weights 0 = 10 := by decide
example : (maskedStep heldZero proposed () initial).weights 1 = 11 := by decide
example : (maskedStep heldZero proposed () initial).activity = 10 := by decide
example : (run (maskedStep heldZero proposed) [(), ()] initial).weights 0 = 10 := by decide
example : (run (maskedStep heldZero proposed) [(), ()] initial).weights 1 = 12 := by decide
example : (run (maskedStep heldZero proposed) [(), ()] initial).activity = 20 := by decide
-- Bypassing suppression changes the protected value.
example : (proposed () initial).weights 0 != initial.weights 0 := by decide
-- Erasing the masked weight before computing the proposal loses its drive.
example : (proposed () { initial with weights := fun _ => 0 }).activity !=
    (maskedStep heldZero proposed () initial).activity := by decide

#print axioms maskedStep_blocked
#print axioms maskedStep_allowed
#print axioms maskedStep_activity
#print axioms masked_run_frame
#print axioms masked_scope_extension

end AssemblyIR
